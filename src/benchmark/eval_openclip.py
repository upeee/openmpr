import torch
import open_clip
from tqdm import tqdm
import json
from PIL import Image


skipped_models = []

eval_dataset_path = "src/data/barcode_to_images_map.json"

with open(eval_dataset_path, "r") as f:
    main_catalogue = json.load(f)

print(f"Loaded {len(main_catalogue)} barcodes from the evaluation dataset.")

labels = [main_catalogue[barcode]["label"] for barcode in main_catalogue.keys()]

all_openclip_models = open_clip.list_pretrained()

model_nparams = {}

for model_, pretrain_tag in tqdm(all_openclip_models, desc="Running through OpenCLIP models"):
    try:
        model_name = f"open_clip/{model_}--{pretrain_tag}"
        device = "cuda" if torch.cuda.is_available() else "cpu"

        if device == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA is not available.")
        elif device == "cpu" and torch.cuda.is_available():
            print("Warning: CUDA is available but not being used. This may lead to slow performance.")
            print("Setting device to 'cuda'.")
            device = "cuda"
    
        print(f"Evaluating model: {model_name} on device: {device}")

        torch.cuda.empty_cache()
        # Model nparams
        model, _, preprocess = open_clip.create_model_and_transforms(model_, pretrained=pretrain_tag)
        model.eval()

        tokenizer = open_clip.get_tokenizer(model_)

        nparams = sum(p.numel() for p in model.parameters())
        model_nparams[model_name] = nparams


        for param in model.parameters():
            param.data = param.data.to(torch.bfloat16 if device == "cuda" else torch.float16)

        assert next(model.parameters()).dtype in [torch.bfloat16, torch.float16], "Model parameters are not in the expected dtype."
        
        model.to(device)

        print("-" * 50)
        print(f"model name: {model_name}")
        print(f"model device: {next(model.parameters()).device}")
        print(f"model dtype: {next(model.parameters()).dtype}")
        print(f"model nparams: {nparams:,}")
        print("-" * 50)

        macro_CMC = {}
        # key = barcode, value = dict(keys: @k, value: avg. macro CMC @k)
        micro_CMC = {}
        # key = @k, value = avg. micro CMC @k

        grand_total_n_images = 0
        micro_CMC = {"@k=1": 0, "@k=3": 0, "@k=5": 0}

        print(f"Starting evaluation for model: {model_name}")
        encode_text_only_once = True
        for barcode in tqdm(main_catalogue.keys(), leave=False):
            images_per_product = main_catalogue[barcode]["image_paths"]
            correct_label = main_catalogue[barcode]["label"]

            macro_CMC[barcode] = {}

            macro_CMC[barcode]["@k=5"] = 0
            macro_CMC[barcode]["@k=3"] = 0
            macro_CMC[barcode]["@k=1"] = 0

            total_n_images_per_product = len(images_per_product)
            grand_total_n_images += total_n_images_per_product

            for image_path in tqdm(images_per_product, leave=False):
                image = preprocess(Image.open(image_path).convert("RGB")).unsqueeze(0)
                text = tokenizer(labels)
                
                image = image.to(device, dtype=torch.bfloat16 if device == "cuda" else torch.float16)
                text = text.to(device)

                with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16 if device == "cuda" else torch.float16):
                    image_features = model.encode_image(image)

                    if encode_text_only_once:
                        text_features = model.encode_text(text)
                        text_features /= text_features.norm(dim=-1, keepdim=True)
                        encode_text_only_once = False
                
                    image_features /= image_features.norm(dim=-1, keepdim=True)
                    text_probs = (100.0 * image_features @ text_features.T)
                
                if ("siglip" in model_name) or ("SigLIP" in model_name):
                    probs = torch.sigmoid(text_probs)
                else:
                    probs = text_probs.softmax(dim=-1)
                
                most_likely_indices = torch.topk(probs, k=5).indices[0].tolist()
                
                if labels.index(correct_label) in most_likely_indices:
                    
                    # @k=1
                    if labels.index(correct_label) == most_likely_indices[0]:
                        micro_CMC["@k=1"] += 1
                        micro_CMC["@k=3"] += 1
                        micro_CMC["@k=5"] += 1

                        macro_CMC[barcode]["@k=1"] += 1
                        macro_CMC[barcode]["@k=3"] += 1
                        macro_CMC[barcode]["@k=5"] += 1
                    
                    # @k=3
                    if labels.index(correct_label) in most_likely_indices[:3] and \
                        labels.index(correct_label) != most_likely_indices[0]:
                        micro_CMC["@k=3"] += 1
                        micro_CMC["@k=5"] += 1

                        macro_CMC[barcode]["@k=3"] += 1
                        macro_CMC[barcode]["@k=5"] += 1

                    if labels.index(correct_label) in most_likely_indices[3:5]:
                        micro_CMC["@k=5"] += 1
                        macro_CMC[barcode]["@k=5"] += 1
            
            macro_CMC[barcode]["@k=1"] /= total_n_images_per_product
            macro_CMC[barcode]["@k=3"] /= total_n_images_per_product
            macro_CMC[barcode]["@k=5"] /= total_n_images_per_product
        
        # Normalize micro CMC
        micro_CMC["@k=1"] /= grand_total_n_images
        micro_CMC["@k=3"] /= grand_total_n_images
        micro_CMC["@k=5"] /= grand_total_n_images

        print("*" * 50)
        print(f"Finished evaluation for model: {model_name}")
        print(f"Micro CMC summary: @k=1: {micro_CMC['@k=1']*100:.2f}%, @k=3: {micro_CMC['@k=3']*100:.2f}%, @k=5: {micro_CMC['@k=5']*100:.2f}%")
        print(f"Micro CMC file saved to: src/benchmark/results/micro_cmc/{model_name.split('/')[-1]}.json")
        print(f"Macro CMC file saved to: src/benchmark/results/macro_cmc/{model_name.split('/')[-1]}.json")
        print("*" * 50)

        with open(f"src/benchmark/results/micro_cmc/{model_name.split('/')[-1]}.json", "w") as f:
            json.dump(micro_CMC, f, indent=4)
        with open(f"src/benchmark/results/macro_cmc/{model_name.split('/')[-1]}.json", "w") as f:
            json.dump(macro_CMC, f, indent=4)
    except Exception as e:
        print(f"Skipping model {model_name} due to error: {e}")
        skipped_models.append({
            "model_name": model_name,
            "error": str(e)
        })
        continue

print("Evaluation completed for all models.")
print(f"Skipped {len(skipped_models)} models due to errors.")

with open("src/benchmark/results/openclip_skipped_models.json", "w") as f:
    json.dump(skipped_models, f, indent=4)

with open("src/benchmark/results/openclip_model_nparams.json", "w") as f:
    json.dump(model_nparams, f, indent=4)