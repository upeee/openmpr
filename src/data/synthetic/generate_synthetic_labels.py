import transformers
import torch
import re
import json
from tqdm import tqdm
import yaml


# Global pipeline variable to avoid reloading the model multiple times
print("Loading model...")
model_name = "meta-llama/Meta-Llama-3.1-8B-Instruct"
pipeline = transformers.pipeline(
    "text-generation",
    model=model_name,
    model_kwargs={"dtype": torch.bfloat16},
    device_map="auto",
)
print("Model loaded.")

# Load product texts from a JSON file
with open("src/data/paths.yaml", "r") as f:
    dataset_paths = yaml.safe_load(f)

dataset_paths = dataset_paths.get("mpr_dataset", {})

if not dataset_paths:
    raise ValueError("Dataset paths for 'mpr_dataset' not found in paths.yaml")
else:
    path_to_product_texts = dataset_paths.get("path_to_labels", "")

    if not path_to_product_texts:
        raise ValueError("'path_to_product_texts' not found in dataset paths")

with open(path_to_product_texts, "r") as f:
    product_texts = json.load(f)

def generate_description(
    system_prompt: str,
    product_desc: str,
    tag_desc: str
) -> str:
    """
    Generate a synthetic product description using an instruction-tuned language model.

    Args:
        system_prompt (str): The system prompt to guide the model.
        product_desc (str): The original product description.
        tag_desc (str): The tag description to be included.
    Returns:
        str: The generated product description.
    """
    user_input = f"PRODUCT_DESCRIPTION: {product_desc}\nTAG_DESCRIPTION: {tag_desc}\n"
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_input}
    ]
    outputs = pipeline(messages)

    return outputs[0]['generated_text'][-1]['content']

system_prompt = ("[[ ## role ## ]]\n"
                 "You are a helpful assistant that generates descriptions for grocery products based on their product and tag descriptions.\n\n"
                 "[[ ## context ## ]]\n"
                 "The original descriptions are noisy, lengthy, and inconsistent in format. "
                 "Your generated descriptions will be used for a multi-product retrieval task, so they should be concise, informative, and relevant to the product. "
                 "The original descriptions are provided in the following format:\n"
                 "PRODUCT_DESCRIPTION: <product_description>\n"
                 "TAG_DESCRIPTION: <tag_description>\n\n"
                 "[[ ## task ## ]]\n"
                 "Generate a concise and informative product description in less than or exactly 77 tokens. "
                 "The expected output must be a JSON object with a single key \"label\" and the value being the generated description.\n\n"
                 "[[ ## expected output ## ]]\n"
                 "```json\n"
                 "{\"label\": \"<generated description>\"}\n"
                 "```\n\n"
                 "[[ ## constraints ## ]]\n"
                 "- Start with \"The product is ...\" in every description.\n"
                 "- The tag description is more important because it is focused on specific visual attributes or features of the product that are prominent in images.\n\n"
                 "[[ ## example ## ]]\n"
                 "INPUT:\n"
                 "PRODUCT_DESCRIPTION: 'The image shows a can of Campbell\'s Condensed Vegetable Beef Soup. The can is predominantly red with white and yellow text. "
                 "The Campbell\'s logo is at the top, and the product name is prominently displayed in the center. "
                 "Below the product name, there is an image of the soup inside the can, which includes visible pieces of vegetables and meat.'\n"
                 "TAG_DESCRIPTION: 'The image shows a can of Campbell\'s Condensed Vegetable Beef Soup. "
                 "The can is predominantly red with white and yellow text. The Campbell\'s logo is at the top in white script, and below it, the words \"CONDENSED SOUP\" are written in smaller white letters. "
                 "The product name \"Vegetable Beef\" is in bold white letters, and below it, the word \"Soup\" is in a smaller font. "
                 "There is a visual representation of the soup inside the can, showing a mix of vegetables and beef in a broth.'\n\n"
                 "OUTPUT:\n"
                 "```json\n"
                 "{\"label\": \"The product is \"Campbell's Beef Vegetable Soup\" which comes in a can that is predominantly red with white and yellow text.\"}\n"
                 "```\n\n")

new_product_texts = []

product_id_descriptions = list(product_texts.items())

for product_id, product_info in tqdm(product_id_descriptions):
    try:
        tag_description = product_info.get("describe the tag of this product.", "")
    except AttributeError:
        raise ValueError(f"Product description for product ID {product_id} is not a dictionary.")

    try:
        product_description = product_info.get("describe the product in the image?", "")
    except AttributeError:
        raise ValueError(f"Tag description for product ID {product_id} is not a dictionary.")

    new_description = generate_description(
        system_prompt,
        product_description,
        tag_description
    )

    match = re.search(r'(?:```json\n)?\{"label": "(.*?)"\}\n?(?:```)?', new_description, re.DOTALL)

    if match:
        label = match.group(1)
        new_product_texts.append({
            "product_id": product_id,
            "label": label
        })
    else:
        raise ValueError(f"Failed to extract JSON from model output for product ID {product_id}. Output was: {new_description}")

print(f"Generated {len(new_product_texts)} new product descriptions.")
output_file_path = f"src/data/synthetic/synthetic_labels_{model_name.split('/')[-1].lower()}.json"

with open(output_file_path, "w") as f:
    json.dump(new_product_texts, f, indent=4)

synthetic_data_path = dataset_paths.get("synthetic", {})
if not synthetic_data_path:
    raise ValueError("Dataset paths for 'synthetic' not found in paths.yaml")
else:
    synthetic_data_path["path_to_synthetic_labels"] = output_file_path

    # Write back to paths.yaml
    with open("src/data/paths.yaml", "w") as f:
        yaml.dump({"mpr_dataset": dataset_paths.get("mpr_dataset", {}),
                    "synthetic": synthetic_data_path}, f)

    print(f"Updated paths.yaml with synthetic labels path: {output_file_path}")