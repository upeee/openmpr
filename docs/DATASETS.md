# Datasets

This page covers datasets relevant to multimodal product retrieval (MPR) research — both benchmarks for evaluation and pretraining corpora that explain why certain models perform the way they do. Dataset quality matters as much as model architecture, and the OpenMPR findings make this concrete.

---

## Evaluation / benchmark datasets

### GroceryVision Challenge (2025)

The dataset used in OpenMPR. 74,200 training images across 409 SKUs, subsampled to 12,944 front-facing perspectives for evaluation. Evaluation follows a single-gallery-shot protocol: one text description per SKU, no query-side images in the gallery. This makes the task harder than multi-shot retrieval — the model has no visual reference to fall back on. License: CC-BY-NC 4.0. Organized by the GroceryVision Challenge 2025.

### MIMEX (Tur et al., 2024)

15,277 images across 28 fine-grained product categories from real micro-market environments, split into 10,357 train and 4,920 test images. Categories include chocolates (Milka, Kinder, Toblerone, Rocher), snacks (Lays, Pringles), beverages (Red Bull, Monster, Sanpellegrino), and personal care items (Loreal, Dove, Colgate). Designed for zero-shot product classification with VLMs, making it directly comparable to the GroceryVision MPR protocol. Unlike GroceryVision, which draws from grocery store shelf images, MIMEX covers micro-market shelving where products are small and densely packed. The accompanying paper benchmarks CLIP and DINOv2 ensembles on this data. MIT license. [HuggingFace](https://huggingface.co/datasets/Anilot/MIMEX)

### Products-10K

Approximately 10,000 SKUs from JD.com, covering a wide range of retail product categories. One of the larger product recognition benchmarks available. Introduced by Bai et al. (2020). Useful for cross-dataset generalization studies, though the catalog scope (Chinese e-commerce) differs from Western grocery retail.

### RP2K

Approximately 13,000 products from Pinduoduo, captured as real shelf images from retail stores. Challenging viewpoint and occlusion conditions make it harder than studio-shot benchmarks. Introduced by Peng et al. (2020). Good for testing robustness to in-the-wild capture conditions rather than controlled product photography.

### Freiburg Groceries Dataset

5,000 images across 25 grocery product classes, captured in real supermarkets. Released in 2016, so it predates most of the VLM era. Useful as a sanity check or baseline comparison when you want a small, well-understood evaluation set. Open license.

### Amazon Berkeley Objects (ABO)

Approximately 147,702 product listings from Amazon, with 3D models and multiple product images per listing. The multi-view and 3D structure makes it more relevant for product understanding research than standard single-image retrieval benchmarks. Released by Collins et al. (2022). Not a natural fit for single-gallery-shot MPR evaluation, but worth knowing for multi-view or shape-based work.

### MVTec AD

5,354 images across 15 object and texture categories, released by Bergmann et al. (2019). The primary use case is anomaly detection in industrial inspection, not product retrieval. Included here because fine-grained visual discrimination under controlled conditions is directly relevant to understanding what contrastive embeddings do and do not preserve. If you are studying the discriminative gap between Recall@1 and Recall@5, MVTec AD provides a useful reference point for what "hard negative" visual similarity looks like.

---

## Pretraining datasets (context for model selection)

### WebLI (Web Language Image)

Google's filtered web-scraped dataset, used to pretrain SigLIP and PaLI. Approximately 10 billion image-text pairs after filtering via PaLI captions. The top-performing models in OpenMPR were pretrained on WebLI. The filtering step matters — raw web data at this scale would not produce the same results.

### DataComp-XL

1.28 billion image-text pairs from CommonPool with aggressive filtering. Open benchmark from Gadre et al. (2023) for studying what makes training data effective. Models trained on DataComp-XL consistently outperform their LAION-2B counterparts on MPR. The filtering is the differentiator, not the scale.

### MetaCLIP-5.4B

CommonCrawl data curated to match CLIP's original training distribution, from Xu et al. (2023). PE-Core, one of the top performers in OpenMPR, was trained on this. The curation strategy — matching concept frequency distributions rather than applying heuristic filters — appears to produce training data that transfers well to fine-grained retrieval tasks.

### WIT-400M (Web Image Text)

The original CLIP training data from Radford et al. (2021). 400 million image-text pairs. Serves as a useful baseline in scaling comparisons and a reasonable midpoint between small curated corpora and raw web scale.

### DataCompDR

The distillation-focused variant of DataComp used for MobileCLIP. Combines multi-modal reinforcement with synthetic captions and ensemble distillation. Despite a smaller effective training size than most large-scale corpora, MobileCLIP-B trained on DataCompDR outperforms 350M-parameter models trained on LAION-2B for MPR. This makes it practically relevant for deployment-constrained settings.

### LAION-2B

2.32 billion image-text pairs, raw web scale, used by most OpenCLIP ViT-L models. High noise from misaligned image-text pairs is the persistent problem. Models trained on LAION-2B perform consistently below DataComp-XL counterparts on MPR. Scale without filtering does not compensate for the noise.

### CommonPool

The unfiltered source pool from DataComp. Small ViTs (ViT-B-32) collapse on it, losing roughly 79% relative Recall@1 versus the WIT-400M baseline. Large ViTs (ViT-L) survive the noise better. Useful for studying noise tolerance as a function of model capacity.

### YFCC-15M

A 15-million image subset of YFCC100M (Yahoo Flickr Creative Commons), drawn from user-uploaded Flickr photos with noisy metadata. Models trained on YFCC-15M collapse on MPR — Recall@1 drops more than 93% relative to WIT-400M. Included here as a negative example. The failure is instructive: user-generated photo metadata is not a reliable signal for product-level semantic alignment.

---

## What the data findings mean in practice

For MPR, pretraining data quality dominates model size. A smaller model trained on WebLI or DataComp-XL will outperform a larger model trained on LAION-2B or CommonPool. This means the standard practice of selecting the largest available OpenCLIP checkpoint is wrong for this task — you should check the pretraining corpus first. The discriminative gap between Recall@1 and Recall@5 (~18 percentage points in the OpenMPR benchmark) persists across all architectures and scales, but it is meaningfully narrower for models with high-quality pretraining data.
