# How multimodal product retrieval works

MPR is recognition reframed as ranking. Instead of training a classifier for every SKU, you map probe images and catalog descriptions into a shared embedding space, then sort by similarity. New products get added by updating the catalog. The model never retrains.

The upside: zero-shot generalization across catalogs. The downside: visually similar products land close together in embedding space, and contrastive training gives you no way to separate them at inference time. That problem is what this benchmark is about.

---

## Task formulation

Given:
- A probe image $v$ from a query product
- A catalog of $N$ text descriptions $\{t_i\}_{i=1}^N$, one per SKU

Produce:
- A ranked list of catalog entries by relevance to $v$

The model scores each image-text pair with cosine similarity:

$$s(v, t) = \frac{f_\theta(v) \cdot g_\phi(t)}{\|f_\theta(v)\| \cdot \|g_\phi(t)\|}$$

where $f_\theta$ is the image encoder and $g_\phi$ is the text encoder. Both produce $L_2$-normalized embeddings on a shared hypersphere.

The GroceryVision evaluation uses a **single-gallery-shot protocol**: each of the 409 SKUs has exactly one catalog description. The task is to rank those 409 entries for each probe image. No fine-tuning, no few-shot examples.

---

## How the models are trained (CLIP objective)

OpenMPR evaluates models trained on the CLIP contrastive objective. During training, the model sees batches of matched image-text pairs $\{(v_i, t_i)\}_{i=1}^B$ and learns to push matched pairs together while pulling mismatched pairs apart.

The loss is InfoNCE, applied symmetrically:

$$\mathcal{L}_{I2T} = -\frac{1}{B} \sum_{i=1}^B \log \frac{e^{s(v_i, t_i)/\tau}}{\sum_{j=1}^B e^{s(v_i, t_j)/\tau}}$$

$$\mathcal{L} = \frac{1}{2}(\mathcal{L}_{I2T} + \mathcal{L}_{T2I})$$

$\tau$ is a learnable temperature. Lower $\tau$ sharpens the softmax and amplifies gradients from hard negatives.

SigLIP variants replace the softmax with sigmoid, treating each image-text pair as an independent binary classification:

$$\mathcal{L}_{\text{SigLIP}} = -\frac{1}{B} \sum_{i,j} \log \sigma\left(y_{ij} \cdot s(v_i, t_j)\right)$$

where $y_{ij} = +1$ for matched pairs and $-1$ for unmatched. At inference, we use `torch.sigmoid` for SigLIP models and `torch.softmax` for CLIP models.

---

## Metric: Recall@K

The primary metric is Recall@K (equivalently, CMC@K, Cumulative Matching Characteristics at rank K). It measures the fraction of queries where the correct catalog entry appears in the top-K results:

$$\text{Recall@}K = \frac{1}{Q} \sum_{q=1}^Q \mathbb{I}(\text{rank}(gt_q) \le K)$$

where $Q$ is the total number of probe images and $gt_q$ is the ground-truth description for query $q$.

**Macro vs. micro Recall@K:**

- **Micro Recall@K**: counts correct retrievals across all probe images uniformly. SKUs with more images contribute more to the score. This is the primary metric used in the ICPR 2026 paper.
- **Macro Recall@K**: averages per-SKU accuracy first, then averages across SKUs. Gives equal weight to each product regardless of how many images it has. Useful for detecting whether rare or hard products are being systematically missed.

The GroceryVision dataset has 20,565 front-facing images across 409 SKUs (17,516 front-view, plus 3,049 front-drop for SKUs lacking front-view frames), roughly 50 images per product on average. All released metrics are computed over this 20,565-image manifest, shipped in `src/data/barcode_to_images_map.json`.

---

## Why Recall@5 is easy and Recall@1 is hard

The best model evaluated (ViT-gopt-16-SigLIP2-384) achieves:
- Recall@5: 94.5%
- Recall@1: 77.0%

That 17.5% gap is consistent across architectures and scales. It is not a data problem or a model size problem. It is a geometry problem.

### Manifold collapse

Contrastive pretraining on broad web-crawled image-text pairs trains models to separate semantic categories: soup from soda, shampoo from coffee. It does not train them to separate *Chicken Corn Chowder* from *Chicken & Dumplings* — two Campbell's products sharing the same red can, similar typography, and nearly identical visual attributes.

In embedding space, these two products land ~4° apart. Cosine similarity cannot distinguish between vectors that close. The correct SKU ends up at rank 2, not rank 1.

This is manifold collapse: the embedding manifold for a product category collapses into a narrow cone where inter-SKU angular distance is below the resolution of dot-product ranking.

The implication: the bottleneck is not representational capacity but the ranking mechanism itself. Retrieval at the category level works. Ranking within the category fails.

---

## Catalog description quality

CLIP text encoders have a 77-token context limit. Original GroceryVision catalog descriptions frequently exceed this, causing truncation of discriminative attributes (color, size, brand-specific terminology).

To address this, we used Llama-3.1-8B-Instruct to compress descriptions to ≤77 tokens while preserving key visual attributes. The prompt enforces outputs beginning with "The product is..." and prioritizes color, shape, brand, size, and packaging. On a random sample of 250 descriptions (61% of the 409-SKU catalog), we observed 100% token compliance and no hallucinations.

Synthetic description generation files are in `src/data/synthetic/`.

---

## Efficiency metric: semantic power density (φ)

Standard accuracy-vs-parameters plots treat the relationship as linear. Deployment utility is not linear; it follows a sigmoid curve anchored at 50% accuracy.

Below 50% Recall@1, a model produces more false retrievals than correct ones (odds ratio < 1). It cannot run autonomously. Above 80%, errors are rare enough to tolerate. We anchor the efficiency metric at 50%, the minimum viability threshold.

Treating retrieval accuracy as a signal amplitude and borrowing from signal processing:

$$\phi = \frac{\left(\dfrac{\text{Recall@1}}{1 - \text{Recall@1} + \epsilon}\right)^2}{N_{\text{params}}} \times 100$$

with $\epsilon = 10^{-6}$, Recall@1 the micro Recall@1, and $N_{\text{params}}$ the parameter count **in millions**. The squared SNR term grows super-linearly above the 50% threshold, penalizing sub-threshold models regardless of parameter efficiency. `scripts/compute_phi.py` recomputes $\phi$ for all 190 models from `results/benchmark_summary.csv`.

MobileCLIP-B (DataCompDR, 150M) achieves $\phi = 2.37$, the highest among models with Recall@1 > 65%; the smaller MobileCLIP-S1 reaches $\phi = 2.82$ at lower absolute accuracy (60.8% Recall@1). A 1.9B-parameter SigLIP2 model achieves only $\phi = 0.60$ despite higher absolute accuracy.

For edge deployment: prioritize Recall@K for coverage, then validate $\phi$ for efficiency.
