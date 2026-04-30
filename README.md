# OpenMPR

We benchmarked 190 open-source VLMs on the GroceryVision Challenge MPR track. The best models reach 94% Recall@5 on grocery product retrieval but only 77% Recall@1 (Maminta & Atienza, ICPR 2026). They narrow a 409-SKU catalog to a handful of candidates. Picking the right one from that shortlist is where they fail.

OpenMPR is the code and data behind that benchmark. Three things stood out:

1. Data quality beats scale. Moving from raw web scrapes to filtered training data delivers up to 16% Recall@1 gains. Doubling model parameters on noisy data delivers almost nothing.
2. Small can beat large. MobileCLIP-B (150M parameters) outperforms 350M competitors trained on unfiltered data.
3. The ranking gap is a geometry problem. Contrastive embeddings cluster product categories well but collapse visually similar SKU variants into nearly identical vectors (about 4 degrees apart in embedding space). That geometry failure is what the 17.5-point gap between Recall@5 and Recall@1 measures.

Detailed results are in our ICPR 2026 paper.

---

## Team

- **Emmanuel G. Maminta**, AI Ph.D. student, Ubiquitous Computing Laboratory, University of the Philippines. [[homepage]](https://egmaminta.github.io/about/)
- **Rowel O. Atienza**, Professor, EEEI, University of the Philippines; research adviser. [[homepage]](https://roatienza.github.io/)

---

## What is MPR

MPR frames product recognition as a ranking problem. Given a probe image and a text catalog, a VLM maps both into a shared embedding space and sorts candidates by cosine similarity. No retraining per catalog. Just nearest-neighbor search at inference time.

The difficulty is fine-grained discrimination. Two Campbell's soup cans share nearly every visual attribute. Standard benchmarks like ImageNet treat this as one class. Retail systems need to distinguish barcodes.

See [docs/HOW-TO-MPR.md](docs/HOW-TO-MPR.md) for the full task formulation, metric definitions, and an explanation of the manifold collapse problem.

---

## Getting started

**Prerequisites:** Python 3.11, [uv](https://docs.astral.sh/uv/) installed.

```bash
git clone https://github.com/upeee/openmpr.git
cd openmpr

uv venv --python 3.11
source .venv/bin/activate    # Windows: .venv\Scripts\activate
uv pip install -r requirements.txt
```

To run the benchmark:

```bash
python src/benchmark/eval_openclip.py
```

Results are written per model to `src/benchmark/results/`. The script iterates all available OpenCLIP checkpoints, skipping those that fail to load, and saves macro and micro Recall@K as JSON files.

---

## Repository structure

```
src/
  benchmark/
    eval_openclip.py           # Main evaluation script (190 models)
    results/                   # Per-model Recall@K outputs (not tracked)
  data/
    barcode_to_images_map.json # Probe image paths per SKU
    product_texts.json         # Catalog descriptions
    synthetic/                 # LLM-compressed descriptions (≤77 tokens)
  analysis/                    # Notebooks for result analysis
docs/
  HOW-TO-MPR.md                # Task formulation, metrics, the gap
  DATASETS.md                  # Datasets for MPR research
  LITERATURE.md                # Curated reference list
```

---

## Citation

```bibtex
@inproceedings{maminta2026openmpr,
  title     = {What Matters for Grocery Product Retrieval with Open Source Vision Language Models},
  author    = {Maminta, Emmanuel G. and Atienza, Rowel O.},
  booktitle = {Proceedings of the International Conference on Pattern Recognition (ICPR)},
  year      = {2026}
}
```

---

## License

Code is released under the Apache 2.0 License. The GroceryVision dataset is subject to CC-BY-NC 4.0, which permits non-commercial use with attribution.
