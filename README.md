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

## Reproducing the benchmark

### Requirements

- Linux with an NVIDIA GPU and a CUDA 12.x driver. CPU-only execution is not supported (inference runs under CUDA autocast). We used a single A100-SXM4-40GB.
- Python 3.11 and [uv](https://docs.astral.sh/uv/) (plain `pip` works too).
- Disk space: ~32 GB for the dataset (8 GB archive + 24 GB extracted), ~0.4 GB of model weights for the quick verification below, and up to ~0.5 TB of Hugging Face cache for the full 190-model sweep.

Exact library versions are pinned in `requirements.txt` (`torch 2.8.0` CUDA 12.8 build, `open_clip_torch 3.1.0`, …).

### Install

```bash
git clone https://github.com/upeee/openmpr.git
cd openmpr

uv venv --python 3.11
source .venv/bin/activate
uv pip install -r requirements.txt
```

The pinned `torch==2.8.0` wheel on PyPI is the CUDA 12.8 build on Linux; no extra index or flags are needed. Plain `pip install -r requirements.txt` works too.

### Get the data

```bash
bash scripts/download_data.sh
```

This downloads the GroceryVision MPR dataset (8 GB, CC-BY-NC 4.0) and extracts it to `data/mpr_challenge/`. Expected layout:

```
data/mpr_challenge/
  appearance_based/
    single_frame_front_view/   # <barcode>_<n>.jpg probe images
    single_frame_front_drop/
    ...
  product_texts.json
```

Already have the dataset? Skip the download and point `mpr_dataset.path_to_images` in `src/data/paths.yaml` at your extracted `appearance_based/` directory.

### Quick verification (one model)

Evaluate a single checkpoint and compare it against the published reference results:

```bash
python src/benchmark/eval_openclip.py --model RN50 --pretrained openai
python scripts/compare_results.py
```

All 20,565 probe images are ranked against the 409-SKU catalog. Expected micro Recall for `RN50/openai`: **@k=1: 40.48%, @k=3: 58.20%, @k=5: 63.71%**. Takes about 12 minutes on an A100 (the first run also downloads ~0.4 GB of weights).

### Full sweep (all models)

```bash
python src/benchmark/eval_openclip.py
```

Iterates every checkpoint listed by `open_clip.list_pretrained()` — 192 under the pinned `open_clip_torch 3.1.0`, of which 2 fail to download and are skipped (recorded in `results/openclip_skipped_models.json`). Per-model micro/macro Recall@K JSONs are written to `src/benchmark/results/`. Expect a multi-day run on a single GPU.

### Reference results

The results reported in the paper are tracked in [`results/`](results/README.md): `benchmark_summary.csv` (one row per model) plus per-model micro/macro Recall@K JSONs, produced on an A100-SXM4-40GB with bfloat16. All published metrics are computed over the 20,565-image evaluation manifest shipped in `src/data/barcode_to_images_map.json`.

Your reruns write to `src/benchmark/results/` and never overwrite the reference. On the same GPU generation results match exactly; on other hardware a few rank flips out of 20,565 images are possible from floating-point differences, which the compare script's default tolerance (0.005) absorbs.

---

## Repository structure

```
src/
  benchmark/
    eval_openclip.py           # Main evaluation script (190 models)
    results/                   # Your reruns land here (not tracked)
  data/
    barcode_to_images_map.json # Evaluation manifest: relative image paths + label per SKU
    paths.yaml                 # Dataset location config
    product_texts.json         # Original catalog descriptions
    synthetic/                 # LLM-compressed descriptions (≤77 tokens) used as labels
  analysis/                    # Notebook reproducing the paper's analysis figures
results/                       # Published reference results (see results/README.md)
scripts/
  download_data.sh             # Dataset download + extraction
  compare_results.py           # Compare a rerun against the reference results
data/                          # Dataset lives here after download (not tracked)
docs/
  HOW-TO-MPR.md                # Task formulation, metrics, the gap
  DATASETS.md                  # Datasets for MPR research
  LITERATURE.md                # Curated reference list
```

The analysis notebook (`src/analysis/analysis.ipynb`) reproduces the paper's figures from `results/`; it needs `pandas`, `matplotlib`, and `jupyter` on top of `requirements.txt` and is not required for reproducing the benchmark numbers.

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
