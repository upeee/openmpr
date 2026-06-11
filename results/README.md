# Reference results

These are the results reported in the ICPR 2026 paper, provided so reviewers and
users can compare their own runs against ours without re-evaluating all 190 models.

**Provenance.** Produced by `src/benchmark/eval_openclip.py` on a single NVIDIA
A100-SXM4-40GB (CUDA 12.8), Python 3.11, `torch 2.8.0+cu128`,
`open_clip_torch 3.1.0`, bfloat16 inference. Of the 192 checkpoints listed by
`open_clip.list_pretrained()` at evaluation time (September 2025), 190 were
evaluated; the 2 failures (weight-download errors) are recorded in
`openclip_skipped_models.json`.

All metrics are computed over the 20,565-image evaluation manifest in
`src/data/barcode_to_images_map.json` (409 SKUs). Micro values are exact
multiples of 1/20565.

## Contents

- `micro_cmc/<model>--<pretrained>.json` — micro Recall@{1,3,5} (every probe image weighted equally). The primary metric in the paper.
- `macro_cmc/<model>--<pretrained>.json` — per-SKU Recall@{1,3,5} (each SKU's accuracy computed over its own images).
- `benchmark_summary.csv` — one row per model. Columns:
  - `vision_model`, `pretrained_on` — OpenCLIP architecture and pretraining tag.
  - `nparams` — total parameter count.
  - `macro_cmc@{5,3,1}` — macro Recall@K averaged across the 409 SKUs.
  - `micro_cmc@{5,3,1}` — micro Recall@K over all 20,565 images.
  - `size` — parameter-count bucket (small/medium/large/massive).
  - `architecture_family` — e.g. ResNet, ViT, ConvNeXt.
  - `performance_to_size_ratio` — semantic power density φ (see docs/HOW-TO-MPR.md).
- `openclip_model_nparams.json` — parameter counts per checkpoint.
- `openclip_skipped_models.json` — the 2 checkpoints that failed to load, with errors.

## Comparing your run against these

Reruns write to `src/benchmark/results/` and never overwrite this directory. After
running one or more models, compare with:

```bash
python scripts/compare_results.py
```

GPU inference is not bit-deterministic (CUDA kernel selection varies across runs
and hardware), so expect small deviations: re-running `RN50/openai` on the same
A100 model reproduced these micro Recall@K values within at most 16 flipped ranks
out of 20,565 images (±0.08 percentage points). The compare script checks micro
Recall@K and per-SKU macro means against a 0.005 tolerance.
