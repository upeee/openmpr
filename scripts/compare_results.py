"""Compare a local benchmark run against the published reference results.

Run from the repository root after evaluating one or more models:

    python src/benchmark/eval_openclip.py --model RN50 --pretrained openai
    python scripts/compare_results.py

Every model JSON present in both --candidate and --reference is compared at the
JSON level. Exits non-zero if any metric differs by more than --tolerance.
"""

import argparse
import json
import os
import sys


def metrics_of(obj):
    """Reduce a result JSON to {@k: value}.

    Micro files already are {@k: value}. Macro files are
    {barcode: {@k: value}} with per-SKU denominators as small as a few
    images, so a single rank flip moves one leaf by 1/num_images; the
    quantity the paper reports is the mean across SKUs, so that is what
    gets compared against the tolerance.
    """
    if any(isinstance(v, dict) for v in obj.values()):
        ks = next(iter(obj.values())).keys()
        return {k: sum(per_sku[k] for per_sku in obj.values()) / len(obj) for k in ks}
    return dict(obj)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--candidate", default="src/benchmark/results", help="Directory with your run's outputs")
    parser.add_argument("--reference", default="results", help="Directory with the published reference results")
    parser.add_argument("--tolerance", type=float, default=0.005, help="Maximum allowed absolute difference per metric")
    args = parser.parse_args()

    failures = 0
    compared = 0
    for kind in ("micro_cmc", "macro_cmc"):
        cand_dir = os.path.join(args.candidate, kind)
        ref_dir = os.path.join(args.reference, kind)
        if not os.path.isdir(cand_dir):
            continue
        for name in sorted(os.listdir(cand_dir)):
            if not name.endswith(".json"):
                continue
            ref_path = os.path.join(ref_dir, name)
            if not os.path.isfile(ref_path):
                print(f"[skip]  {kind}/{name}: no reference file")
                continue
            with open(os.path.join(cand_dir, name)) as f:
                cand = metrics_of(json.load(f))
            with open(ref_path) as f:
                ref = metrics_of(json.load(f))
            compared += 1
            if set(cand) != set(ref):
                print(f"[FAIL]  {kind}/{name}: key sets differ")
                failures += 1
                continue
            diffs = {k: abs(cand[k] - ref[k]) for k in ref}
            max_key = max(diffs, key=diffs.get)
            max_diff = diffs[max_key]
            if max_diff == 0:
                print(f"[exact] {kind}/{name}")
            elif max_diff <= args.tolerance:
                print(f"[ok]    {kind}/{name}: max |diff| = {max_diff:.2e} ({max_key})")
            else:
                print(f"[FAIL]  {kind}/{name}: max |diff| = {max_diff:.2e} ({max_key}) exceeds tolerance {args.tolerance}")
                failures += 1

    if compared == 0:
        print(f"No overlapping result files between '{args.candidate}' and '{args.reference}'.")
        print("Run the benchmark first, e.g.: python src/benchmark/eval_openclip.py --model RN50 --pretrained openai")
        sys.exit(2)

    print(f"\nCompared {compared} file(s); {failures} failure(s).")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
