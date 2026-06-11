"""Compute semantic power density (phi) for every model in the benchmark.

Phi is the efficiency metric introduced in the ICPR 2026 paper (Eq. 2):

    phi = ( Recall@1 / (1 - Recall@1 + eps) )^2 / N_params  * 100

where Recall@1 is the micro Recall@1 over the 20,565-image evaluation set,
N_params is the parameter count IN MILLIONS, and eps = 1e-6. The squared
signal-to-noise term grows super-linearly above the 50% viability threshold,
penalizing sub-threshold models regardless of parameter efficiency.

Run from the repository root:

    python scripts/compute_phi.py                 # paper reference results
    python scripts/compute_phi.py --csv <other>   # e.g. a rerun summary

Reproduces the paper's Figure 4 values and Table: MobileCLIP-B (datacompdr)
phi = 2.37, ViT-gopt-16-SigLIP2-384 phi = 0.60.
"""

import argparse
import csv

EPS = 1e-6


def phi(recall_at_1, nparams):
    n_millions = nparams / 1e6
    snr = recall_at_1 / (1.0 - recall_at_1 + EPS)
    return (snr ** 2) / n_millions * 100.0


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--csv", default="results/benchmark_summary.csv", help="Benchmark summary CSV to read")
    parser.add_argument("--out", default=None, help="Optional output CSV path (default: print only)")
    parser.add_argument("--top", type=int, default=10, help="How many top-phi models to print")
    args = parser.parse_args()

    rows = []
    with open(args.csv, newline="") as f:
        for rec in csv.DictReader(f):
            if not rec.get("vision_model"):
                continue
            r1 = float(rec["micro_cmc@1"])
            nparams = int(rec["nparams"])
            rows.append({
                "vision_model": rec["vision_model"],
                "pretrained_on": rec["pretrained_on"],
                "nparams": nparams,
                "micro_cmc@1": r1,
                "phi": phi(r1, nparams),
            })

    rows.sort(key=lambda r: -r["phi"])

    print(f"phi for {len(rows)} models (top {args.top}):")
    print(f"{'model':32s} {'pretrained':24s} {'params(M)':>9s} {'R@1':>6s} {'phi':>6s}")
    for r in rows[: args.top]:
        print(f"{r['vision_model']:32s} {r['pretrained_on']:24s} "
              f"{r['nparams']/1e6:9.0f} {r['micro_cmc@1']*100:6.2f} {r['phi']:6.2f}")

    if args.out:
        with open(args.out, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["vision_model", "pretrained_on", "nparams", "micro_cmc@1", "phi"])
            for r in rows:
                w.writerow([r["vision_model"], r["pretrained_on"], r["nparams"],
                            f"{r['micro_cmc@1']:.6f}", f"{r['phi']:.6f}"])
        print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
