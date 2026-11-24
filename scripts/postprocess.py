"""Recompute all Phase XXXII statistics from the frozen raw timing CSVs.

Raw files are read-only; no measurement is altered.  Primary statistic:
D1 = median(X_i)/median(Y_i) per fixture, geometric mean across fixtures.
Secondary cross-check: D2 = median(X_i/Y_i).

Usage (from repository root)::

    python scripts/postprocess.py
    python scripts/postprocess.py --raw-dir data/raw --out-dir validation/short
    python scripts/postprocess.py --tag repro --raw-dir data/raw --out-dir validation/repro

The frozen ``phase32_*`` CSVs are never modified: by default outputs carry the
``_recomputed`` suffix; with ``--tag`` they carry ``<tag>_`` and read
``<tag>_cold_raw.csv`` etc. from the input directory.
"""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
BOOT_SEED = 20261130
NBOOT = 20000
FIXTURES = ["tiny_8x8", "wide_32x512", "tall_512x32", "square_300x300",
            "cold_64x128_0.01", "illcond_150x600", "moderate_400x100", "warmish_90x90"]


def read_csv(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(out_dir, name, rows):
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / name, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)


def summarize(rows, protocol):
    groups = defaultdict(dict)
    for r in rows:
        groups[(r["fixture"], r["variant"])][int(r["repeat"])] = float(r["wall_ms"])
    brng = np.random.default_rng(BOOT_SEED)
    out = []
    for x, y in (("A", "B"), ("A", "C"), ("B", "C")):
        for fx in FIXTURES:
            if (fx, x) not in groups or (fx, y) not in groups:
                continue
            reps = sorted(set(groups[(fx, x)]) & set(groups[(fx, y)]))
            X = np.array([groups[(fx, x)][i] for i in reps])
            Y = np.array([groups[(fx, y)][i] for i in reps])
            idx = brng.integers(0, len(reps), size=(NBOOT, len(reps)))
            out.append({
                "protocol": protocol, "fixture": fx, "compare": f"{x}/{y}",
                "median_X_ms": float(np.median(X)), "median_Y_ms": float(np.median(Y)),
                "D1_ratio_of_medians": float(np.median(X) / np.median(Y)),
                "D1_ci_lo": float(np.quantile(np.median(X[idx], axis=1) /
                                              np.median(Y[idx], axis=1), 0.025)),
                "D1_ci_hi": float(np.quantile(np.median(X[idx], axis=1) /
                                              np.median(Y[idx], axis=1), 0.975)),
                "D2_median_paired_ratio": float(np.median(X / Y)),
                "frac_Y_faster": float(np.mean(Y < X)),
                "definition": "D1 = median(X_i)/median(Y_i) (PRIMARY)",
            })
    return out


def aggregate(rows):
    agg = {}
    for proto in ("COLD", "WARM", "MARGINALS"):
        agg[proto] = {}
        for pair in ("A/B", "A/C", "B/C"):
            sub = [r for r in rows if r["protocol"] == proto and r["compare"] == pair]
            if not sub:
                continue
            for key in ("D1_ratio_of_medians", "D2_median_paired_ratio"):
                vals = [r[key] for r in sub]
                agg[proto][f"{pair}_{key}"] = {
                    "geomean": float(np.exp(np.mean(np.log(vals)))),
                    "n_fixtures": len(vals),
                    "count_Y_faster": int(sum(v > 1 for v in vals)),
                    "min": float(min(vals)), "max": float(max(vals)),
                }
    return agg


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--raw-dir", type=Path, default=ROOT / "data" / "raw",
                   help="directory containing input raw CSVs")
    p.add_argument("--out-dir", type=Path, default=ROOT / "data" / "processed",
                   help="directory for derived tables")
    p.add_argument("--tag", default=None,
                   help="if set, read <tag>_{cold,warm,marginals,accuracy}_raw.csv "
                        "and write <tag>_derived_* instead of touching phase32 files")
    return p.parse_args()


def main():
    args = parse_args()
    raw_dir = args.raw_dir.resolve()
    out_dir = args.out_dir.resolve()
    if args.tag:
        prefix = args.tag
        cold_name = f"{prefix}_cold_raw.csv"
        warm_name = f"{prefix}_warm_raw.csv"
        marg_name = f"{prefix}_marginals_raw.csv"
        acc_name = f"{prefix}_accuracy.csv"
        out_stem = f"{prefix}_derived"
    else:
        cold_name, warm_name = "phase32_cold_raw.csv", "phase32_warm_raw.csv"
        marg_name, acc_name = "phase32_marginals_raw.csv", "phase32_accuracy.csv"
        out_stem = "phase32_recomputed"

    missing = [n for n in (cold_name, warm_name) if not (raw_dir / n).exists()]
    if missing:
        raise SystemExit(f"missing input CSV(s) in {raw_dir}: {missing}")
    cold = read_csv(raw_dir / cold_name)
    warm = read_csv(raw_dir / warm_name)
    marg = read_csv(raw_dir / marg_name) if (raw_dir / marg_name).exists() else []
    summary = summarize(cold, "COLD") + summarize(warm, "WARM")
    marg_summary = summarize(marg, "MARGINALS") if marg else []
    write_csv(out_dir, f"{out_stem}_summary.csv", summary)
    if marg_summary:
        write_csv(out_dir, f"{out_stem}_marginals_summary.csv", marg_summary)
    agg = aggregate(summary + marg_summary)
    (out_dir / f"{out_stem}_aggregate.json").write_text(
        json.dumps({"primary": "D1 = median(X_i)/median(Y_i), geomean across fixtures",
                    "secondary": "D2 = median(X_i/Y_i)",
                    "bootstrap": {"seed": BOOT_SEED, "repeats": NBOOT, "unit": "paired repeat index"},
                    "pools": "COLD and WARM are independent; never pooled",
                    "aggregate": agg}, indent=2), encoding="utf-8")

    print("=== Recomputed D1 geometric means ===")
    for proto in ("COLD", "WARM", "MARGINALS"):
        for pair in ("A/B", "A/C", "B/C"):
            key = f"{pair}_D1_ratio_of_medians"
            if proto not in agg or key not in agg[proto]:
                continue
            r = agg[proto][key]
            print(f"{proto:9s} {pair}  D1={r['geomean']:.6f}  "
                  f"Y_faster={r['count_Y_faster']}/{r['n_fixtures']}")

    # Iteration counts are taken from the accuracy table (timing-free diagnostic).
    acc_path = raw_dir / acc_name
    if acc_path.exists():
        acc = read_csv(acc_path)
        niter = {(r["fixture"], r["variant"]): int(r["niter"]) for r in acc}
        print("\n=== Iteration counts (timing-free diagnostic) ===")
        for fx in FIXTURES:
            if (fx, "A") in niter and (fx, "B") in niter and (fx, "C") in niter:
                print(f"{fx:20s} A={niter[(fx,'A')]:4d} B={niter[(fx,'B')]:4d} C={niter[(fx,'C')]:4d}")


if __name__ == "__main__":
    main()
