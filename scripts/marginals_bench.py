"""Supplementary workload: same cost/kernel, DIFFERENT marginals on each call.

This reproduces the Phase XXXII "kernel-reusable query" experiment.  Each repeat
perturbs a and b (fixed seed), keeps M and reg fixed, and for C uses the public
``prepare(M, reg)`` snapshot with ``solve(new_a, new_b)``.  A/B have no equivalent
cross-call cache and are called normally.  Results are a separate workload: they
must not be pooled with the cold or warm identical-query pools.

Usage:
    python scripts/marginals_bench.py --pot-src /path/to/POT-0.9.7.post1 \
        --out data/raw --tag repro [--compat-shim]
"""
import argparse
import csv
import gc
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))

from fixtures import KWARGS, REPEATS, SEED, load_fixtures  # noqa: E402
from three_way_bench import build, guard_identities, guard_output_names  # noqa: E402

WORKLOAD = ("tiny_8x8", "square_300x300", "warmish_90x90")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pot-src", required=True, type=Path)
    p.add_argument("--out", default=ROOT / "data" / "raw", type=Path)
    p.add_argument("--tag", default="repro")
    p.add_argument("--force", action="store_true")
    p.add_argument("--compat-shim", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    out = args.out.resolve()
    guard_output_names(args.tag, out, args.force)
    marg_target = out / f"{args.tag}_marginals_raw.csv"
    if marg_target.exists() and not args.force:
        raise SystemExit(f"OUTPUT GUARD: refusing to overwrite {marg_target} "
                         "(use --force)")
    ot, impl, rs, adapter_version = build(args.pot_src, args.compat_shim)
    guard_identities(args.pot_src, sys.modules["ot.bregman._sinkhorn"],
                     sys.modules["ot.bregman._sinkhorn_candidate"], rs)
    fixtures = load_fixtures()
    rng = np.random.default_rng(SEED + 9)
    rows = []
    for name in WORKLOAD:
        fx = fixtures[name]
        a, b, M, reg = fx["a"], fx["b"], fx["M"], fx["reg"]
        pairs = []
        for _ in range(REPEATS):
            aa = a * np.exp(0.1 * rng.standard_normal(a.size)); aa /= aa.sum()
            bb = b * np.exp(0.1 * rng.standard_normal(b.size)); bb /= bb.sum()
            pairs.append((aa, bb))
        prep = rs.prepare(M, reg)
        for rep in range(REPEATS):
            aa, bb = pairs[rep]
            for v in ("A", "B", "C"):
                gc.collect()
                t0 = time.perf_counter()
                if v == "C":
                    G = prep.solve(aa, bb, numItermax=KWARGS["numItermax"],
                                   stopThr=KWARGS["stopThr"], log=False)
                else:
                    G = impl[v](aa, bb, M, reg, numItermax=KWARGS["numItermax"],
                                stopThr=KWARGS["stopThr"], log=False)
                ms = (time.perf_counter() - t0) * 1000.0
                rows.append({"fixture": name, "variant": v, "repeat": rep,
                             "wall_ms": ms, "shape_ok": bool(np.asarray(G).shape == M.shape)})
        print("done", name, flush=True)
    out = args.out.resolve(); out.mkdir(parents=True, exist_ok=True)
    path = out / f"{args.tag}_marginals_raw.csv"
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print(f"wrote {path.relative_to(ROOT) if ROOT in path.parents else path}: "
          f"{len(rows)} samples; adapter {adapter_version}")


if __name__ == "__main__":
    main()
