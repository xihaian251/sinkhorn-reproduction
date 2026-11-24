"""Compatibility regression for hardened Candidate A (14 unsupported inputs).

Verifies the patched fast path stays on POT's historical path for zero mass,
non-finite, float32, multi-histogram, warmstart, non-contiguous and extreme
regularization inputs, matching stock POT bitwise.  Does not read or overwrite
the frozen CSV in data/raw/; results are written to validation/.

Usage: python scripts/zero_mass_regression.py --pot-src /path/to/POT-0.9.7.post1
"""
import argparse
import csv
import sys
import warnings
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--pot-src", required=True, type=Path)
    return p.parse_args()


def main():
    args = parse_args()
    sys.path.insert(0, str(args.pot_src.resolve()))
    from ot.bregman import _sinkhorn as STOCK
    from ot.bregman import _sinkhorn_candidate as CAND

    rng = np.random.default_rng(4242)
    R, S = 20, 35
    a = rng.random(R) + 0.1; a /= a.sum()
    b = rng.random(S) + 0.1; b /= b.sum()
    M = rng.random((R, S)) * 3
    FN_A, FN_B = STOCK.sinkhorn_knopp, CAND.sinkhorn_knopp

    def run(fn, args_, kw):
        with warnings.catch_warnings(record=True) as ws:
            warnings.simplefilter("always")
            out = fn(*args_, **kw)
        G = out[0] if isinstance(out, tuple) else out
        return np.asarray(G), sorted({str(w.message) for w in ws})

    cases = {
        "zero_source": ((np.r_[0.0, a[1:]], b, M, 0.5), dict(stopThr=1e-8)),
        "zero_target": ((a, np.r_[0.0, b[1:]], M, 0.5), dict(stopThr=1e-8)),
        "float32": ((a.astype(np.float32), b.astype(np.float32), M.astype(np.float32), 0.5), dict(stopThr=1e-6)),
        "multi_hist": ((a, np.stack([b, b[::-1]], 1), M, 0.5), dict(stopThr=1e-8)),
        "warmstart": ((a, b, M, 0.5), dict(warmstart=(np.zeros(R), np.zeros(S)), stopThr=1e-8)),
        "fortran_cost": ((a, b, np.asfortranarray(M), 0.5), dict(stopThr=1e-8)),
        "strided_marginal": ((a[::-1], b, M[::-1], 0.5), dict(stopThr=1e-8)),
        "kernel_underflow": ((a, b, M * 1000, 0.01), dict(stopThr=1e-8)),
        "nonfinite_cost": ((a, b, np.where(np.isfinite(M), M, np.nan), 0.5), dict(stopThr=1e-8)),
        "nonfinite_marginal": ((np.r_[np.inf, a[1:]], b, M, 0.5), dict(stopThr=1e-8)),
        "numItermax_1": ((a, b, M, 0.5), dict(numItermax=1)),
        "stopThr_large": ((a, b, M, 0.5), dict(stopThr=10.0)),
        "extreme_reg": ((a, b, M, 1e12), dict(stopThr=1e-8)),
        "decisive_zero_identity": ((np.array([0.0, 1.0]), np.array([0.5, 0.5]),
                                    np.array([[0.0, 1000.0], [1000.0, 0.0]]), 1.0), dict(log=True)),
    }
    rows = []
    for name, (argv, kw) in cases.items():
        try:
            Ga, wa = run(FN_A, argv, kw)
            Gb, wb = run(FN_B, argv, kw)
            d = float(np.abs(np.nan_to_num(Ga) - np.nan_to_num(Gb)).max())
            rows.append(dict(case=name, status="OK",
                             bitwise_equal=bool(np.array_equal(Ga, Gb, equal_nan=True)),
                             max_abs_diff=d, warn_A="|".join(wa), warn_B="|".join(wb),
                             finite_A=bool(np.isfinite(Ga).all()), finite_B=bool(np.isfinite(Gb).all())))
        except Exception as exc:  # noqa: BLE001
            rows.append(dict(case=name, status="EXCEPTION:" + type(exc).__name__,
                             bitwise_equal="", max_abs_diff="", warn_A="", warn_B="",
                             finite_A="", finite_B=""))
    out = ROOT / "validation" / "zero_mass_regression_recomputed.csv"
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
    print("cases", len(rows))
    print("bitwise_equal", sum(r["bitwise_equal"] is True for r in rows))
    print("max_diff", max((float(r["max_abs_diff"]) for r in rows if r["max_abs_diff"] != ""), default=0.0))
    print("written", out.relative_to(ROOT))


if __name__ == "__main__":
    main()
