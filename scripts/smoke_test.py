"""Small-scale three-way correctness smoke test (no timing, no full benchmark).

Checks: POT 0.9.7.post1, patch can be applied, three representative workloads
(zero-mass negative control, ordinary positive input, long-iteration fixture),
plan shape/mass/finiteness/marginals and adapter fallback.

Usage:
    python scripts/smoke_test.py --pot-src /path/to/POT-0.9.7.post1 [--compat-shim]
"""
import argparse
import sys
import warnings
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))

from fixtures import load_fixtures  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--pot-src", required=True, type=Path)
    p.add_argument("--compat-shim", action="store_true")
    return p.parse_args()


def check_plan(G, a, b, M, label):
    G = np.asarray(G)
    assert G.shape == M.shape, f"{label}: shape {G.shape} != {M.shape}"
    assert np.isfinite(G).all(), f"{label}: non-finite entries"
    assert (G >= 0).all(), f"{label}: negative entries"
    assert abs(G.sum() - 1.0) < 1e-10, f"{label}: mass {G.sum()}"
    row = float(np.linalg.norm(G.sum(1) - a))
    col = float(np.linalg.norm(G.sum(0) - b))
    print(f"  {label}: shape={G.shape} mass={G.sum():.12f} row_l2={row:.2e} col_l2={col:.2e}")


def main():
    args = parse_args()
    sys.path.insert(0, str(args.pot_src.resolve()))
    import ot
    from ot.bregman import _sinkhorn as STOCK
    from ot.bregman import _sinkhorn_candidate as CAND
    if args.compat_shim:
        from windows_shim import install
        rs = install()
    else:
        import residual_sinkhorn as rs
    assert ot.__version__ == "0.9.7.post1", ot.__version__
    print(f"POT {ot.__version__}, NumPy {np.__version__}, adapter module {rs.__file__}")

    fx = load_fixtures()
    impl = {"A": STOCK.sinkhorn_knopp, "B": CAND.sinkhorn_knopp, "C": rs.sinkhorn}

    print("\n[1] ordinary positive input: tiny_8x8")
    f = fx["tiny_8x8"]
    for v, fn in impl.items():
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            G = fn(f["a"], f["b"], f["M"], f["reg"], numItermax=3000, stopThr=1e-7, log=False)
        check_plan(G, f["a"], f["b"], f["M"], v)

    print("\n[2] long-iteration fixture: cold_64x128_0.01 (1391 iterations)")
    f = fx["cold_64x128_0.01"]
    for v, fn in impl.items():
        if v == "C":
            rs.clear_cache()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            G = fn(f["a"], f["b"], f["M"], f["reg"], numItermax=3000, stopThr=1e-7, log=False)
        check_plan(G, f["a"], f["b"], f["M"], v)

    print("\n[3] zero-mass negative control (A and B must stay identical)")
    f = fx["tiny_8x8"]
    a0 = f["a"].copy(); a0[0] = 0.0
    out = []
    for v in ("A", "B", "C"):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            G = np.asarray(impl[v](a0, f["b"], f["M"], f["reg"],
                                   numItermax=3000, stopThr=1e-7, log=False))
        out.append(G)
        print(f"  {v}: warnings={len(caught)} finite={bool(np.isfinite(G).all())}")
    tv_ab = 0.5 * float(np.abs(out[0] - out[1]).sum())
    tv_ac = 0.5 * float(np.abs(out[0] - out[2]).sum())
    assert np.array_equal(out[0], out[1]), f"zero-mass A/B not bitwise equal, TV={tv_ab}"
    assert tv_ac < 1e-12, f"zero-mass C fallback differs from stock, TV={tv_ac}"
    print(f"  A vs B bitwise equal; TV(A,C)={tv_ac:.2e} (delegated to stock POT)")

    print("\nPASS: smoke test completed")


if __name__ == "__main__":
    main()
