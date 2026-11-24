"""Three-way dense Sinkhorn benchmark (A=Stock POT, B=hardened Candidate A, C=adapter 0.2.0).

Reproduces the Phase XXXII cold/warm experiments.  All paths are resolved relative
to the repository or supplied on the command line; no machine-specific absolute
path is required.

Usage (from repository root)::

    python scripts/three_way_bench.py --pot-src /path/to/POT-0.9.7.post1 \
        --out data/raw --tag repro

`--pot-src` points to an extracted POT 0.9.7.post1 source tree containing `ot/`.
The hardened Candidate A module must be present as
`ot/bregman/_sinkhorn_candidate.py` (see patches/ and REPRODUCIBILITY.md).
Install the author's pinned 0.2.0 wheel separately (see README).  On Windows use
`--compat-shim`.

Protocol: 3 warmups, 17 paired shuffled repeats, numItermax=3000, stopThr=1e-7,
log=False, full dense plan, single BLAS thread (set env vars before launch).
"""
import argparse
import csv
import gc
import hashlib
import importlib.metadata as imeta
import json
import random
import sys
import time
import warnings
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))

from fixtures import FIX_SPEC, KWARGS, REPEATS, SEED, WARMUP, load_fixtures  # noqa: E402

# Historical source identities from data/processed/phase32_config.json.
HIST_A = "71ee6030e5dffc4dff9b0200a8d2f2317da5e2546102e0f478d1b247bafe2592"
HIST_B = "586f71bb1a80af5260139df663387df96f301d7a5034f6a2ea2975c75e87ea61"


def crlf_hash(data: bytes) -> str:
    return hashlib.sha256(data.replace(b"\r\n", b"\n").replace(b"\n", b"\r\n")).hexdigest()


RESERVED_TAGS = {"phase32", "phase31", "phase29", "frozen", "raw"}


def guard_output_names(tag, out, force=False):
    """Refuse to write into the frozen historical CSV namespace.

    ``phase32_*`` files are the immutable historical evidence.  A benchmark tag of
    ``phase32`` (or another reserved name) is rejected unconditionally, and existing
    output files are never overwritten unless ``--force`` is given explicitly.
    Called before any timing loop so a bad name produces no CSVs.
    """
    if tag in RESERVED_TAGS:
        raise SystemExit(
            f"OUTPUT GUARD: tag '{tag}' is reserved for frozen historical data; "
            "choose e.g. --tag repro or --tag short")
    stem = f"{tag}_"
    targets = [out / f"{stem}{suffix}" for suffix in
               ("cold_raw.csv", "warm_raw.csv", "accuracy.csv")]
    existing = [str(p) for p in targets if p.exists()]
    if existing and not force:
        raise SystemExit(
            "OUTPUT GUARD: refusing to overwrite existing output file(s): " +
            ", ".join(existing) + " (use --force to replace non-reserved outputs)")
    return targets


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pot-src", required=True, type=Path,
                   help="POT 0.9.7.post1 extracted source tree (contains ot/)")
    p.add_argument("--out", default=ROOT / "data" / "raw", type=Path)
    p.add_argument("--compat-shim", action="store_true",
                   help="enable Windows import shim for adapter 0.2.0")
    p.add_argument("--tag", default="repro", help="prefix for output CSV filenames")
    p.add_argument("--force", action="store_true",
                   help="allow writing over an existing tagged output (never allowed for "
                        "reserved/phase32 names)")
    p.add_argument("--short", action="store_true",
                   help="preflight only: 2 small fixtures, 1 warmup, 3 repeats; "
                        "writes <tag>_* CSVs and never touches phase32_* files")
    return p.parse_args()


def guard_identities(pot_src, STOCK, CANDIDATE, rs):
    """Fail fast if A/B are swapped, missing, or from the wrong source tree.

    Called before any timing loop, so a bad setup never produces timing records.
    Also returns the adapter distribution version read from installed metadata
    (never hard-coded) for the output provenance record.
    """
    expected = {
        "A": (pot_src / "ot" / "bregman" / "_sinkhorn.py", HIST_A),
        "B": (pot_src / "ot" / "bregman" / "_sinkhorn_candidate.py", HIST_B),
    }
    modules = {"A": STOCK, "B": CANDIDATE}
    for label, mod in modules.items():
        path, digest = expected[label]
        actual_path = Path(mod.__file__).resolve()
        if actual_path != path.resolve():
            raise SystemExit(
                f"IDENTITY GUARD: {label} loaded from {actual_path}, expected {path.resolve()}")
        if not actual_path.is_file():
            raise SystemExit(f"IDENTITY GUARD: {label} source file missing: {actual_path}")
        actual_hash = crlf_hash(actual_path.read_bytes())
        if actual_hash != digest:
            hint = ""
            if label == "A" and actual_hash == HIST_B:
                hint = " (looks like the patched candidate: A/B appear SWAPPED)"
            elif label == "B" and actual_hash == HIST_A:
                hint = " (looks like pristine stock: A/B appear SWAPPED)"
            raise SystemExit(
                f"IDENTITY GUARD: {label} source hash {actual_hash} != {digest}{hint}; "
                "run prepare_candidate.py on a pristine POT 0.9.7.post1 tree")
        print(f"identity {label}: {actual_path.name} sha256={actual_hash} OK")

    try:
        adapter_version = imeta.version("residual-sinkhorn")
    except imeta.PackageNotFoundError as exc:
        raise SystemExit("IDENTITY GUARD: residual-sinkhorn distribution not installed") from exc
    rs_file = Path(rs.__file__).resolve()
    if "residual_sinkhorn" not in rs_file.parts:
        raise SystemExit(f"IDENTITY GUARD: adapter module unexpected path: {rs_file}")
    print(f"identity C: residual-sinkhorn {adapter_version} at {rs_file}")
    return adapter_version


def build(pot_src, compat_shim):
    sys.path.insert(0, str(pot_src.resolve()))
    import ot
    from ot.bregman import _sinkhorn as STOCK
    try:
        from ot.bregman import _sinkhorn_candidate as CANDIDATE
    except ImportError as exc:
        raise SystemExit("IDENTITY GUARD: ot.bregman._sinkhorn_candidate missing; "
                         "run prepare_candidate.py first") from exc
    if compat_shim:
        from windows_shim import install
        rs = install()
    else:
        import residual_sinkhorn as rs
    if ot.__version__ != "0.9.7.post1":
        raise SystemExit(f"IDENTITY GUARD: Expected POT 0.9.7.post1, got {ot.__version__}")
    adapter_version = guard_identities(pot_src, STOCK, CANDIDATE, rs)
    return ot, {"A": STOCK.sinkhorn_knopp, "B": CANDIDATE.sinkhorn_knopp,
                "C": rs.sinkhorn}, rs, adapter_version


def invoke(impl, v, a, b, M, reg, log=False):
    kw = dict(KWARGS); kw["log"] = log
    return impl[v](a, b, M, reg, **kw)


def timed(impl, rs, v, a, b, M, reg, protocol):
    gc.collect()
    if v == "C" and protocol == "COLD":
        rs.clear_cache()
    t0 = time.perf_counter()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        G = invoke(impl, v, a, b, M, reg, log=False)
    ms = (time.perf_counter() - t0) * 1000.0
    return ms, np.asarray(G), [str(w.message) for w in caught]


def diagnose(impl, rs, v, a, b, M, reg):
    if v in ("A", "B"):
        _, log = invoke(impl, v, a, b, M, reg, log=True)
        return int(log["niter"]) + 1, "pot_public_log", False
    rs.clear_cache()
    d = rs._dense(a, b, M, reg, int(KWARGS["numItermax"]), float(KWARGS["stopThr"]),
                  early=True, warn=False, details=True)
    if d is None:
        _, log = invoke(impl, "A", a, b, M, reg, log=True)
        return int(log["niter"]) + 1, "stock_pot_fallback_log", False
    return int(d[1]), "residual_internal_details", bool(d[2])


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)


def main():
    args = parse_args()
    out = args.out.resolve()
    guard_output_names(args.tag, out, args.force)
    ot, impl, rs, adapter_version = build(args.pot_src, args.compat_shim)
    fixtures = load_fixtures()
    if args.short:
        names = ["tiny_8x8", "warmish_90x90"]
        warmup, repeats = 1, 3
        print("SHORT PREFLIGHT MODE: 2 fixtures, 1 warmup, 3 repeats", flush=True)
    else:
        names, warmup, repeats = list(FIX_SPEC), WARMUP, REPEATS
    timing, accuracy = [], []
    for proto in ("COLD", "WARM"):
        rng = random.Random(SEED)
        for name in names:
            fx = fixtures[name]
            a, b, M, reg, Q = fx["a"], fx["b"], fx["M"], fx["reg"], fx["Q"]
            for v in impl:
                for _ in range(warmup):
                    if v == "C" and proto == "COLD":
                        rs.clear_cache()
                    invoke(impl, v, a, b, M, reg)
            if proto == "WARM":
                for _ in range(2):
                    invoke(impl, "C", a, b, M, reg)
                invoke(impl, "A", a, b, M, reg)
                invoke(impl, "B", a, b, M, reg)
            for rep in range(repeats):
                order = list(impl); rng.shuffle(order)
                for v in order:
                    ms, G, ws = timed(impl, rs, v, a, b, M, reg, proto)
                    if G.shape != M.shape or not np.isfinite(G).all():
                        raise RuntimeError(f"invalid plan {name}/{v}/{rep}")
                    timing.append({"protocol": proto, "fixture": name, "variant": v,
                                   "repeat": rep, "wall_ms": ms, "warnings": "|".join(ws)})
            if proto == "COLD":
                GA = np.asarray(invoke(impl, "A", a, b, M, reg))
                for v in impl:
                    if v == "C":
                        rs.clear_cache()
                    with warnings.catch_warnings(record=True) as caught:
                        warnings.simplefilter("always")
                        G = np.asarray(invoke(impl, v, a, b, M, reg))
                    niter, nsrc, early = diagnose(impl, rs, v, a, b, M, reg)
                    accuracy.append({
                        "fixture": name, "variant": v, "niter": niter,
                        "niter_source": nsrc, "early_vs_pot_checkpoint": early,
                        "shape_ok": bool(G.shape == M.shape), "mass": float(G.sum()),
                        "min": float(G.min()), "finite": bool(np.isfinite(G).all()),
                        "row_l2": float(np.linalg.norm(G.sum(1) - a)),
                        "col_l2": float(np.linalg.norm(G.sum(0) - b)),
                        "tv_vs_A": 0.5 * float(np.abs(G - GA).sum()),
                        "tv_vs_planted_Q": 0.5 * float(np.abs(G - Q).sum()),
                        "warnings": "|".join(sorted({str(w.message) for w in caught}))})
            print("done", proto, name, flush=True)
    write_csv(out / f"{args.tag}_cold_raw.csv", [r for r in timing if r["protocol"] == "COLD"])
    write_csv(out / f"{args.tag}_warm_raw.csv", [r for r in timing if r["protocol"] == "WARM"])
    write_csv(out / f"{args.tag}_accuracy.csv", accuracy)
    print(json.dumps({"pot": ot.__version__, "numpy": np.__version__,
                      "timing_samples": len(timing),
                      "adapter_version_from_metadata": adapter_version}, indent=2))


if __name__ == "__main__":
    main()
