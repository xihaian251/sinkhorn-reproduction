# Sinkhorn Three-Way Reproduction Package

Independent, reproducible evidence for the three-way Sinkhorn comparison discussed in
[PythonOT/POT Issue #934](https://github.com/PythonOT/POT/issues/934). The package lets a
third party verify the reported timings, iteration counts and numerical accuracy on the
exact frozen inputs used in the original experiment, without re-deriving any result.

## What is compared

| Label | Implementation |
| --- | --- |
| **A** | Stock POT `sinkhorn_knopp`, unmodified POT 0.9.7.post1. |
| **B** | The same POT version plus the *hardened minimal patch* in `patches/`: removes the scaled-kernel copy `(1/a)[:,None]*K` and reuses `K.T @ u` matrix-vector products. POT's original stopping rule is preserved. |
| **C** | `residual-sinkhorn` **0.2.0**, the adapter released by hys-ops (imported as `residual_sinkhorn.sinkhorn`). Uses a residual-based early check and an optional thread-local kernel cache. |

**Origin of ideas.** The computational ideas, residual stopping strategy and adapter 0.2.0
are by **hys-ops**. This package contains an *independent* reproduction: common-version
performance attribution, compatibility testing, frozen-fixture replication, numerical
verification and statistical reconciliation. It is not a competing pull request. See
`ACKNOWLEDGMENTS.md`.

## Environment of the recorded experiment

POT 0.9.7.post1 · residual-sinkhorn 0.2.0 · NumPy 2.5.3 · SciPy 1.18.1 ·
Python 3.13.1 (Windows) · float64 · one BLAS/OMP thread · `log=False` ·
`stopThr=1e-7` · `numItermax=3000` · 3 warmups + 17 paired shuffled repeats.

The adapter **0.2.0** is the version actually used. The author subsequently released
**0.2.1** with a native Windows import fix; it is *not* the version behind the recorded
results. See `LIMITATIONS.md` for the optional 0.2.1 path.

## Quick start (no local absolute paths required)

All commands assume the repository root as working directory.

```bash
python -m pip install -r requirements.txt
python scripts/verify_fixtures.py
python scripts/postprocess.py
python scripts/verify_release.py
```

`postprocess.py` recomputes every published statistic from `data/raw/` and prints the
geometric-mean speedups and iteration counts. Expected output is documented in
`RESULTS.md` and in `validation/recompute_output.txt`.

## Obtaining POT and the adapter (not redistributed here)

Third-party code is referenced, not redistributed.

```bash
# POT 0.9.7.post1 source tree, then build its native extensions in place
git clone --branch 0.9.7.post1 --depth 1 https://github.com/PythonOT/POT.git POT-0.9.7.post1
git -C POT-0.9.7.post1 submodule update --init --recursive
cd POT-0.9.7.post1 && python setup.py build_ext --inplace && cd ..
# (Windows needs MSVC build tools + Windows SDK; see REPRODUCIBILITY.md)

# adapter 0.2.0 wheel, pinned by the author
curl -L -o residual_sinkhorn-0.2.0-py3-none-any.whl \
  https://raw.githubusercontent.com/hys-ops/lloyd-nonconvergence/b0b262330aa56c3583371795e1fe68c7138dcbe2/notes/sinkhorn/downloads/residual_sinkhorn-0.2.0-py3-none-any.whl
sha256sum residual_sinkhorn-0.2.0-py3-none-any.whl
# expected: d7099ea4dbad90242bf7318e2235c18cad579e43d3312faa8bfc76e78c854759

python -m pip install --no-deps ./residual_sinkhorn-0.2.0-py3-none-any.whl
```

Prepare the two POT solvers with the identity-checked helper (see `REPRODUCIBILITY.md`):

```bash
python scripts/prepare_candidate.py --pot-src ./POT-0.9.7.post1
```

This keeps **A** = original `ot/bregman/_sinkhorn.py` and creates **B** = patched
`ot/bregman/_sinkhorn_candidate.py`, verifying both source hashes.

## Running the benchmark

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python scripts/three_way_bench.py --pot-src ./POT-0.9.7.post1 \
  --out data/raw --tag repro --compat-shim
```

Omit `--compat-shim` on POSIX. Use `--tag repro` so the frozen `phase32_*` raw files are
never overwritten. Outputs appear as `data/raw/repro_{cold,warm,accuracy}.csv` and can be
fed to `postprocess.py` via its arguments.

## Repository layout

```
README.md                 overview (this file)
REPRODUCIBILITY.md        exact commands, patch application, statistics protocol
RESULTS.md                headline numbers and per-fixture tables
LIMITATIONS.md            scope, known caveats, 0.2.1 optional path
ACKNOWLEDGMENTS.md        attribution and licensing
LICENSE                   MIT license for material authored in this package
requirements.txt          runtime dependencies
scripts/                  portable benchmark, postprocessing and verification scripts
patches/                  hardened Candidate A patch
data/raw/                 frozen original timing and accuracy CSVs (read-only)
data/fixtures/            eight frozen fixtures and their SHA-256 manifest
data/processed/           derived summaries, aggregate statistics and provenance
validation/               Wolfram checks, recomputation log, patch verification
environment/              recorded versions, wheel provenance, upstream URLs
MANIFEST.sha256           checksums of all package files
```

## Reproducibility statement

No timing sample was modified or removed. The eight fixtures were frozen once and are
distributed with hashes. Cold and warm pools are independent and are never combined.
The primary speedup statistic is the ratio of per-fixture medians (D1), aggregated by a
geometric mean; the median of paired ratios (D2) is retained only as a cross-check.
