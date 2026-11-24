# Reproducibility Protocol

## 1. Verify the frozen inputs

```bash
python scripts/verify_fixtures.py
```

This checks all 32 arrays (a, b, M, Q for eight fixtures) against
`data/fixtures/fixture_hashes.json`. The fixture generator is intentionally **not**
included: the arrays were frozen once and must not be regenerated.

## 2. Prepare the POT 0.9.7.post1 source tree

```bash
git clone --branch 0.9.7.post1 --depth 1 https://github.com/PythonOT/POT.git POT-0.9.7.post1
```

A fresh checkout contains **no compiled native extensions**. `import ot` needs
`ot.lp.emd_wrap`, `ot.bsp.bsp_wrap` and (optionally) `ot.partial.partial_cython`, so a
plain clone will fail at import time. Build them in place before importing the tree:

```bash
# prerequisites: C/C++ compiler + Python headers; numpy and cython available
python -m pip install "numpy>=1.24" cython setuptools wheel
git -C POT-0.9.7.post1 submodule update --init --recursive   # Eigen headers
cd POT-0.9.7.post1
python setup.py build_ext --inplace
cd ..
python -c "import sys; sys.path.insert(0, 'POT-0.9.7.post1'); import ot; print(ot.__version__)"
```

On Windows this requires the MSVC build tools (matching the CPython installation) and
the Windows SDK; on Linux/macOS a normal C compiler is sufficient. If a build is not
possible, `pip install "POT==0.9.7.post1"` provides the extensions but `--pot-src` must
still point at the source tree used for A/B, so keep the source checkout for the patch.

The benchmark imports two solvers from this tree:

* **A** = the original `ot/bregman/_sinkhorn.py`;
* **B** = `ot/bregman/_sinkhorn_candidate.py`, produced by applying the hardened patch
  to a copy of the stock source.

Do **not** copy the stock file to the candidate first and then patch `_sinkhorn.py`: that
reverses the two identities. Use the supplied script, which applies the patch to the
stock file, saves the patched result as the candidate, restores stock, and verifies both
hashes against the historical values:

```bash
python scripts/prepare_candidate.py --pot-src ./POT-0.9.7.post1
```

The script checks the POT version (`0.9.7.post1`), the pristine stock hash, runs
`git apply --check`, applies the patch, writes the candidate, restores stock in a
`finally` block (so a failure never leaves the wrong source state), and verifies:

```
A (ot/bregman/_sinkhorn.py)            normalized sha256 = 71ee6030...2592
B (ot/bregman/_sinkhorn_candidate.py)  normalized sha256 = 586f71bb...ea61
```

"Normalized" means newlines reduced to CRLF; historical trees used CRLF while a fresh
POSIX checkout usually uses LF. This normalization is semantics-preserving.

The patch itself touches only the solver body and contains **no** early stopping, no
cross-call cache, and no optimization from PR #913. `git apply --check` must succeed on
the unmodified tree. Install the candidate module's dependencies into the environment
used to run the tree (`pip install "POT==0.9.7.post1"` gives the same version but the
scripts take `--pot-src` explicitly so A and B are loaded from the *same* tree).

## 3. Install the adapter

```bash
curl -L -o residual_sinkhorn-0.2.0-py3-none-any.whl \
  https://raw.githubusercontent.com/hys-ops/lloyd-nonconvergence/b0b262330aa56c3583371795e1fe68c7138dcbe2/notes/sinkhorn/downloads/residual_sinkhorn-0.2.0-py3-none-any.whl
sha256sum residual_sinkhorn-0.2.0-py3-none-any.whl
python -m pip install --no-deps ./residual_sinkhorn-0.2.0-py3-none-any.whl
```

The expected SHA-256 is `d7099ea4dbad90242bf7318e2235c18cad579e43d3312faa8bfc76e78c854759`.
The wheel is not redistributed here. On Windows + CPython 3.13 pass `--compat-shim` to the
benchmark; this enables the supplied 4-line shim that routes the adapter's failing
`ctypes.CDLL(None)` call through the author's own `np.array_equal` fallback.

## 4. Run the benchmark

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python scripts/three_way_bench.py --pot-src ./POT-0.9.7.post1 \
  --out data/raw --tag repro --compat-shim
```

The script uses the frozen seeds and parameters embedded in `scripts/fixtures.py`:
seed 20261129, 3 warmups, 17 paired repeats, `numItermax=3000`, `stopThr=1e-7`,
`log=False`, full dense plan, single BLAS thread. Iteration counts are collected in a
separate timing-free pass (POT public log for A/B; the adapter's internal `details`
channel for C). The timed adapter calls **never** use `log=True`, which would silently
delegate to stock POT.

**Cold** calls `clear_cache()` before every timed C call. **Warm** primes the adapter
twice and retains the cache. The two pools are written to separate CSVs.

Before any timing loop the script guards implementation identity: A must load from
`ot/bregman/_sinkhorn.py` with normalized SHA-256 `71ee6030…2592`, B from
`ot/bregman/_sinkhorn_candidate.py` with `586f71bb…ea61`, and the adapter distribution
version is read from installed metadata (not hard-coded). A swapped, missing or
wrong-version setup exits immediately and writes **no** timing records. A deliberate
A/B swap negative control is recorded in
`validation/identity_guard_negative_control.txt`.

A full run takes roughly four minutes on the reference machine. For a quick preflight
that does not re-run the historical benchmark:

```bash
python scripts/smoke_test.py --pot-src ./POT-0.9.7.post1 --compat-shim
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python scripts/three_way_bench.py --pot-src ./POT-0.9.7.post1 \
  --out validation/short_raw --tag short --compat-shim --short
python scripts/postprocess.py --raw-dir validation/short_raw \
  --out-dir validation/short_derived --tag short
```

`--short` uses two small fixtures, one warmup and three repeats; it writes only
`<tag>_*` CSVs. The separate kernel-reuse (different marginals) workload is run with:

```bash
python scripts/marginals_bench.py --pot-src ./POT-0.9.7.post1 \
  --out data/raw --tag repro --compat-shim
```

## 5. Recompute statistics

```bash
python scripts/postprocess.py
python scripts/postprocess.py --raw-dir validation/short_raw \
  --out-dir validation/short_derived --tag short
```

`--raw-dir`, `--out-dir` and `--tag` allow recomputation of any tagged run into a
separate directory. The frozen `phase32_*` raw CSVs are never overwritten; the default
outputs carry the `_recomputed` suffix.

The script reads `data/raw/phase32_{cold,warm,marginals,accuracy}.csv` and computes:

* **D1 (primary)** = `median(X_i) / median(Y_i)` per fixture, then geometric mean over fixtures;
* **D2 (secondary)** = `median(X_i / Y_i)` per fixture, then geometric mean;
* 95% bootstrap confidence intervals from 20,000 resamples of the paired repeat index
  (seed 20261130);
* iteration counts from the frozen accuracy CSV.

Expected canonical D1 values are in `RESULTS.md`. Raw files are read-only.

## 6. Release preflight

```bash
python scripts/verify_release.py --wheel /path/to/residual_sinkhorn-0.2.0-py3-none-any.whl
```

Checks the wheel checksum, the raw CSV checksums recorded in
`data/processed/phase32_config.json`, and scans text files for leftover absolute paths.
To rebuild the checksum manifest after editing any file:

```bash
python -c "import hashlib,pathlib; r=pathlib.Path('.'); print('\n'.join(sorted(f'{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.as_posix()}' for p in r.rglob('*') if p.is_file() and p.name!='MANIFEST.sha256')))"
```
