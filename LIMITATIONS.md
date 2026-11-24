# Limitations and Scope

## Scope of the performance claim

The eight frozen fixtures cover small, wide, tall, square, ill-conditioned and
low-regularization dense CPU problems. They are **not** a representative sample of all
Sinkhorn workloads. The results must not be generalized to GPUs, PyTorch/JAX backends,
`sinkhorn_stabilized`, sparse costs, non-float64 inputs, or problem sizes and condition
numbers outside this set.

Small-sample timing variation is substantial. For `warmish_90x90` the ratio A/B was
0.67× in one Phase XXXI run, 0.81× in an independent A/B-only control run and 1.39× in
the recorded Phase XXXII cold run. Conclusions therefore rest on paired bootstrap
intervals and cross-fixture consistency, not on a single median ratio.

## Version used in the recorded results

The historical experiment used **residual-sinkhorn 0.2.0**. The author subsequently
published **0.2.1**, which fixes the Windows import issue natively by catching `TypeError`
around the `ctypes` lookup. The author explicitly stated that 0.2.1 was *not* used to
re-run the full 3.13 comparison. This package records 0.2.0 results; 0.2.1 numbers would
be a new experiment and must be labelled as such.

Optional 0.2.1 reproduction path (produces **new**, non-comparable numbers):

```bash
# Obtain 0.2.1 from the author's Sinkhorn notes (see environment/upstream_urls.json),
# install it in a fresh environment, then:
python scripts/three_way_bench.py --pot-src ./POT-0.9.7.post1 \
  --out data/raw --tag v021          # do NOT pass --compat-shim
```

## Windows compatibility layer

On Windows + CPython 3.13, 0.2.0 fails to import because `ctypes.CDLL(None)` raises
`TypeError` and the wheel catches only `AttributeError`/`OSError`. The supplied shim
converts this to `OSError` so the author's own NumPy fallback is selected. This does not
change the solver, but warm-cache checks then use `np.array_equal` instead of `memcmp`,
adding a cost proportional to the matrix size. Warm-cache figures are therefore
conservative relative to the POSIX native path.

## Stopping semantics

C may stop between POT's ten-iteration checkpoints. Its iteration count is obtained from
an internal read-only details path, not from `log=True` (which would force a stock-POT
fallback). Reporting iteration counts from `log=True` for C would invalidate the
comparison. The residual-based early certificate is a float64 estimate, not a rigorously
outward-rounded error bound.

## What the patch is and is not

The hardened patch is a *minimal POT integration* of two optimizations already present
in the adapter: eliminating the scaled-kernel copy and reusing matrix-vector products.
It retains POT's stopping rule and compatibility guards. It does **not** include:

* residual-based early stopping between POT checkpoints;
* any cross-call kernel cache or `prepare()`-style snapshot API;
* any optimization from POT PR #913.

The first two are research features of the adapter and remain outside POT upstream at
the time of writing.

## Licensing and redistribution

The adapter wheel and POT source tree are third-party assets and are **not** included.
Only a pinned download link and SHA-256 are provided. See `ACKNOWLEDGMENTS.md`.
