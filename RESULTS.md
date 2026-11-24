# Recorded Results (Phase XXXII)

All numbers below are derived from the frozen raw CSVs in `data/raw/` and are
recomputed by `scripts/postprocess.py`. Nothing here requires re-running the benchmark.

## Primary statistic

**D1 = median(X_i) / median(Y_i)** within each fixture, aggregated by the geometric
mean across the eight fixtures. A value greater than 1 means implementation **Y** is
faster than **X**. D2 = median(X_i/Y_i) is retained only as a cross-check.

| Comparison | Cold D1 | Warm D1 |
| --- | ---: | ---: |
| A / B (Stock / hardened POT) | **1.161×** | 1.029× |
| A / C (Stock / adapter 0.2.0) | **1.568×** | **1.629×** |
| B / C (hardened POT / adapter) | **1.351×** | **1.583×** |

Cold: C faster in 8/8 fixtures vs A and B; B faster in 8/8 vs A.
Warm: C faster in 8/8 vs A and B; B faster in 5/8 vs A (bootstrap CI excludes 1 in 0/8).
Cold and warm samples are independent and must not be pooled.

Secondary D2 geometric means: Cold A/B 1.112, A/C 1.565, B/C 1.390;
Warm A/B 1.028, A/C 1.676, B/C 1.640. Both statistics agree in sign.

## The `cold_64x128_0.01` case

This fixture deserves separate attention. All three implementations performed **1,391
iterations**, yet in the cold-cache protocol C was approximately **2.35× faster than A**
and **2.28× faster than B** (median times 26.97 ms / 26.13 ms / 11.47 ms).

Independent `cProfile` sampling on this fixture (five calls each) indicates that
per-iteration checking overhead is a substantial contributor to the gap; profiling
**suggests** it is not explained by convergence speed alone:

| Implementation | Total time (5 calls) | Dominant calls |
| --- | ---: | --- |
| A | 0.239 s | 34,775 `nx.any` checks (0.129 s), 13,910 `dot` |
| B | 0.221 s | 34,775 `nx.any` checks (0.126 s), 13,910 `dot` |
| C | 0.055 s | 2,105 `sum`, 710 `norm`, 710 `dot` |

Across all eight fixtures, iteration counts were identical for A/B/C in six cases. C
stopped earlier only on `tiny_8x8` (15 vs 21) and `warmish_90x90` (8 vs 11) — an
observation about those two fixtures, not a general stopping rule. Profiling **indicates**
that lower per-iteration checking cost contributes materially to the observed speedups,
and the equal-iteration `cold_64x128_0.01` result is consistent with that mechanism.
This evidence is from one profiled fixture and should not be generalized unconditionally
to all workloads. The kernel cache is a separate warm-cache contribution and early
stopping is a fixture-specific secondary effect; these contributions must not be
conflated, and the causal attribution remains tentative.

## Warm-cache and kernel-reuse workloads

Warm repeats the *identical* (a, b, M, reg) query and lets C reuse its kernel; A/B have
no equivalent cross-call cache. A supplementary workload keeps M and reg fixed but
perturbs the marginals and uses C's public `prepare()` API:

| Workload | A/C | B/C |
| --- | ---: | ---: |
| Warm, identical query | 1.629× | 1.583× |
| Same kernel, different marginals | 1.835× | 1.755× |

## Numerical accuracy

All plans were finite, non-negative, of the correct shape and total mass 1. Row-marginal
L2 residuals were at most 2.4e-17; column-marginal L2 at most 9.92e-8. On the six
fixtures with identical iteration counts, plans agreed to approximately 1e-16 in total
variation (TV). On the two early-stopping fixtures:

| Fixture | TV(A,C) | TV(C, planted Q) | TV(A, planted Q) |
| --- | ---: | ---: | ---: |
| `tiny_8x8` | 9.59e-10 | 9.60e-10 | 5.91e-13 |
| `warmish_90x90` | 2.99e-12 | 2.99e-12 | 3.49e-16 |

Early stopping therefore has a measurable accuracy cost, but the absolute error remains
below `stopThr=1e-7` in these cases. The marginal threshold is **not** treated as a
general certificate for full-plan TV error.

Fallback tests confirmed that zero-mass marginals, float32 inputs, multiple histograms,
warm starts, non-contiguous arrays, non-finite costs and true Gibbs-kernel underflow all
delegate to stock POT with TV = 0 against a direct stock call.

## Statistical reconciliation

An earlier report quoted `warmish_90x90` as 0.959× while listing medians 0.593 ms and
0.883 ms. Recomputed from untouched raw data: D1 = 0.672, D2 = 0.959. The discrepancy
was a mixture of two legitimate ratio definitions, not a data error. All results here
use D1. See `data/raw/statistics_reconciled.csv` and the Phase XXXI reconciliation note.

## Verification artifacts

* `validation/wolfram_phase32.txt` — exact/numeric checks of the TV identity and of the
  planted optimum's Sinkhorn scaling fixed point.
* `data/raw/phase32_fallback_audit.csv`, `data/raw/phase32_underflow_probe.csv` —
  fallback behaviour.
* `data/raw/zero_mass_regression.csv` — compatibility regression for the hardened patch
  (14 unsupported-input cases: 10 bitwise identical to stock; 4 degenerate cases differ by
  at most 6.94e-18, consistent with float64 rounding).
