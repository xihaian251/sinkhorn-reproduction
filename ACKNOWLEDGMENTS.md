# Acknowledgments and Attribution

## Origin of the research

The Sinkhorn performance ideas, the residual stopping strategy, the kernel-reuse design
and the `residual-sinkhorn` adapter (versions 0.2.0 and 0.2.1) are the work of
**hys-ops**. The underlying discussion is
[PythonOT/POT Issue #934](https://github.com/PythonOT/POT/issues/934), where the author
released the pinned 0.2.0 wheel and supplied the `log=False` / cold-vs-warm guidance that
made a fair comparison possible.

## Contribution of this package

This package is an independent third-party reproduction. Its contributions are:

* a minimal POT integration (the hardened Candidate A patch) with compatibility guards
  restricting the fast path to finite, strictly positive float64 inputs;
* a zero-mass negative-control regression demonstrating the patch stays on POT's
  historical path for unsupported inputs;
* common-version, paired, cold/warm performance attribution using the same eight frozen
  fixtures for A, B and C;
* compatibility testing, wheel provenance verification and numerical cross-validation;
* statistical reconciliation of the ratio-of-medians vs median-of-ratios definitions.

No competing pull request was opened against POT as part of this work.

## Third-party assets and licensing

| Asset | Provider | License / terms | Distribution in this package |
| --- | --- | --- | --- |
| POT 0.9.7.post1 source | PythonOT contributors | MIT (see `environment/POT_LICENSE.txt`) | Not redistributed; official tag URL provided. |
| `residual-sinkhorn` 0.2.0 wheel | hys-ops | Author-provided wheel; no separate license file is present inside the wheel | Not redistributed; pinned URL and SHA-256 provided. |
| Wolfram 15.0.1 | Wolfram Research | Proprietary; used locally for verification only | No code or kernel redistributed; only textual results included in `validation/`. |

Users must obtain third-party code from the linked upstream sources and comply with
their respective licenses.

## License for material authored in this package

The scripts, documentation, patch and derived/processed data written for this
reproduction (everything in `scripts/`, `patches/`, `*.md`, `data/raw/` derived
CSVs that we produced, `data/processed/`, `validation/` and `environment/`, except
`environment/POT_LICENSE.txt`) are released under the **MIT License**, copyright
(c) 2026 the reproduction authors. A copy is in `LICENSE` at the repository root.
They are provided for reproduction and verification of the reported experiment.

The hardened patch is an original minimal integration authored for this reproduction;
it does not copy or re-license POT source (which remains MIT, see
`environment/POT_LICENSE.txt`) and does not re-license the third-party adapter. The
`residual-sinkhorn` wheel is not redistributed and no license is asserted over it;
users obtain it from the author's pinned URL and use it under the author's terms.

## Contact and coordination

The reproduction request that prompted this package is
[Issue #934 comment 7124596382](https://github.com/PythonOT/POT/issues/934#issuecomment-7124596382).
A short GitHub-facing summary is maintained outside this package as a draft that requires
explicit authorization before publication; it is intentionally **not** included here.
