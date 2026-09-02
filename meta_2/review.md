# Meta-iteration 2 root review

All three lineages completed 3 attempts across all 5 FDs (45 additional fits;
90 total across both meta-iterations). All final fits were finite and feasible,
and no test split was evaluated. The root process reloaded the winning model
files and reproduced their full train/validation scores.

| Baseline | Selected multiplier after meta 2 | Train E_data | Validation E_data | Validation fitness | Identity fitness |
|---|---|---:|---:|---:|---:|
| Greenshields | `exp(0.023123 - 0.088317*rho - 0.150081*rho^2)` | 7.047837 | 7.507391 | 7.627391 | 9.620405 |
| IDM | `1` | 8.438750 | 5.945012 | 5.955012 | 5.955012 |
| Weidmann | `exp(0.136544*rho*(1-rho/0.806121))` | 7.205753 | 6.222293 | 6.322293 | 6.490163 |
| Triangular | `1` | 9.223384 | 6.447489 | 6.457489 | 6.457489 |
| Del Castillo | `exp(0.005495)` | 7.047598 | 6.583956 | 6.603956 | 6.638030 |

## Findings

- An intercept plus quadratic density shape substantially improves Greenshields
  and supersedes the meta-1 cubic-without-intercept correction.
- Del Castillo supports only a tiny global rescaling; density-dependent terms
  overfit and lose validation performance.
- Weidmann's meta-1 jam-anchored correction remains the winner.
- None of the 18 searched structures beats identity for IDM or Triangular after
  fitting solely on the chronological training interval.
- Centering and direct rational forms did not improve any incumbent.

