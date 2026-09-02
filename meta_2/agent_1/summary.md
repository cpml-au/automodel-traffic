# Meta 2 agent 1 summary

All 15 prescribed full-train fits and full-validation evaluations completed.
Every selected fit was finite and passed the shared physical feasibility check.
The held-out test split was not evaluated.

| Baseline | Best intercept-family expression | Parameters | Validation E_data | Validation fitness | Meta-1 incumbent fitness | Beats incumbent? |
|---|---|---|---:|---:|---:|:---:|
| Greenshields | `exp(c0 + a*rho + b*rho^2)` | `c0=0.02312297116, a=-0.08831739726, b=-0.1500812210` | 7.507391 | 7.627391 | 8.205311 | yes |
| IDM | `exp(c0)` | `c0=-0.1883932724` | 6.208821 | 6.228821 | 5.955012 | no |
| Weidmann | `exp(c0)` | `c0=-0.01971739213` | 6.610597 | 6.630597 | 6.322293 | no |
| Triangular | `exp(c0)` | `c0=-0.09737332019` | 7.508256 | 7.528256 | 6.457489 | no |
| Del Castillo | `exp(c0)` | `c0=0.005495202656` | 6.583956 | 6.603956 | 6.638030 | yes |

## Interpretation

- The quadratic intercept family substantially improves Greenshields beyond
  the meta-1 cubic no-intercept incumbent, despite using fewer tree nodes
  (12 versus 16).
- A two-node constant log-scale correction produces a small but penalty-aware
  improvement for Del Castillo.
- The fitted intercept families do not transfer well enough to displace the
  IDM identity, the jam-anchored Weidmann correction, or the Triangular identity.
- The complete expression history, training components, validation components,
  optimizer status/evaluation counts, runtimes, peak RSS, feasibility, seeds,
  and parameters are retained in each attempt's `results.json` and reports.

