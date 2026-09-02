# Meta-iteration 1 root review

All three lineages completed 3 attempts across all 5 basic FDs (45 fits total).
Every saved final fit was finite and feasible; no test split was evaluated. The
root process re-evaluated the shortlist with the shared evaluator.

| Baseline | Identity validation fitness | Meta-1 winner | Validation E_data | Validation fitness | Decision |
|---|---:|---|---:|---:|---|
| Greenshields | 9.620405 | `exp(a*rho+b*rho^2+c*rho^3)` | 8.045311 | 8.205311 | keep |
| IDM | 5.955012 | identity | 5.945012 | 5.955012 | keep identity |
| Weidmann | 6.490163 | `exp(a*rho*(1-rho/r_j))` | 6.222293 | 6.322293 | keep |
| Triangular | 6.457489 | identity | 6.447489 | 6.457489 | keep identity |
| Del Castillo | 6.638030 | identity | 6.628030 | 6.638030 | keep identity; saturating gain is smaller than penalty |

Root re-evaluation reproduced the Greenshields and Weidmann scores exactly. The
Del Castillo saturating candidate re-evaluated at `E_data=6.610250` and
`fitness=6.690250`, still worse than identity after complexity.

## Findings

- Positive local corrections can improve Greenshields substantially and preserve
  physical feasibility.
- A one-parameter jam-anchored term improves both density and velocity components
  for Weidmann.
- Corrections fitted on train commonly hurt validation for IDM and Triangular;
  additional terms generally amplify this rather than help.
- Del Castillo has only a tiny unpenalized gain from a saturating term. A simpler
  structure or a term with an intercept is needed to clear the complexity cost.
- Many multi-parameter Powell fits reached the 30-evaluation cap. Meta 2 must use
  a larger budget for shortlisted structures and test simpler reductions.

## Meta-iteration 2 directions

1. Add a global intercept to exponential polynomials to decouple scale and shape.
2. Center exponent bases at each FD's critical/physical density to improve
   conditioning and reduce train-to-validation drift.
3. Try direct low-node polynomial/rational multipliers whose smaller complexity
   penalty may retain marginal validation gains.

