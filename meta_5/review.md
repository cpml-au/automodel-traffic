# Meta-iteration 5 root review

Meta 5 completed 45 fits across nonlinear convolution contrasts, density-gated
contrasts, and mixed repaired-Hodge/convolution refinements. Thirty-three final
candidates were feasible; twelve pathological or non-monotone endpoints were
rejected. No test data was evaluated during selection.

Two Triangular candidates beat its identity incumbent. Independent JAX x64
evaluation reproduced both candidates and their feasibility:

| Candidate | Nodes | Validation E_data | Validation fitness |
|---|---:|---:|---:|
| `exp(-921.970200*rho*C)` | 16 | 5.067495 | **5.227495** |
| `exp(-207.794278*C + 14690.289134*C^2)` | 27 | 5.138736 | 5.408736 |

Here `C = conv_3(rho,ones) - 3*conv_1(rho,ones)`. The density-gated candidate
is selected because it has both lower unpenalized error and lower complexity.

| Baseline | Selected multiplier after meta 5 | Train E_data | Validation E_data | Validation fitness | Previous fitness |
|---|---|---:|---:|---:|---:|
| Greenshields | retained meta-3 convolution contrast | 6.334473 | 6.703732 | 6.953732 | 6.953732 |
| IDM | retained meta-4 Hodge-gated contrast | 7.679225 | 4.728745 | 5.048745 | 5.048745 |
| Weidmann | retained jam-anchored pointwise term | 7.205753 | 6.222293 | 6.322293 | 6.322293 |
| Triangular | `exp(-921.970200*rho*C)` | 7.965049 | 5.067508 | 5.227508 | 6.457489 |
| Del Castillo | retained meta-3 convolution model | 6.364161 | 5.520221 | 5.730221 | 5.730221 |

## Findings and stopping rationale

- Density-gating the level-cancelling convolution contrast supplies the first
  large Triangular improvement: validation fitness falls 19.05% from identity.
- A signed-plus-quadratic contrast independently corroborates that Triangular
  benefits from spatial curvature, but its larger tree loses to the density gate.
- Nonlinear contrast terms did not improve the other four incumbents after the
  complexity penalty. Mixed Hodge/convolution refinements improved some raw
  validation errors but likewise did not earn their added nodes.
- The user requested exactly one further meta-iteration followed by Phase 4.
  The five structures in `automodel/final_candidates.json` are therefore frozen
  before any train+validation refitting or held-out test access.
