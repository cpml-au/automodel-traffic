# Meta-iteration 3 root review

Meta 3 completed 45 fits across direct convolution, explicit Hodge-star, and
incumbent-plus-convolution lineages. Forty-four final candidates were feasible;
one Weidmann hybrid was rejected for non-monotone homogeneous-state velocity.
No test data was evaluated. Root re-evaluation reproduced all three winners.

| Baseline | Selected multiplier after meta 3 | Train E_data | Validation E_data | Validation fitness | Previous fitness |
|---|---|---:|---:|---:|---:|
| Greenshields | `g_inc*exp(-293.458109*(conv_3-3*conv_1))` | 6.334473 | 6.703732 | 6.953732 | 7.627391 |
| IDM | `exp(-46.991436*(conv_3-3*conv_1))` | 8.265197 | 5.277018 | 5.417018 | 5.955012 |
| Weidmann | retained jam-anchored incumbent | 7.205753 | 6.222293 | 6.322293 | 6.322293 |
| Triangular | identity | 9.223384 | 6.447489 | 6.457489 | 6.457489 |
| Del Castillo | `g_inc*exp(-2.121812*conv_3-256.869230*(conv_3-3*conv_1))` | 6.364161 | 5.520221 | 5.730221 | 6.603956 |

## Findings

- The level-cancelling `conv_3 - 3*conv_1` feature is the decisive DEC term. It
  improves Greenshields, IDM, and Del Castillo under the corrected nonlocal
  Rusanov speed bound.
- Direct raw convolution does not beat an incumbent; combining raw level and
  contrast helps Del Castillo only.
- All explicit Hodge-star expressions compiled and ran, confirming the grammar
  repair, but none beat an incumbent. The closest was a metric-weighted
  quadratic for Triangular, 0.0654 fitness above identity.
- Weidmann and Triangular remain unchanged. Meta 4 should refine queue contrasts
  with density gates and retain a non-convolution nonlinear control family.

