# Attempt 2 evaluation

All values use the complete I80 prediction validation split (times 64-107).
The held-out test split was not evaluated. Fitness is validation
`E_data + 0.01 * tree_nodes`; lower is better. This is a nonlocal
convolution candidate using the `row_abs_jacobian_sum` speed bound.

| Baseline | Expression | Parameters | E_rho | E_v | E_data | Fitness | Finite/feasible | Runtime (s) | Peak RSS (MB) |
|---|---|---|---:|---:|---:|---:|:---:|---:|---:|
| Greenshields | `exp(a*conv_3(rho,ones))` | a=-1.662726641 | 9.851598 | 7.643034 | 8.747316 | 8.807316 | yes | 0.298 | 2182.0 |
| IDM | `exp(a*conv_3(rho,ones))` | a=-8.669891236 | 6.774052 | 8.576088 | 7.675070 | 7.735070 | yes | 0.735 | 3231.7 |
| Weidmann | `exp(a*conv_3(rho,ones))` | a=-0.0003757845476 | 5.796240 | 7.164296 | 6.480268 | 6.540268 | yes | 0.297 | 2394.1 |
| Triangular | `exp(a*conv_3(rho,ones))` | a=-5.089466615 | 6.439287 | 8.991336 | 7.715312 | 7.775312 | yes | 0.322 | 2702.4 |
| Del Castillo | `exp(a*conv_3(rho,ones))` | a=-2.490014657 | 6.015738 | 7.849091 | 6.932414 | 6.992414 | yes | 0.371 | 3002.5 |

Complexity penalty: 0.06.
`test_evaluated=false`; `nonlocal=true`;
`speed_bound=row_abs_jacobian_sum`.
