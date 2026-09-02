# Attempt 3 evaluation

All values use the complete I80 prediction validation split (times 64-107).
The held-out test split was not evaluated. Fitness is validation
`E_data + 0.01 * tree_nodes`; lower is better. This is a nonlocal
convolution candidate using the `row_abs_jacobian_sum` speed bound.

| Baseline | Expression | Parameters | E_rho | E_v | E_data | Fitness | Finite/feasible | Runtime (s) | Peak RSS (MB) |
|---|---|---|---:|---:|---:|---:|:---:|---:|---:|
| Greenshields | `exp(a*conv_1(rho,ones)+b*conv_3(rho,ones))` | a=-4.710309853, b=-0.1568147475 | 9.856059 | 7.667631 | 8.761845 | 8.881845 | yes | 0.343 | 3191.7 |
| IDM | `exp(a*conv_1(rho,ones)+b*conv_3(rho,ones))` | a=-23.96934913, b=-0.7346754944 | 6.919096 | 8.726273 | 7.822684 | 7.942684 | yes | 0.858 | 3287.2 |
| Weidmann | `exp(a*conv_1(rho,ones)+b*conv_3(rho,ones))` | a=58.55936855, b=-19.51949472 | 5.391590 | 7.111099 | 6.251344 | 6.371344 | yes | 0.356 | 3474.6 |
| Triangular | `exp(a*conv_1(rho,ones)+b*conv_3(rho,ones))` | a=436.6349402, b=-149.1320997 | 5.446793 | 7.688803 | 6.567798 | 6.687798 | yes | 0.355 | 3755.1 |
| Del Castillo | `exp(a*conv_1(rho,ones)+b*conv_3(rho,ones))` | a=-6.346689871, b=-0.4440752208 | 6.039652 | 7.846522 | 6.943088 | 7.063088 | yes | 0.387 | 4132.4 |

Complexity penalty: 0.12.
`test_evaluated=false`; `nonlocal=true`;
`speed_bound=row_abs_jacobian_sum`.
