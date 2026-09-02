# Attempt 1 evaluation

All values use the complete I80 prediction validation split (times 64-107).
The held-out test split was not evaluated. Fitness is validation
`E_data + 0.01 * tree_nodes`; lower is better. This is a nonlocal
convolution candidate using the `row_abs_jacobian_sum` speed bound.

| Baseline | Expression | Parameters | E_rho | E_v | E_data | Fitness | Finite/feasible | Runtime (s) | Peak RSS (MB) |
|---|---|---|---:|---:|---:|---:|:---:|---:|---:|
| Greenshields | `exp(a*conv_1(rho,ones))` | a=-4.693158198 | 10.017974 | 7.653662 | 8.835818 | 8.895818 | yes | 0.404 | 1013.4 |
| IDM | `exp(a*conv_1(rho,ones))` | a=-23.93975411 | 6.905225 | 8.493686 | 7.699455 | 7.759455 | yes | 0.969 | 3163.0 |
| Weidmann | `exp(a*conv_1(rho,ones))` | a=0.0009458826666 | 5.796285 | 7.163875 | 6.480080 | 6.540080 | yes | 0.184 | 1244.9 |
| Triangular | `exp(a*conv_1(rho,ones))` | a=-15.19070443 | 6.501988 | 8.974314 | 7.738151 | 7.798151 | yes | 0.199 | 1562.3 |
| Del Castillo | `exp(a*conv_1(rho,ones))` | a=-6.349669457 | 6.038437 | 7.399783 | 6.719110 | 6.779110 | yes | 0.270 | 1885.6 |

Complexity penalty: 0.06.
`test_evaluated=false`; `nonlocal=true`;
`speed_bound=row_abs_jacobian_sum`.
