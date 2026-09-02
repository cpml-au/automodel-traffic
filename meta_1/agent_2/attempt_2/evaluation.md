# Attempt 2 evaluation

Selection uses the complete I80 validation split and `E_fitness = E_data + 0.01 * tree_nodes`. Lower is better.

| Rank | Baseline | Expression | Nodes | E_rho | E_v | E_data | E_fitness | Finite | Feasible | Seed |
|---:|---|---|---:|---:|---:|---:|---:|---|---|---:|
| 1 | del_castillo | `exp((a*rho+b*rho^2)*(1-rho/0.61532169))` | 15 | 6.050886 | 7.443784 | 6.747335 | 6.897335 | yes | yes | 2124 |
| 2 | idm | `exp((a*rho+b*rho^2)*(1-rho/0.694751529445))` | 15 | 6.332852 | 8.239949 | 7.286401 | 7.436401 | yes | yes | 2121 |
| 3 | weidmann | `exp((a*rho+b*rho^2)*(1-rho/0.80612097))` | 15 | 7.761977 | 6.900209 | 7.331093 | 7.481093 | yes | yes | 2122 |
| 4 | greenshields | `exp((a*rho+b*rho^2)*(1-rho/0.55995123))` | 15 | 9.938209 | 7.595424 | 8.766816 | 8.916816 | yes | yes | 2120 |
| 5 | triangular | `exp((a*rho+b*rho^2)*(1-rho/0.671299943071))` | 15 | 12.831498 | 7.996794 | 10.414146 | 10.564146 | yes | yes | 2123 |

Failures: none.

No test prediction or score was computed.
