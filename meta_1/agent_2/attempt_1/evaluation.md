# Attempt 1 evaluation

Selection uses the complete I80 validation split and `E_fitness = E_data + 0.01 * tree_nodes`. Lower is better.

| Rank | Baseline | Expression | Nodes | E_rho | E_v | E_data | E_fitness | Finite | Feasible | Seed |
|---:|---|---|---:|---:|---:|---:|---:|---|---|---:|
| 1 | weidmann | `exp(a*rho*(1-rho/0.80612097))` | 10 | 5.923585 | 6.521000 | 6.222293 | 6.322293 | yes | yes | 2112 |
| 2 | del_castillo | `exp(a*rho*(1-rho/0.61532169))` | 10 | 6.753607 | 6.547346 | 6.650476 | 6.750476 | yes | yes | 2114 |
| 3 | triangular | `exp(a*rho*(1-rho/0.671299943071))` | 10 | 6.539398 | 9.179962 | 7.859680 | 7.959680 | yes | yes | 2113 |
| 4 | idm | `exp(a*rho*(1-rho/0.694751529445))` | 10 | 6.621785 | 9.723469 | 8.172626 | 8.272626 | yes | yes | 2111 |
| 5 | greenshields | `exp(a*rho*(1-rho/0.55995123))` | 10 | 11.619335 | 7.637562 | 9.628448 | 9.728448 | yes | yes | 2110 |

Failures: none.

No test prediction or score was computed.
