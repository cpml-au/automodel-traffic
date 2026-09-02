# Attempt 3 evaluation

Selection uses the complete I80 validation split and `E_fitness = E_data + 0.01 * tree_nodes`. Lower is better.

| Rank | Baseline | Expression | Nodes | E_rho | E_v | E_data | E_fitness | Finite | Feasible | Seed |
|---:|---|---|---:|---:|---:|---:|---:|---|---|---:|
| 1 | weidmann | `exp((a*rho+b*rho^2+c*rho^3)*(1-rho/0.80612097))` | 21 | 5.923585 | 6.521000 | 6.222293 | 6.432293 | yes | yes | 2132 |
| 2 | idm | `exp((a*rho+b*rho^2+c*rho^3)*(1-rho/0.694751529445))` | 21 | 6.711102 | 6.083894 | 6.397498 | 6.607498 | yes | yes | 2131 |
| 3 | del_castillo | `exp((a*rho+b*rho^2+c*rho^3)*(1-rho/0.61532169))` | 21 | 6.050886 | 7.443784 | 6.747335 | 6.957335 | yes | yes | 2134 |
| 4 | triangular | `exp((a*rho+b*rho^2+c*rho^3)*(1-rho/0.671299943071))` | 21 | 6.540417 | 7.919237 | 7.229827 | 7.439827 | yes | yes | 2133 |
| 5 | greenshields | `exp((a*rho+b*rho^2+c*rho^3)*(1-rho/0.55995123))` | 21 | 9.938174 | 7.596918 | 8.767546 | 8.977546 | yes | yes | 2130 |

Failures: none.

No test prediction or score was computed.
