# Attempt 1 evaluation

Selection uses the complete I80 validation split and `E_fitness = E_data + 0.01 * tree_nodes`; lower is better.

| Baseline | r* | Exact expression | Nodes | E_rho | E_v | E_data | E_fitness | Validation runtime (s) | Peak RSS (MB) | Finite | Feasible | Seed |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---|---|---:|
| greenshields | 0.279975615 | `exp(a*(rho-0.279975615))` | 6 | 6.290743 | 10.275656 | 8.283199 | 8.343199 | 0.380 | 829.51 | yes | yes | 5110 |
| idm | 0.347375764723 | `exp(a*(rho-0.347375764723))` | 6 | 19.334074 | 5.670273 | 12.502173 | 12.562173 | 0.525 | 1282.04 | yes | yes | 5111 |
| weidmann | 0.403060485 | `exp(a*(rho-0.403060485))` | 6 | 14.448137 | 8.516461 | 11.482300 | 11.542300 | 0.148 | 1499.62 | yes | yes | 5112 |
| triangular | 0.335649971536 | `exp(a*(rho-0.335649971536))` | 6 | 9.799481 | 7.311349 | 8.555415 | 8.615415 | 0.164 | 1749.59 | yes | yes | 5113 |
| del_castillo | 0.307660845 | `exp(a*(rho-0.307660845))` | 6 | 19.613121 | 5.592000 | 12.602561 | 12.662561 | 0.262 | 2063.37 | yes | yes | 5114 |

Failures: none.

Diagnostics: positivity/finiteness and non-negative, non-increasing corrected velocity were checked over 79 points on each FD's physical density domain by the shared feasibility routine.

`test_evaluated = false`; no test prediction or score was computed.
