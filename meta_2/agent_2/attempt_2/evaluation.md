# Attempt 2 evaluation

Selection uses the complete I80 validation split and `E_fitness = E_data + 0.01 * tree_nodes`; lower is better.

| Baseline | r* | Exact expression | Nodes | E_rho | E_v | E_data | E_fitness | Validation runtime (s) | Peak RSS (MB) | Finite | Feasible | Seed |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---|---|---:|
| greenshields | 0.279975615 | `exp(a*(rho-0.279975615) + b*(rho-0.279975615)^2)` | 14 | 19.568201 | 5.569127 | 12.568665 | 12.708665 | 0.153 | 2288.91 | yes | yes | 5120 |
| idm | 0.347375764723 | `exp(a*(rho-0.347375764723) + b*(rho-0.347375764723)^2)` | 14 | 19.301783 | 5.752895 | 12.527339 | 12.667339 | 0.564 | 2796.58 | yes | yes | 5121 |
| weidmann | 0.403060485 | `exp(a*(rho-0.403060485) + b*(rho-0.403060485)^2)` | 14 | 13.946380 | 8.477074 | 11.211727 | 11.351727 | 0.162 | 3056.24 | yes | yes | 5122 |
| triangular | 0.335649971536 | `exp(a*(rho-0.335649971536) + b*(rho-0.335649971536)^2)` | 14 | 6.636270 | 6.470028 | 6.553149 | 6.693149 | 0.165 | 3312.15 | yes | yes | 5123 |
| del_castillo | 0.307660845 | `exp(a*(rho-0.307660845) + b*(rho-0.307660845)^2)` | 14 | 19.715843 | 5.630664 | 12.673254 | 12.813254 | 0.294 | 3628.21 | yes | yes | 5124 |

Failures: none.

Diagnostics: positivity/finiteness and non-negative, non-increasing corrected velocity were checked over 79 points on each FD's physical density domain by the shared feasibility routine.

`test_evaluated = false`; no test prediction or score was computed.
