# Attempt 3 evaluation

Selection uses the complete I80 validation split and `E_fitness = E_data + 0.01 * tree_nodes`; lower is better.

| Baseline | r* | Exact expression | Nodes | E_rho | E_v | E_data | E_fitness | Validation runtime (s) | Peak RSS (MB) | Finite | Feasible | Seed |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---|---|---:|
| greenshields | 0.279975615 | `exp(a*(rho-0.279975615) + b*(rho-0.279975615)^2 + c*(rho-0.279975615)^3)` | 22 | 6.306548 | 10.235648 | 8.271098 | 8.491098 | 0.161 | 3867.53 | yes | yes | 5130 |
| idm | 0.347375764723 | `exp(a*(rho-0.347375764723) + b*(rho-0.347375764723)^2 + c*(rho-0.347375764723)^3)` | 22 | 19.315201 | 5.960401 | 12.637800 | 12.857800 | 0.585 | 4372.08 | yes | yes | 5131 |
| weidmann | 0.403060485 | `exp(a*(rho-0.403060485) + b*(rho-0.403060485)^2 + c*(rho-0.403060485)^3)` | 22 | 13.661269 | 8.438970 | 11.050119 | 11.270119 | 0.201 | 4637.81 | yes | yes | 5132 |
| triangular | 0.335649971536 | `exp(a*(rho-0.335649971536) + b*(rho-0.335649971536)^2 + c*(rho-0.335649971536)^3)` | 22 | 7.523218 | 6.482409 | 7.002813 | 7.222813 | 0.178 | 4903.56 | yes | yes | 5133 |
| del_castillo | 0.307660845 | `exp(a*(rho-0.307660845) + b*(rho-0.307660845)^2 + c*(rho-0.307660845)^3)` | 22 | 6.403678 | 8.915874 | 7.659776 | 7.879776 | 0.269 | 5278.92 | yes | yes | 5134 |

Failures: none.

Diagnostics: positivity/finiteness and non-negative, non-increasing corrected velocity were checked over 79 points on each FD's physical density domain by the shared feasibility routine.

`test_evaluated = false`; no test prediction or score was computed.
