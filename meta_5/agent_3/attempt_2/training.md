# Attempt 2 training

All global incumbent coefficients were held fixed. Only the new constants
used two deterministic Powell restarts and at most 60
evaluations per restart on full train times 0--63.

| FD | Full expression | Nodes | Parameters | E_rho | E_v | E_data | Optimizer | Evals | Feasible | Runtime (s) | RSS (MB) | Seed |
|---|---|---:|---|---:|---:|---:|---|---:|:---:|---:|---:|---:|
| Greenshields | `(exp(0.0231229711605-0.0883173972597*rho-0.15008122102*rho^2)*exp(-293.458109123*C))*exp(a*hlin*C+b*hquad*C)` | 59 | a=-102.8410728, b=-200.000003 | 7.468164 | 5.151348 | 6.309756 | stopped: Maximum number of function evaluations has been exceeded. | 120 | yes | 73.057 | 1455.7 | 13120 |
| IDM | `(exp(-46.9914359751*C)*exp(-119537.388349*hquad*C))*exp(a*hlin*C+b*hquad*C)` | 66 | a=246.9610852, b=-44303.4881 | 6.492272 | 8.574686 | 7.533479 | stopped: Maximum number of function evaluations has been exceeded. | 120 | yes | 305.095 | 3672.6 | 13121 |
| Weidmann | `(exp(0.136544052144*rho*(1-rho/0.80612097)))*exp(a*hlin*C+b*hquad*C)` | 44 | a=0, b=0 | 8.205365 | 6.206141 | 7.205753 | stopped: No finite feasible point found; zero/incumbent fallback evaluated. | 82 | no | 8.287 | 3672.6 | 13122 |
| Triangular | `(1)*exp(a*hlin*C+b*hquad*C)` | 35 | a=1598.514331, b=-382259.7217 | 6.846574 | 8.575584 | 7.711079 | stopped: Maximum number of function evaluations has been exceeded. | 120 | yes | 49.299 | 3804.6 | 13123 |
| Del Castillo | `(exp(0.0054952026563)*exp(-2.12181160185*conv_3(rho,ones)-256.869229508*C))*exp(a*hlin*C+b*hquad*C)` | 55 | a=1999.641094, b=-452328.8146 | 6.390989 | 5.714113 | 6.052551 | stopped: Maximum number of function evaluations has been exceeded. | 120 | no | 60.858 | 4354.7 | 13124 |

`test_evaluated = false`.
