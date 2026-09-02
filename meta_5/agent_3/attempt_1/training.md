# Attempt 1 training

All global incumbent coefficients were held fixed. Only the new constants
used two deterministic Powell restarts and at most 60
evaluations per restart on full train times 0--63.

| FD | Full expression | Nodes | Parameters | E_rho | E_v | E_data | Optimizer | Evals | Feasible | Runtime (s) | RSS (MB) | Seed |
|---|---|---:|---|---:|---:|---:|---|---:|:---:|---:|---:|---:|
| Greenshields | `(exp(0.0231229711605-0.0883173972597*rho-0.15008122102*rho^2)*exp(-293.458109123*C))*exp(a*C+b*hquad*C)` | 55 | a=48.62308613, b=-42852.50609 | 7.333546 | 5.176506 | 6.255026 | stopped: Maximum number of function evaluations has been exceeded. | 120 | yes | 69.025 | 1426.7 | 13110 |
| IDM | `(exp(-46.9914359751*C)*exp(-119537.388349*hquad*C))*exp(a*C+b*hquad*C)` | 62 | a=86.08646158, b=-29600.21354 | 6.365540 | 8.445823 | 7.405681 | stopped: Maximum number of function evaluations has been exceeded. | 100 | yes | 254.728 | 3633.9 | 13111 |
| Weidmann | `(exp(0.136544052144*rho*(1-rho/0.80612097)))*exp(a*C+b*hquad*C)` | 40 | a=0, b=0 | 8.205365 | 6.206141 | 7.205753 | stopped: No finite feasible point found; zero/incumbent fallback evaluated. | 82 | no | 9.812 | 587.0 | 13112 |
| Triangular | `(1)*exp(a*C+b*hquad*C)` | 31 | a=50.99860897, b=0 | 7.336160 | 9.429861 | 8.383011 | success: Optimization terminated successfully. | 82 | yes | 23.912 | 899.0 | 13113 |
| Del Castillo | `(exp(0.0054952026563)*exp(-2.12181160185*conv_3(rho,ones)-256.869229508*C))*exp(a*C+b*hquad*C)` | 51 | a=71.14073361, b=-54261.81513 | 6.994307 | 5.470668 | 6.232488 | stopped: Maximum number of function evaluations has been exceeded. | 100 | yes | 56.235 | 3845.4 | 13114 |

`test_evaluated = false`.
