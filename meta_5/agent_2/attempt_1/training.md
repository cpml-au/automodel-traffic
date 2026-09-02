# Attempt 1 training

Factor: `exp(a*rho*C)`. Full train times 0--63; incumbent constants fixed.

| FD | Full expression | Nodes | New parameters | E_rho | E_v | E_data | Evals | Optimizer | Feasible | Fit time (s) | RSS (MB) | Seed |
|---|---|---:|---|---:|---:|---:|---:|---|:---:|---:|---:|---:|
| Greenshields | `(exp(0.0231229711605-0.0883173972597*rho-0.15008122102*rho^2)*exp(-293.458109123*C))*exp(a*rho*C)` | 40 | a=1119.626208 | 7.414973 | 5.893704 | 6.654338 | 84 | success: Optimization terminated successfully. | yes | 45.466 | 1185.2 | 13110 |
| IDM | `(exp(-46.9914359751*C)*exp(-119537.388349*hquad*C))*exp(a*rho*C)` | 47 | a=124.395271 | 6.586385 | 8.744513 | 7.665449 | 89 | success: Optimization terminated successfully. | yes | 261.251 | 3477.8 | 13111 |
| Weidmann | `(exp(0.136544052144*rho*(1-rho/0.80612097)))*exp(a*rho*C)` | 25 | a=2999.998768 | 6568.004883 | inf | inf | 66 | success: Optimization terminated successfully. | no | 7.760 | 3478.9 | 13112 |
| Triangular | `(1)*exp(a*rho*C)` | 16 | a=-921.9701998 | 6.890831 | 9.039266 | 7.965049 | 120 | stopped: Maximum number of function evaluations has been exceeded. | yes | 49.201 | 3815.3 | 13113 |
| Del Castillo | `(exp(0.0054952026563)*exp(-2.12181160185*conv_3-256.869229508*C))*exp(a*rho*C)` | 36 | a=994.5579335 | 7.320286 | 6.122615 | 6.721450 | 75 | success: Optimization terminated successfully. | yes | 45.411 | 4204.6 | 13114 |

`test_evaluated = false`.
