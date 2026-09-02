# Attempt 2 training

Factor: `exp(a*rho^2*C)`. Full train times 0--63; incumbent constants fixed.

| FD | Full expression | Nodes | New parameters | E_rho | E_v | E_data | Evals | Optimizer | Feasible | Fit time (s) | RSS (MB) | Seed |
|---|---|---:|---|---:|---:|---:|---:|---|:---:|---:|---:|---:|
| Greenshields | `(exp(0.0231229711605-0.0883173972597*rho-0.15008122102*rho^2)*exp(-293.458109123*C))*exp(a*rho^2*C)` | 41 | a=-390.1454983 | 7.406329 | 5.160749 | 6.283539 | 120 | stopped: Maximum number of function evaluations has been exceeded. | yes | 65.546 | 1407.5 | 13120 |
| IDM | `(exp(-46.9914359751*C)*exp(-119537.388349*hquad*C))*exp(a*rho^2*C)` | 48 | a=0.0003736235817 | 6.585188 | 8.773273 | 7.679231 | 99 | success: Optimization terminated successfully. | yes | 287.930 | 3606.0 | 13121 |
| Weidmann | `(exp(0.136544052144*rho*(1-rho/0.80612097)))*exp(a*rho^2*C)` | 26 | a=14999.99855 | 68.068947 | 389906526371840.000000 | 194953263185920.000000 | 72 | success: Optimization terminated successfully. | no | 6.149 | 3609.0 | 13122 |
| Triangular | `(1)*exp(a*rho^2*C)` | 17 | a=14999.99855 | 73.301208 | 757421605818204160.000000 | 378710802909102080.000000 | 72 | success: Optimization terminated successfully. | no | 8.967 | 3615.4 | 13123 |
| Del Castillo | `(exp(0.0054952026563)*exp(-2.12181160185*conv_3-256.869229508*C))*exp(a*rho^2*C)` | 37 | a=4497.764819 | 7.245275 | 5.908467 | 6.576871 | 82 | success: Optimization terminated successfully. | no | 52.521 | 3850.5 | 13124 |

`test_evaluated = false`.
