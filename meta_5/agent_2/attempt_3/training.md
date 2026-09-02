# Attempt 3 training

Factor: `exp(a*rho*C+b*rho^2*C)`. Full train times 0--63; incumbent constants fixed.

| FD | Full expression | Nodes | New parameters | E_rho | E_v | E_data | Evals | Optimizer | Feasible | Fit time (s) | RSS (MB) | Seed |
|---|---|---:|---|---:|---:|---:|---:|---|:---:|---:|---:|---:|
| Greenshields | `(exp(0.0231229711605-0.0883173972597*rho-0.15008122102*rho^2)*exp(-293.458109123*C))*exp(a*rho*C+b*rho^2*C)` | 55 | a=1119.680194, b=0 | 7.414944 | 5.893732 | 6.654338 | 120 | stopped: Maximum number of function evaluations has been exceeded. | yes | 79.063 | 1472.7 | 13130 |
| IDM | `(exp(-46.9914359751*C)*exp(-119537.388349*hquad*C))*exp(a*rho*C+b*rho^2*C)` | 62 | a=124.0956041, b=0 | 6.586371 | 8.744529 | 7.665450 | 120 | stopped: Maximum number of function evaluations has been exceeded. | yes | 317.207 | 3683.9 | 13131 |
| Weidmann | `(exp(0.136544052144*rho*(1-rho/0.80612097)))*exp(a*rho*C+b*rho^2*C)` | 40 | a=2999.998768, b=0 | 6568.006836 | inf | inf | 120 | stopped: Maximum number of function evaluations has been exceeded. | no | 18.584 | 3688.4 | 13132 |
| Triangular | `(1)*exp(a*rho*C+b*rho^2*C)` | 31 | a=1887.817768, b=-6170.114358 | 6.706952 | 8.751342 | 7.729147 | 120 | stopped: Maximum number of function evaluations has been exceeded. | yes | 60.393 | 3766.0 | 13133 |
| Del Castillo | `(exp(0.0054952026563)*exp(-2.12181160185*conv_3-256.869229508*C))*exp(a*rho*C+b*rho^2*C)` | 51 | a=994.4396655, b=0 | 7.320292 | 6.122607 | 6.721449 | 120 | stopped: Maximum number of function evaluations has been exceeded. | yes | 59.669 | 4081.4 | 13134 |

`test_evaluated = false`.
