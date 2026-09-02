# Attempt 3 training

All global incumbent coefficients were held fixed. Only the new constants
used two deterministic Powell restarts and at most 60
evaluations per restart on full train times 0--63.

| FD | Full expression | Nodes | Parameters | E_rho | E_v | E_data | Optimizer | Evals | Feasible | Runtime (s) | RSS (MB) | Seed |
|---|---|---:|---|---:|---:|---:|---|---:|:---:|---:|---:|---:|
| Greenshields | `(exp(0.0231229711605-0.0883173972597*rho-0.15008122102*rho^2)*exp(-293.458109123*C))*exp(a*(79*hquad-rho^2)*C)` | 48 | a=0 | 7.518588 | 5.150358 | 6.334473 | success: Optimization terminated successfully. | 44 | yes | 5.949 | 640.2 | 13130 |
| IDM | `(exp(-46.9914359751*C)*exp(-119537.388349*hquad*C))*exp(a*(79*hquad-rho^2)*C)` | 55 | a=0 | 6.585182 | 8.773269 | 7.679225 | success: Optimization terminated successfully. | 44 | yes | 36.906 | 1451.7 | 13131 |
| Weidmann | `(exp(0.136544052144*rho*(1-rho/0.80612097)))*exp(a*(79*hquad-rho^2)*C)` | 33 | a=0 | 8.205365 | 6.206141 | 7.205753 | stopped: No finite feasible point found; zero/incumbent fallback evaluated. | 44 | no | 2.161 | 1520.5 | 13132 |
| Triangular | `(1)*exp(a*(79*hquad-rho^2)*C)` | 24 | a=0 | 7.853766 | 10.593001 | 9.223384 | success: Optimization terminated successfully. | 44 | yes | 3.375 | 1557.0 | 13133 |
| Del Castillo | `(exp(0.0054952026563)*exp(-2.12181160185*conv_3(rho,ones)-256.869229508*C))*exp(a*(79*hquad-rho^2)*C)` | 44 | a=0 | 7.237927 | 5.490397 | 6.364162 | success: Optimization terminated successfully. | 44 | yes | 4.660 | 1600.9 | 13134 |

`test_evaluated = false`.
