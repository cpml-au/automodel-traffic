# Attempt 3 training

All incumbent coefficients were fixed. New constants used two deterministic
Powell restarts and at most 60 evaluations per restart.

| FD | Full expression | Nodes | Parameters | E_rho | E_v | E_data | Evals | Feasible | Runtime (s) | RSS (MB) | Seed |
|---|---|---:|---|---:|---:|---:|---:|:---:|---:|---:|---:|
| Greenshields | `(exp(0.0231229711605-0.0883173972597*rho-0.15008122102*rho^2)*exp(-293.458109123*(conv_3-3*conv_1)))*exp(a*(conv_3-3*conv_1)+b*hquad)` | 45 | a=-6.362556798, b=26.72142449 | 7.696022 | 4.853293 | 6.274657 | 119 | yes | 56.990 | 1247.8 | 11130 |
| IDM | `(exp(-46.9914359751*(conv_3-3*conv_1)))*exp(a*(conv_3-3*conv_1)+b*hquad)` | 34 | a=0.09895592915, b=-45.27079703 | 6.891514 | 7.958031 | 7.424772 | 119 | yes | 308.871 | 3481.3 | 11131 |
| Weidmann | `(exp(0.136544052144*rho*(1-rho/0.80612097)))*exp(a*(conv_3-3*conv_1)+b*hquad)` | 30 | a=-299.9991404, b=-1757.434601 | 13.280429 | 58.134773 | 35.707600 | 120 | yes | 23.142 | 3512.8 | 11132 |
| Triangular | `(1)*exp(a*(conv_3-3*conv_1)+b*hquad)` | 21 | a=50.98755103, b=-3.328582147 | 7.316834 | 9.430723 | 8.373778 | 120 | yes | 46.407 | 3808.3 | 11133 |
| Del Castillo | `(exp(0.0054952026563)*exp(-2.12181160185*conv_3-256.869229508*(conv_3-3*conv_1)))*exp(a*(conv_3-3*conv_1)+b*hquad)` | 41 | a=-0.2062989687, b=3.00290988 | 7.259736 | 5.466441 | 6.363089 | 88 | yes | 38.792 | 4104.2 | 11134 |

`test_evaluated = false`.
