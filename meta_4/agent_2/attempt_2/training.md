# Attempt 2 training

All incumbent coefficients were fixed. New constants used two deterministic
Powell restarts and at most 60 evaluations per restart.

| FD | Full expression | Nodes | Parameters | E_rho | E_v | E_data | Evals | Feasible | Runtime (s) | RSS (MB) | Seed |
|---|---|---:|---|---:|---:|---:|---:|:---:|---:|---:|---:|
| Greenshields | `(exp(0.0231229711605-0.0883173972597*rho-0.15008122102*rho^2)*exp(-293.458109123*(conv_3-3*conv_1)))*exp(a*hquad*(conv_3-3*conv_1))` | 43 | a=-30813.75332 | 7.406340 | 5.160738 | 6.283539 | 120 | yes | 69.031 | 1427.9 | 11120 |
| IDM | `(exp(-46.9914359751*(conv_3-3*conv_1)))*exp(a*hquad*(conv_3-3*conv_1))` | 32 | a=-119537.3883 | 6.585182 | 8.773269 | 7.679225 | 120 | yes | 352.943 | 3579.9 | 11121 |
| Weidmann | `(exp(0.136544052144*rho*(1-rho/0.80612097)))*exp(a*hquad*(conv_3-3*conv_1))` | 28 | a=999999.9785 | 4934147.000000 | inf | inf | 77 | no | 7.568 | 3583.2 | 11122 |
| Triangular | `(1)*exp(a*hquad*(conv_3-3*conv_1))` | 19 | a=53932.07703 | 7.443946 | 9.430357 | 8.437152 | 120 | yes | 52.859 | 3847.2 | 11123 |
| Del Castillo | `(exp(0.0054952026563)*exp(-2.12181160185*conv_3-256.869229508*(conv_3-3*conv_1)))*exp(a*hquad*(conv_3-3*conv_1))` | 39 | a=-54288.2137 | 7.073241 | 5.436813 | 6.255027 | 120 | yes | 72.809 | 4472.4 | 11124 |

`test_evaluated = false`.
