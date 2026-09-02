# Attempt 1 training

All incumbent coefficients were fixed. New constants used two deterministic
Powell restarts and at most 60 evaluations per restart.

| FD | Full expression | Nodes | Parameters | E_rho | E_v | E_data | Evals | Feasible | Runtime (s) | RSS (MB) | Seed |
|---|---|---:|---|---:|---:|---:|---:|:---:|---:|---:|---:|
| Greenshields | `(exp(0.0231229711605-0.0883173972597*rho-0.15008122102*rho^2)*exp(-293.458109123*(conv_3-3*conv_1)))*exp(a*hlin*(conv_3-3*conv_1))` | 42 | a=-102.8027683 | 7.468549 | 5.151144 | 6.309846 | 116 | yes | 66.900 | 1384.4 | 11110 |
| IDM | `(exp(-46.9914359751*(conv_3-3*conv_1)))*exp(a*hlin*(conv_3-3*conv_1))` | 31 | a=-443.270414 | 6.843102 | 9.276692 | 8.059897 | 98 | yes | 292.694 | 3569.8 | 11111 |
| Weidmann | `(exp(0.136544052144*rho*(1-rho/0.80612097)))*exp(a*hlin*(conv_3-3*conv_1))` | 27 | a=999.9989421 | 34.823795 | inf | inf | 62 | no | 8.045 | 3572.8 | 11112 |
| Triangular | `(1)*exp(a*hlin*(conv_3-3*conv_1))` | 18 | a=190.7181361 | 7.387576 | 9.393833 | 8.390704 | 87 | yes | 40.202 | 3718.7 | 11113 |
| Del Castillo | `(exp(0.0054952026563)*exp(-2.12181160185*conv_3-256.869229508*(conv_3-3*conv_1)))*exp(a*hlin*(conv_3-3*conv_1))` | 38 | a=-181.0737941 | 7.173889 | 5.450612 | 6.312251 | 114 | yes | 74.707 | 4313.1 | 11114 |

`test_evaluated = false`.
