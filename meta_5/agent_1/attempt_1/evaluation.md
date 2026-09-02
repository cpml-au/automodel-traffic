# Attempt 1 evaluation

Full validation times 64--107 were used. Test times 108--179 were not
evaluated. Fitness is `E_data + 0.01*total_tree_nodes`, including the
fixed incumbent. Nonlocal speed bound: `row-wise sum(abs(full flux Jacobian)); includes all off-diagonal Hodge/convolution coupling`.

| Baseline | Full expression | New constants | E_rho | E_v | E_data | Fitness | Incumbent fitness | Delta | Finite/feasible | Runtime (s) | RSS (MB) |
|---|---|---|---:|---:|---:|---:|---:|---:|:---:|---:|---:|
| Greenshields | `(exp(0.0231229711605-0.0883173972597*rho-0.15008122102*rho^2)*exp(-293.4581091232274*(conv_3(rho,ones)-3*conv_1(rho,ones))))*exp(a*(conv_3(rho,ones)-3*conv_1(rho,ones))^2)` | a=15665.9025 | 8.301754 | 5.125488 | 6.713621 | 7.103621 | 6.953732 | +0.149889 | yes | 0.672 | 1334.1 |
| IDM | `(exp(-46.991435975062515*(conv_3(rho,ones)-3*conv_1(rho,ones)))*exp(-119537.38834933033*hquad(rho)*(conv_3(rho,ones)-3*conv_1(rho,ones))))*exp(a*(conv_3(rho,ones)-3*conv_1(rho,ones))^2)` | a=-5623.601642 | 5.280914 | 4.175968 | 4.728441 | 5.188441 | 5.048745 | +0.139697 | yes | 0.899 | 3515.4 |
| Weidmann | `(exp(0.1365440521436772*rho*(1-rho/0.80612097)))*exp(a*(conv_3(rho,ones)-3*conv_1(rho,ones))^2)` | a=19999.99902 | 5.940094 | 6.001433 | 5.970764 | 6.210764 | 6.322293 | -0.111529 | no | 0.417 | 3518.4 |
| Triangular | `(1)*exp(a*(conv_3(rho,ones)-3*conv_1(rho,ones))^2)` | a=-19845.25563 | 5.998992 | 6.552504 | 6.275748 | 6.425748 | 6.457489 | -0.031741 | yes | 1.718 | 3838.3 |
| Del Castillo | `(exp(0.0054952026563039776)*exp(-2.1218116018455078*conv_3(rho,ones)-256.86922950791177*(conv_3(rho,ones)-3*conv_1(rho,ones))))*exp(a*(conv_3(rho,ones)-3*conv_1(rho,ones))^2)` | a=19984.00728 | 6.155076 | 4.861579 | 5.508327 | 5.858327 | 5.730221 | +0.128106 | yes | 0.421 | 4392.5 |

All candidates use exact `C.convolution` and are `nonlocal = true`.
`test_evaluated = false`.
