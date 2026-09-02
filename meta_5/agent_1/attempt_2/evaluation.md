# Attempt 2 evaluation

Full validation times 64--107 were used. Test times 108--179 were not
evaluated. Fitness is `E_data + 0.01*total_tree_nodes`, including the
fixed incumbent. Nonlocal speed bound: `row-wise sum(abs(full flux Jacobian)); includes all off-diagonal Hodge/convolution coupling`.

| Baseline | Full expression | New constants | E_rho | E_v | E_data | Fitness | Incumbent fitness | Delta | Finite/feasible | Runtime (s) | RSS (MB) |
|---|---|---|---:|---:|---:|---:|---:|---:|:---:|---:|---:|
| Greenshields | `(exp(0.0231229711605-0.0883173972597*rho-0.15008122102*rho^2)*exp(-293.4581091232274*(conv_3(rho,ones)-3*conv_1(rho,ones))))*exp(a*(conv_3(rho,ones)-3*conv_1(rho,ones))+b*(conv_3(rho,ones)-3*conv_1(rho,ones))^2)` | a=1.299781964, b=16417.28439 | 8.301180 | 5.127316 | 6.714248 | 7.224248 | 6.953732 | +0.270515 | yes | 0.649 | 1349.2 |
| IDM | `(exp(-46.991435975062515*(conv_3(rho,ones)-3*conv_1(rho,ones)))*exp(-119537.38834933033*hquad(rho)*(conv_3(rho,ones)-3*conv_1(rho,ones))))*exp(a*(conv_3(rho,ones)-3*conv_1(rho,ones))+b*(conv_3(rho,ones)-3*conv_1(rho,ones))^2)` | a=82.42163419, b=-12687.60394 | 5.073222 | 4.363048 | 4.718135 | 5.298135 | 5.048745 | +0.249390 | yes | 0.972 | 3523.0 |
| Weidmann | `(exp(0.1365440521436772*rho*(1-rho/0.80612097)))*exp(a*(conv_3(rho,ones)-3*conv_1(rho,ones))+b*(conv_3(rho,ones)-3*conv_1(rho,ones))^2)` | a=299.999305, b=0 | 40.180069 | 6233093.500000 | 3116566.750000 | 3116567.110000 | 6.322293 | +3116560.787707 | no | 0.225 | 3524.1 |
| Triangular | `(1)*exp(a*(conv_3(rho,ones)-3*conv_1(rho,ones))+b*(conv_3(rho,ones)-3*conv_1(rho,ones))^2)` | a=-207.7942781, b=14690.28913 | 5.337041 | 4.940441 | 5.138741 | 5.408741 | 6.457489 | -1.048748 | yes | 0.322 | 3849.6 |
| Del Castillo | `(exp(0.0054952026563039776)*exp(-2.1218116018455078*conv_3(rho,ones)-256.86922950791177*(conv_3(rho,ones)-3*conv_1(rho,ones))))*exp(a*(conv_3(rho,ones)-3*conv_1(rho,ones))+b*(conv_3(rho,ones)-3*conv_1(rho,ones))^2)` | a=-0.2062989687, b=19997.85724 | 6.156638 | 4.860575 | 5.508606 | 5.978606 | 5.730221 | +0.248385 | yes | 0.475 | 4406.5 |

All candidates use exact `C.convolution` and are `nonlocal = true`.
`test_evaluated = false`.
