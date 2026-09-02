# Attempt 3 evaluation

Full validation times 64--107 were used. Test times 108--179 were not
evaluated. Fitness is `E_data + 0.01*total_tree_nodes`, including the
fixed incumbent. Nonlocal speed bound: `row-wise sum(abs(full flux Jacobian)); includes all off-diagonal Hodge/convolution coupling`.

| Baseline | Full expression | New constants | E_rho | E_v | E_data | Fitness | Incumbent fitness | Delta | Finite/feasible | Runtime (s) | RSS (MB) |
|---|---|---|---:|---:|---:|---:|---:|---:|:---:|---:|---:|
| Greenshields | `(exp(0.0231229711605-0.0883173972597*rho-0.15008122102*rho^2)*exp(-293.4581091232274*(conv_3(rho,ones)-3*conv_1(rho,ones))))*exp(a*(conv_3(rho,ones)-3*conv_1(rho,ones))^3)` | a=999489.5344 | 8.270281 | 5.149343 | 6.709812 | 7.199812 | 6.953732 | +0.246079 | yes | 0.670 | 1156.3 |
| IDM | `(exp(-46.991435975062515*(conv_3(rho,ones)-3*conv_1(rho,ones)))*exp(-119537.38834933033*hquad(rho)*(conv_3(rho,ones)-3*conv_1(rho,ones))))*exp(a*(conv_3(rho,ones)-3*conv_1(rho,ones))^3)` | a=0 | 5.279587 | 4.177902 | 4.728745 | 5.288745 | 5.048745 | +0.240000 | yes | 0.779 | 3485.4 |
| Weidmann | `(exp(0.1365440521436772*rho*(1-rho/0.80612097)))*exp(a*(conv_3(rho,ones)-3*conv_1(rho,ones))^3)` | a=999999.9785 | 5.959136 | 6.524138 | 6.241637 | 6.581637 | 6.322293 | +0.259344 | no | 0.413 | 3487.7 |
| Triangular | `(1)*exp(a*(conv_3(rho,ones)-3*conv_1(rho,ones))^3)` | a=999200.3799 | 6.533462 | 6.420808 | 6.477135 | 6.727135 | 6.457489 | +0.269646 | yes | 0.337 | 3839.4 |
| Del Castillo | `(exp(0.0054952026563039776)*exp(-2.1218116018455078*conv_3(rho,ones)-256.86922950791177*(conv_3(rho,ones)-3*conv_1(rho,ones))))*exp(a*(conv_3(rho,ones)-3*conv_1(rho,ones))^3)` | a=999775.631 | 6.128078 | 4.924322 | 5.526200 | 5.976200 | 5.730221 | +0.245979 | yes | 0.426 | 4184.8 |

All candidates use exact `C.convolution` and are `nonlocal = true`.
`test_evaluated = false`.
