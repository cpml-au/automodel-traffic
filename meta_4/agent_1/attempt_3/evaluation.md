# Attempt 3 evaluation

Validation uses full I80 times 64--107. The held-out test interval was
not evaluated. Fitness is `E_data + 0.01*total_tree_nodes`, including the
fixed incumbent. Nonlocal speed uses `row-wise sum(abs(full flux Jacobian)); includes off-diagonal coupling`.

| Baseline | Full expression | New constant | E_rho | E_v | E_data | Fitness | Meta-3 fitness | Delta | Finite/feasible | Runtime (s) | RSS (MB) |
|---|---|---|---:|---:|---:|---:|---:|---:|:---:|---:|---:|
| Greenshields | `(exp(0.0231229711605-0.0883173972597*rho-0.15008122102*rho^2)*exp(-293.4581091232274*(conv_3(rho,ones)-3*conv_1(rho,ones))))*exp(a*(conv_3(rho,ones)-3*conv_1(rho,ones)))` | a=0.2169680955 | 8.258936 | 5.148437 | 6.703686 | 7.083686 | 6.953732 | +0.129954 | yes | 0.873 | 1059.4 |
| IDM | `(exp(-46.991435975062515*(conv_3(rho,ones)-3*conv_1(rho,ones))))*exp(a*(conv_3(rho,ones)-3*conv_1(rho,ones)))` | a=-0.1546941142 | 5.523607 | 5.027435 | 5.275521 | 5.545521 | 5.417018 | +0.128503 | yes | 0.842 | 3347.3 |
| Weidmann | `(exp(0.136544052144*rho*(1-rho/0.80612097)))*exp(a*(conv_3(rho,ones)-3*conv_1(rho,ones)))` | a=299.999305 | 40.180069 | 6233093.500000 | 3116566.750000 | 3116566.980000 | 6.322293 | +3116560.657707 | no | 0.201 | 3348.4 |
| Triangular | `(1)*exp(a*(conv_3(rho,ones)-3*conv_1(rho,ones)))` | a=50.95367165 | 8.443944 | 6.879675 | 7.661809 | 7.801809 | 6.457489 | +1.344320 | yes | 0.339 | 3746.7 |
| Del Castillo | `(exp(0.0054952026563)*exp(-2.1218116018455078*conv_3(rho,ones)-256.86922950791177*(conv_3(rho,ones)-3*conv_1(rho,ones))))*exp(a*(conv_3(rho,ones)-3*conv_1(rho,ones)))` | a=0.1336327269 | 6.114310 | 4.925790 | 5.520050 | 5.860050 | 5.730221 | +0.129829 | yes | 0.415 | 4139.0 |

All candidates are `nonlocal = true` and use exact `C.convolution`.
`test_evaluated = false`.
