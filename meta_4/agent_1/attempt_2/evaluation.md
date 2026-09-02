# Attempt 2 evaluation

Validation uses full I80 times 64--107. The held-out test interval was
not evaluated. Fitness is `E_data + 0.01*total_tree_nodes`, including the
fixed incumbent. Nonlocal speed uses `row-wise sum(abs(full flux Jacobian)); includes off-diagonal coupling`.

| Baseline | Full expression | New constant | E_rho | E_v | E_data | Fitness | Meta-3 fitness | Delta | Finite/feasible | Runtime (s) | RSS (MB) |
|---|---|---|---:|---:|---:|---:|---:|---:|:---:|---:|---:|
| Greenshields | `(exp(0.0231229711605-0.0883173972597*rho-0.15008122102*rho^2)*exp(-293.4581091232274*(conv_3(rho,ones)-3*conv_1(rho,ones))))*exp(a*(conv_3(rho,ones)-4*conv_1(rho,ones)))` | a=-8.189793376 | 9.636522 | 4.879273 | 7.257897 | 7.637897 | 6.953732 | +0.684165 | yes | 0.602 | 1185.3 |
| IDM | `(exp(-46.991435975062515*(conv_3(rho,ones)-3*conv_1(rho,ones))))*exp(a*(conv_3(rho,ones)-4*conv_1(rho,ones)))` | a=22.62634051 | 6.464761 | 7.869470 | 7.167115 | 7.437115 | 5.417018 | +2.020097 | yes | 0.919 | 3373.6 |
| Weidmann | `(exp(0.136544052144*rho*(1-rho/0.80612097)))*exp(a*(conv_3(rho,ones)-4*conv_1(rho,ones)))` | a=10.58117363 | 6.189585 | 8.961783 | 7.575684 | 7.805684 | 6.322293 | +1.483391 | yes | 0.370 | 3582.1 |
| Triangular | `(1)*exp(a*(conv_3(rho,ones)-4*conv_1(rho,ones)))` | a=21.02270101 | 6.678946 | 9.687588 | 8.183268 | 8.323268 | 6.457489 | +1.865779 | yes | 0.342 | 3955.6 |
| Del Castillo | `(exp(0.0054952026563)*exp(-2.1218116018455078*conv_3(rho,ones)-256.86922950791177*(conv_3(rho,ones)-3*conv_1(rho,ones))))*exp(a*(conv_3(rho,ones)-4*conv_1(rho,ones)))` | a=-0.203587472 | 6.140401 | 4.899692 | 5.520046 | 5.860046 | 5.730221 | +0.129825 | yes | 0.458 | 4251.8 |

All candidates are `nonlocal = true` and use exact `C.convolution`.
`test_evaluated = false`.
