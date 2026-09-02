# Attempt 1 evaluation

Validation uses full I80 times 64--107. The held-out test interval was
not evaluated. Fitness is `E_data + 0.01*total_tree_nodes`, including the
fixed incumbent. Nonlocal speed uses `row-wise sum(abs(full flux Jacobian)); includes off-diagonal coupling`.

| Baseline | Full expression | New constant | E_rho | E_v | E_data | Fitness | Meta-3 fitness | Delta | Finite/feasible | Runtime (s) | RSS (MB) |
|---|---|---|---:|---:|---:|---:|---:|---:|:---:|---:|---:|
| Greenshields | `(exp(0.0231229711605-0.0883173972597*rho-0.15008122102*rho^2)*exp(-293.4581091232274*(conv_3(rho,ones)-3*conv_1(rho,ones))))*exp(a*(conv_3(rho,ones)-2*conv_1(rho,ones)))` | a=8.203115717 | 9.689939 | 4.934185 | 7.312062 | 7.692062 | 6.953732 | +0.738330 | yes | 0.659 | 1172.3 |
| IDM | `(exp(-46.991435975062515*(conv_3(rho,ones)-3*conv_1(rho,ones))))*exp(a*(conv_3(rho,ones)-2*conv_1(rho,ones)))` | a=-32.36973356 | 5.838327 | 7.824235 | 6.831282 | 7.101282 | 5.417018 | +1.684264 | yes | 2.045 | 3383.7 |
| Weidmann | `(exp(0.136544052144*rho*(1-rho/0.80612097)))*exp(a*(conv_3(rho,ones)-2*conv_1(rho,ones)))` | a=-10.5811892 | 5.837512 | 9.342930 | 7.590221 | 7.820221 | 6.322293 | +1.497928 | yes | 0.344 | 3627.8 |
| Triangular | `(1)*exp(a*(conv_3(rho,ones)-2*conv_1(rho,ones)))` | a=-16.06577648 | 6.334051 | 9.170494 | 7.752273 | 7.892273 | 6.457489 | +1.434784 | yes | 0.354 | 3942.1 |
| Del Castillo | `(exp(0.0054952026563)*exp(-2.1218116018455078*conv_3(rho,ones)-256.86922950791177*(conv_3(rho,ones)-3*conv_1(rho,ones))))*exp(a*(conv_3(rho,ones)-2*conv_1(rho,ones)))` | a=0.1878953909 | 6.135357 | 4.903715 | 5.519536 | 5.859536 | 5.730221 | +0.129315 | yes | 0.446 | 4224.5 |

All candidates are `nonlocal = true` and use exact `C.convolution`.
`test_evaluated = false`.
