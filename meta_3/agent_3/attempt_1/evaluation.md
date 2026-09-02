# Attempt 1 evaluation

Validation uses full I80 times 64--107. The held-out test interval was
not evaluated. Fitness is `E_data + 0.01*total_tree_nodes`.
The nonlocal solver speed is bounded by `row-wise sum(abs(full flux Jacobian)); includes off-diagonal coupling`.

| Baseline | Full expression | New constants | E_rho | E_v | E_data | Fitness | Finite/feasible | Runtime (s) | RSS (MB) |
|---|---|---|---:|---:|---:|---:|:---:|---:|---:|
| Greenshields | `(exp(0.0231229711605-0.0883173972597*rho-0.15008122102*rho^2))*exp(a*conv_3(rho,ones))` | a=1.507525246 | 8.802771 | 7.344305 | 8.073538 | 8.263538 | yes | 0.612 | 1014.2 |
| IDM | `(1)*exp(a*conv_3(rho,ones))` | a=-8.67180587 | 6.774087 | 8.576706 | 7.675396 | 7.755396 | yes | 0.882 | 3171.9 |
| Weidmann | `(exp(0.136544052144*rho*(1-rho/0.80612097)))*exp(a*conv_3(rho,ones))` | a=-3.527086724 | 5.979670 | 9.259740 | 7.619705 | 7.789705 | yes | 0.331 | 3409.5 |
| Triangular | `(1)*exp(a*conv_3(rho,ones))` | a=-5.020002344 | 6.435994 | 8.949199 | 7.692596 | 7.772596 | yes | 0.317 | 3658.8 |
| Del Castillo | `(exp(0.0054952026563))*exp(a*conv_3(rho,ones))` | a=-2.65263026 | 6.025274 | 7.930385 | 6.977829 | 7.067829 | yes | 0.394 | 3939.7 |

All candidates are labeled `nonlocal = true`.
`test_evaluated = false`.
