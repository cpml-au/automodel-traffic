# Attempt 3 evaluation

Validation uses full I80 times 64--107. The held-out test interval was
not evaluated. Fitness is `E_data + 0.01*total_tree_nodes`.
The nonlocal solver speed is bounded by `row-wise sum(abs(full flux Jacobian)); includes off-diagonal coupling`.

| Baseline | Full expression | New constants | E_rho | E_v | E_data | Fitness | Finite/feasible | Runtime (s) | RSS (MB) |
|---|---|---|---:|---:|---:|---:|:---:|---:|---:|
| Greenshields | `(exp(0.0231229711605-0.0883173972597*rho-0.15008122102*rho^2))*exp(a*conv_3(rho,ones)+b*(conv_3(rho,ones)-3*conv_1(rho,ones)))` | a=1.51443316, b=-294.0992119 | 9.023210 | 4.936089 | 6.979650 | 7.289650 | yes | 0.398 | 4433.8 |
| IDM | `(1)*exp(a*conv_3(rho,ones)+b*(conv_3(rho,ones)-3*conv_1(rho,ones)))` | a=-8.6693169, b=-99.00303342 | 5.461517 | 6.337881 | 5.899699 | 6.099699 | yes | 0.745 | 3364.1 |
| Weidmann | `(exp(0.136544052144*rho*(1-rho/0.80612097)))*exp(a*conv_3(rho,ones)+b*(conv_3(rho,ones)-3*conv_1(rho,ones)))` | a=-3.527123604, b=-30.01934285 | 5.425266 | 9.032372 | 7.228819 | 7.518819 | yes | 0.297 | 4455.2 |
| Triangular | `(1)*exp(a*conv_3(rho,ones)+b*(conv_3(rho,ones)-3*conv_1(rho,ones)))` | a=-4.153758107, b=-79.43070771 | 5.811677 | 8.694822 | 7.253249 | 7.453249 | yes | 0.285 | 4455.2 |
| Del Castillo | `(exp(0.0054952026563))*exp(a*conv_3(rho,ones)+b*(conv_3(rho,ones)-3*conv_1(rho,ones)))` | a=-2.121811602, b=-256.8692295 | 6.115411 | 4.925031 | 5.520221 | 5.730221 | yes | 0.310 | 4465.2 |

All candidates are labeled `nonlocal = true`.
`test_evaluated = false`.
