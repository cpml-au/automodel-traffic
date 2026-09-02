# Attempt 2 evaluation

Validation uses full I80 times 64--107. The held-out test interval was
not evaluated. Fitness is `E_data + 0.01*total_tree_nodes`.
The nonlocal solver speed is bounded by `row-wise sum(abs(full flux Jacobian)); includes off-diagonal coupling`.

| Baseline | Full expression | New constants | E_rho | E_v | E_data | Fitness | Finite/feasible | Runtime (s) | RSS (MB) |
|---|---|---|---:|---:|---:|---:|:---:|---:|---:|
| Greenshields | `(exp(0.0231229711605-0.0883173972597*rho-0.15008122102*rho^2))*exp(a*(conv_3(rho,ones)-3*conv_1(rho,ones)))` | a=-293.4581091 | 8.259530 | 5.147935 | 6.703732 | 6.953732 | yes | 0.371 | 4273.5 |
| IDM | `(1)*exp(a*(conv_3(rho,ones)-3*conv_1(rho,ones)))` | a=-46.99143598 | 5.524559 | 5.029477 | 5.277018 | 5.417018 | yes | 0.713 | 3333.0 |
| Weidmann | `(exp(0.136544052144*rho*(1-rho/0.80612097)))*exp(a*(conv_3(rho,ones)-3*conv_1(rho,ones)))` | a=299.999305 | 40.180069 | 6233093.500000 | 3116566.750000 | 3116566.980000 | no | 0.248 | 4320.3 |
| Triangular | `(1)*exp(a*(conv_3(rho,ones)-3*conv_1(rho,ones)))` | a=50.95367165 | 8.443944 | 6.879675 | 7.661809 | 7.801809 | yes | 0.376 | 4320.3 |
| Del Castillo | `(exp(0.0054952026563))*exp(a*(conv_3(rho,ones)-3*conv_1(rho,ones)))` | a=-259.3539644 | 7.040762 | 4.451153 | 5.745957 | 5.895957 | yes | 0.443 | 4322.3 |

All candidates are labeled `nonlocal = true`.
`test_evaluated = false`.
