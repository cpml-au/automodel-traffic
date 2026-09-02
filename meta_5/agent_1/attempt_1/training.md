# Attempt 1 training

Template: `g_inc*exp(a*C^2), C=conv_3(rho,ones)-3*conv_1(rho,ones)`.

All coefficients of each global incumbent were fixed. Only the new
coefficient(s) `a` were fitted on full train
times 0--63 with two deterministic Powell starts, at most 60
evaluations per start, and bounds `[[-20000.0, 20000.0]]`.
Nonlocal speed bound: `row-wise sum(abs(full flux Jacobian)); includes all off-diagonal Hodge/convolution coupling`.

| Baseline | Full expression | Nodes | New constants | E_rho | E_v | E_data | Feasible | Evaluations | Seed | Fit time (s) | RSS (MB) |
|---|---|---:|---|---:|---:|---:|:---:|---:|---:|---:|---:|
| Greenshields | `(exp(0.0231229711605-0.0883173972597*rho-0.15008122102*rho^2)*exp(-293.4581091232274*(conv_3(rho,ones)-3*conv_1(rho,ones))))*exp(a*(conv_3(rho,ones)-3*conv_1(rho,ones))^2)` | 39 | a=15665.9025 | 7.506216 | 5.123817 | 6.315017 | yes | 120 | 12110 | 53.013 | 1325.1 |
| IDM | `(exp(-46.991435975062515*(conv_3(rho,ones)-3*conv_1(rho,ones)))*exp(-119537.38834933033*hquad(rho)*(conv_3(rho,ones)-3*conv_1(rho,ones))))*exp(a*(conv_3(rho,ones)-3*conv_1(rho,ones))^2)` | 46 | a=-5623.601642 | 6.594120 | 8.757157 | 7.675639 | yes | 66 | 12111 | 193.302 | 3515.4 |
| Weidmann | `(exp(0.1365440521436772*rho*(1-rho/0.80612097)))*exp(a*(conv_3(rho,ones)-3*conv_1(rho,ones))^2)` | 24 | a=19999.99902 | 8.016014 | 7.459433 | 7.737723 | no | 73 | 12112 | 8.705 | 3518.4 |
| Triangular | `(1)*exp(a*(conv_3(rho,ones)-3*conv_1(rho,ones))^2)` | 15 | a=-19845.25563 | 7.211318 | 9.905568 | 8.558443 | yes | 120 | 12113 | 51.907 | 3835.3 |
| Del Castillo | `(exp(0.0054952026563039776)*exp(-2.1218116018455078*conv_3(rho,ones)-256.86922950791177*(conv_3(rho,ones)-3*conv_1(rho,ones))))*exp(a*(conv_3(rho,ones)-3*conv_1(rho,ones))^2)` | 35 | a=19984.00728 | 7.222084 | 5.442624 | 6.332354 | yes | 120 | 12114 | 56.517 | 4381.6 |

Aggregate completed fit runtime: 363.444 seconds.
`test_evaluated = false`.
