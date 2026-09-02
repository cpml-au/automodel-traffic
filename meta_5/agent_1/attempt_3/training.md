# Attempt 3 training

Template: `g_inc*exp(a*C^3), C=conv_3(rho,ones)-3*conv_1(rho,ones)`.

All coefficients of each global incumbent were fixed. Only the new
coefficient(s) `a` were fitted on full train
times 0--63 with two deterministic Powell starts, at most 60
evaluations per start, and bounds `[[-1000000.0, 1000000.0]]`.
Nonlocal speed bound: `row-wise sum(abs(full flux Jacobian)); includes all off-diagonal Hodge/convolution coupling`.

| Baseline | Full expression | Nodes | New constants | E_rho | E_v | E_data | Feasible | Evaluations | Seed | Fit time (s) | RSS (MB) |
|---|---|---:|---|---:|---:|---:|:---:|---:|---:|---:|---:|
| Greenshields | `(exp(0.0231229711605-0.0883173972597*rho-0.15008122102*rho^2)*exp(-293.4581091232274*(conv_3(rho,ones)-3*conv_1(rho,ones))))*exp(a*(conv_3(rho,ones)-3*conv_1(rho,ones))^3)` | 49 | a=999489.5344 | 7.518608 | 5.146430 | 6.332519 | yes | 82 | 12130 | 36.634 | 1142.1 |
| IDM | `(exp(-46.991435975062515*(conv_3(rho,ones)-3*conv_1(rho,ones)))*exp(-119537.38834933033*hquad(rho)*(conv_3(rho,ones)-3*conv_1(rho,ones))))*exp(a*(conv_3(rho,ones)-3*conv_1(rho,ones))^3)` | 56 | a=0 | 6.585182 | 8.773269 | 7.679225 | yes | 69 | 12131 | 195.794 | 3483.6 |
| Weidmann | `(exp(0.1365440521436772*rho*(1-rho/0.80612097)))*exp(a*(conv_3(rho,ones)-3*conv_1(rho,ones))^3)` | 34 | a=999999.9785 | 8.178328 | 6.425850 | 7.302089 | no | 77 | 12132 | 9.173 | 3485.4 |
| Triangular | `(1)*exp(a*(conv_3(rho,ones)-3*conv_1(rho,ones))^3)` | 25 | a=999200.3799 | 7.708842 | 10.495633 | 9.102238 | yes | 120 | 12133 | 49.503 | 3830.1 |
| Del Castillo | `(exp(0.0054952026563039776)*exp(-2.1218116018455078*conv_3(rho,ones)-256.86922950791177*(conv_3(rho,ones)-3*conv_1(rho,ones))))*exp(a*(conv_3(rho,ones)-3*conv_1(rho,ones))^3)` | 45 | a=999775.631 | 7.238368 | 5.486335 | 6.362351 | yes | 74 | 12134 | 36.043 | 4174.3 |

Aggregate completed fit runtime: 327.147 seconds.
`test_evaluated = false`.
