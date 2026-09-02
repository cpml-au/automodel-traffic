# Attempt 1 training

Template: `g_inc*exp(a*(conv_3-2*conv_1))`.

All per-FD meta-3 incumbent coefficients were held fixed. Only `a` was
fitted on full I80 training times 0--63. Two deterministic Powell starts
used at most 60 evaluations each with bounds `[-300.0, 300.0]`.
Convolution is exact `C.convolution`; speed bound is `row-wise sum(abs(full flux Jacobian)); includes off-diagonal coupling`.

| Baseline | Full expression | Nodes | New constant | E_rho | E_v | E_data | Feasible | Evaluations | Seed | Fit time (s) | RSS (MB) |
|---|---|---:|---|---:|---:|---:|:---:|---:|---:|---:|---:|
| Greenshields | `(exp(0.0231229711605-0.0883173972597*rho-0.15008122102*rho^2)*exp(-293.4581091232274*(conv_3(rho,ones)-3*conv_1(rho,ones))))*exp(a*(conv_3(rho,ones)-2*conv_1(rho,ones)))` | 38 | a=8.203115717 | 7.614853 | 4.951677 | 6.283265 | yes | 90 | 10110 | 38.316 | 1159.0 |
| IDM | `(exp(-46.991435975062515*(conv_3(rho,ones)-3*conv_1(rho,ones))))*exp(a*(conv_3(rho,ones)-2*conv_1(rho,ones)))` | 27 | a=-32.36973356 | 6.988837 | 6.703240 | 6.846038 | yes | 120 | 10111 | 336.809 | 3383.7 |
| Weidmann | `(exp(0.136544052144*rho*(1-rho/0.80612097)))*exp(a*(conv_3(rho,ones)-2*conv_1(rho,ones)))` | 23 | a=-10.5811892 | 8.340456 | 6.120105 | 7.230280 | yes | 120 | 10112 | 40.872 | 3619.9 |
| Triangular | `(1)*exp(a*(conv_3(rho,ones)-2*conv_1(rho,ones)))` | 14 | a=-16.06577648 | 7.675507 | 10.336260 | 9.005883 | yes | 101 | 10113 | 39.536 | 3937.3 |
| Del Castillo | `(exp(0.0054952026563)*exp(-2.1218116018455078*conv_3(rho,ones)-256.86922950791177*(conv_3(rho,ones)-3*conv_1(rho,ones))))*exp(a*(conv_3(rho,ones)-2*conv_1(rho,ones)))` | 34 | a=0.1878953909 | 7.239893 | 5.488348 | 6.364120 | yes | 71 | 10114 | 32.209 | 4216.6 |

Aggregate completed fit runtime: 487.742 seconds.
`test_evaluated = false`.
