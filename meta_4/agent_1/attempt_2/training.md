# Attempt 2 training

Template: `g_inc*exp(a*(conv_3-4*conv_1))`.

All per-FD meta-3 incumbent coefficients were held fixed. Only `a` was
fitted on full I80 training times 0--63. Two deterministic Powell starts
used at most 60 evaluations each with bounds `[-300.0, 300.0]`.
Convolution is exact `C.convolution`; speed bound is `row-wise sum(abs(full flux Jacobian)); includes off-diagonal coupling`.

| Baseline | Full expression | Nodes | New constant | E_rho | E_v | E_data | Feasible | Evaluations | Seed | Fit time (s) | RSS (MB) |
|---|---|---:|---|---:|---:|---:|:---:|---:|---:|---:|---:|
| Greenshields | `(exp(0.0231229711605-0.0883173972597*rho-0.15008122102*rho^2)*exp(-293.4581091232274*(conv_3(rho,ones)-3*conv_1(rho,ones))))*exp(a*(conv_3(rho,ones)-4*conv_1(rho,ones)))` | 38 | a=-8.189793376 | 7.615558 | 4.951323 | 6.283440 | yes | 90 | 10120 | 37.719 | 1170.4 |
| IDM | `(exp(-46.991435975062515*(conv_3(rho,ones)-3*conv_1(rho,ones))))*exp(a*(conv_3(rho,ones)-4*conv_1(rho,ones)))` | 27 | a=22.62634051 | 6.972140 | 7.242716 | 7.107428 | yes | 110 | 10121 | 307.554 | 3371.7 |
| Weidmann | `(exp(0.136544052144*rho*(1-rho/0.80612097)))*exp(a*(conv_3(rho,ones)-4*conv_1(rho,ones)))` | 23 | a=10.58117363 | 8.377429 | 6.301757 | 7.339593 | yes | 120 | 10122 | 37.378 | 3575.3 |
| Triangular | `(1)*exp(a*(conv_3(rho,ones)-4*conv_1(rho,ones)))` | 14 | a=21.02270101 | 7.407726 | 9.876639 | 8.642183 | yes | 120 | 10123 | 45.136 | 3947.7 |
| Del Castillo | `(exp(0.0054952026563)*exp(-2.1218116018455078*conv_3(rho,ones)-256.86922950791177*(conv_3(rho,ones)-3*conv_1(rho,ones))))*exp(a*(conv_3(rho,ones)-4*conv_1(rho,ones)))` | 34 | a=-0.203587472 | 7.240313 | 5.487924 | 6.364119 | yes | 73 | 10124 | 32.937 | 4243.9 |

Aggregate completed fit runtime: 460.723 seconds.
`test_evaluated = false`.
