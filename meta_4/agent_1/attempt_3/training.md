# Attempt 3 training

Template: `g_inc*exp(a*(conv_3-3*conv_1))`.

All per-FD meta-3 incumbent coefficients were held fixed. Only `a` was
fitted on full I80 training times 0--63. Two deterministic Powell starts
used at most 60 evaluations each with bounds `[-300.0, 300.0]`.
Convolution is exact `C.convolution`; speed bound is `row-wise sum(abs(full flux Jacobian)); includes off-diagonal coupling`.

| Baseline | Full expression | Nodes | New constant | E_rho | E_v | E_data | Feasible | Evaluations | Seed | Fit time (s) | RSS (MB) |
|---|---|---:|---|---:|---:|---:|:---:|---:|---:|---:|---:|
| Greenshields | `(exp(0.0231229711605-0.0883173972597*rho-0.15008122102*rho^2)*exp(-293.4581091232274*(conv_3(rho,ones)-3*conv_1(rho,ones))))*exp(a*(conv_3(rho,ones)-3*conv_1(rho,ones)))` | 38 | a=0.2169680955 | 7.518530 | 5.150417 | 6.334474 | yes | 66 | 10130 | 29.430 | 1042.6 |
| IDM | `(exp(-46.991435975062515*(conv_3(rho,ones)-3*conv_1(rho,ones))))*exp(a*(conv_3(rho,ones)-3*conv_1(rho,ones)))` | 27 | a=-0.1546941142 | 6.994291 | 9.536100 | 8.265196 | yes | 82 | 10131 | 218.099 | 3344.3 |
| Weidmann | `(exp(0.136544052144*rho*(1-rho/0.80612097)))*exp(a*(conv_3(rho,ones)-3*conv_1(rho,ones)))` | 23 | a=299.999305 | 34.172211 | 6453532.000000 | 3226783.000000 | no | 58 | 10132 | 5.870 | 3347.7 |
| Triangular | `(1)*exp(a*(conv_3(rho,ones)-3*conv_1(rho,ones)))` | 14 | a=50.95367165 | 7.336185 | 9.429837 | 8.383011 | yes | 120 | 10133 | 47.218 | 3738.5 |
| Del Castillo | `(exp(0.0054952026563)*exp(-2.1218116018455078*conv_3(rho,ones)-256.86922950791177*(conv_3(rho,ones)-3*conv_1(rho,ones))))*exp(a*(conv_3(rho,ones)-3*conv_1(rho,ones)))` | 34 | a=0.1336327269 | 7.237841 | 5.490483 | 6.364162 | yes | 83 | 10134 | 37.739 | 4128.5 |

Aggregate completed fit runtime: 338.356 seconds.
`test_evaluated = false`.
