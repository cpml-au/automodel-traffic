# Attempt 3 training

Template: `g_inc*exp(a*conv_3(rho,ones)+b*(conv_3(rho,ones)-3*conv_1(rho,ones)))`.

Every incumbent coefficient was held fixed. Only the new convolution
coefficient(s) `a, b` were fitted on full
I80 training times 0--63. Two deterministic Powell starts used at most
45 evaluations each with bounds `[-300.0, 300.0]`.

| Baseline | Full expression | Nodes | New constants | E_rho | E_v | E_data | Feasible | Evaluations | Seed | Fit time (s) | RSS (MB) |
|---|---|---:|---|---:|---:|---:|:---:|---:|---:|---:|---:|
| Greenshields | `(exp(0.0231229711605-0.0883173972597*rho-0.15008122102*rho^2))*exp(a*conv_3(rho,ones)+b*(conv_3(rho,ones)-3*conv_1(rho,ones)))` | 31 | a=1.51443316, b=-294.0992119 | 7.569241 | 5.017954 | 6.293597 | yes | 90 | 9130 | 39.481 | 4420.3 |
| IDM | `(1)*exp(a*conv_3(rho,ones)+b*(conv_3(rho,ones)-3*conv_1(rho,ones)))` | 20 | a=-8.6693169, b=-99.00303342 | 6.907928 | 6.788635 | 6.848281 | yes | 90 | 9131 | 247.221 | 3360.7 |
| Weidmann | `(exp(0.136544052144*rho*(1-rho/0.80612097)))*exp(a*conv_3(rho,ones)+b*(conv_3(rho,ones)-3*conv_1(rho,ones)))` | 29 | a=-3.527123604, b=-30.01934285 | 8.165186 | 5.787605 | 6.976396 | yes | 90 | 9132 | 30.460 | 4455.2 |
| Triangular | `(1)*exp(a*conv_3(rho,ones)+b*(conv_3(rho,ones)-3*conv_1(rho,ones)))` | 20 | a=-4.153758107, b=-79.43070771 | 7.396147 | 10.105503 | 8.750825 | yes | 90 | 9133 | 31.817 | 4455.2 |
| Del Castillo | `(exp(0.0054952026563))*exp(a*conv_3(rho,ones)+b*(conv_3(rho,ones)-3*conv_1(rho,ones)))` | 21 | a=-2.121811602, b=-256.8692295 | 7.237926 | 5.490397 | 6.364161 | yes | 90 | 9134 | 35.014 | 4465.2 |

Aggregate completed fit runtime: 383.992 seconds.
`test_evaluated = false`.
