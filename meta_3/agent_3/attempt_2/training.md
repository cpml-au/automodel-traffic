# Attempt 2 training

Template: `g_inc*exp(a*(conv_3(rho,ones)-3*conv_1(rho,ones)))`.

Every incumbent coefficient was held fixed. Only the new convolution
coefficient(s) `a` were fitted on full
I80 training times 0--63. Two deterministic Powell starts used at most
45 evaluations each with bounds `[-300.0, 300.0]`.

| Baseline | Full expression | Nodes | New constants | E_rho | E_v | E_data | Feasible | Evaluations | Seed | Fit time (s) | RSS (MB) |
|---|---|---:|---|---:|---:|---:|:---:|---:|---:|---:|---:|
| Greenshields | `(exp(0.0231229711605-0.0883173972597*rho-0.15008122102*rho^2))*exp(a*(conv_3(rho,ones)-3*conv_1(rho,ones)))` | 25 | a=-293.4581091 | 7.518588 | 5.150358 | 6.334473 | yes | 90 | 9120 | 37.031 | 4269.0 |
| IDM | `(1)*exp(a*(conv_3(rho,ones)-3*conv_1(rho,ones)))` | 14 | a=-46.99143598 | 6.994205 | 9.536189 | 8.265197 | yes | 90 | 9121 | 235.696 | 3328.1 |
| Weidmann | `(exp(0.136544052144*rho*(1-rho/0.80612097)))*exp(a*(conv_3(rho,ones)-3*conv_1(rho,ones)))` | 23 | a=299.999305 | 34.172211 | 6453532.000000 | 3226783.000000 | no | 58 | 9122 | 6.363 | 4320.3 |
| Triangular | `(1)*exp(a*(conv_3(rho,ones)-3*conv_1(rho,ones)))` | 14 | a=50.95367165 | 7.336185 | 9.429837 | 8.383011 | yes | 90 | 9123 | 40.519 | 4320.3 |
| Del Castillo | `(exp(0.0054952026563))*exp(a*(conv_3(rho,ones)-3*conv_1(rho,ones)))` | 15 | a=-259.3539644 | 7.317589 | 5.478734 | 6.398162 | yes | 90 | 9124 | 43.395 | 4322.3 |

Aggregate completed fit runtime: 363.004 seconds.
`test_evaluated = false`.
