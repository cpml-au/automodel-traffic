# Attempt 1 training

Template: `g_inc*exp(a*conv_3(rho,ones))`.

Every incumbent coefficient was held fixed. Only the new convolution
coefficient(s) `a` were fitted on full
I80 training times 0--63. Two deterministic Powell starts used at most
45 evaluations each with bounds `[-300.0, 300.0]`.

| Baseline | Full expression | Nodes | New constants | E_rho | E_v | E_data | Feasible | Evaluations | Seed | Fit time (s) | RSS (MB) |
|---|---|---:|---|---:|---:|---:|:---:|---:|---:|---:|---:|
| Greenshields | `(exp(0.0231229711605-0.0883173972597*rho-0.15008122102*rho^2))*exp(a*conv_3(rho,ones))` | 19 | a=1.507525246 | 7.416124 | 6.639184 | 7.027654 | yes | 81 | 9110 | 28.347 | 1005.0 |
| IDM | `(1)*exp(a*conv_3(rho,ones))` | 8 | a=-8.67180587 | 7.124813 | 7.355252 | 7.240033 | yes | 90 | 9111 | 207.552 | 3164.0 |
| Weidmann | `(exp(0.136544052144*rho*(1-rho/0.80612097)))*exp(a*conv_3(rho,ones))` | 17 | a=-3.527086724 | 8.387808 | 6.213935 | 7.300872 | yes | 90 | 9112 | 23.141 | 3402.3 |
| Triangular | `(1)*exp(a*conv_3(rho,ones))` | 8 | a=-5.020002344 | 7.738992 | 10.343769 | 9.041381 | yes | 90 | 9113 | 27.110 | 3649.8 |
| Del Castillo | `(exp(0.0054952026563))*exp(a*conv_3(rho,ones))` | 9 | a=-2.65263026 | 7.192472 | 6.717060 | 6.954766 | yes | 85 | 9114 | 29.352 | 3930.7 |

Aggregate completed fit runtime: 315.503 seconds.
`test_evaluated = false`.
