# Attempt 2 training

Template: `g_inc*exp(a*C+b*C^2), C=conv_3(rho,ones)-3*conv_1(rho,ones)`.

All coefficients of each global incumbent were fixed. Only the new
coefficient(s) `a, b` were fitted on full train
times 0--63 with two deterministic Powell starts, at most 60
evaluations per start, and bounds `[[-300.0, 300.0], [-20000.0, 20000.0]]`.
Nonlocal speed bound: `row-wise sum(abs(full flux Jacobian)); includes all off-diagonal Hodge/convolution coupling`.

| Baseline | Full expression | Nodes | New constants | E_rho | E_v | E_data | Feasible | Evaluations | Seed | Fit time (s) | RSS (MB) |
|---|---|---:|---|---:|---:|---:|:---:|---:|---:|---:|---:|
| Greenshields | `(exp(0.0231229711605-0.0883173972597*rho-0.15008122102*rho^2)*exp(-293.4581091232274*(conv_3(rho,ones)-3*conv_1(rho,ones))))*exp(a*(conv_3(rho,ones)-3*conv_1(rho,ones))+b*(conv_3(rho,ones)-3*conv_1(rho,ones))^2)` | 51 | a=1.299781964, b=16417.28439 | 7.505183 | 5.122665 | 6.313924 | yes | 120 | 12120 | 50.520 | 1336.9 |
| IDM | `(exp(-46.991435975062515*(conv_3(rho,ones)-3*conv_1(rho,ones)))*exp(-119537.38834933033*hquad(rho)*(conv_3(rho,ones)-3*conv_1(rho,ones))))*exp(a*(conv_3(rho,ones)-3*conv_1(rho,ones))+b*(conv_3(rho,ones)-3*conv_1(rho,ones))^2)` | 58 | a=82.42163419, b=-12687.60394 | 6.461137 | 8.483037 | 7.472087 | yes | 120 | 12121 | 352.413 | 3518.1 |
| Weidmann | `(exp(0.1365440521436772*rho*(1-rho/0.80612097)))*exp(a*(conv_3(rho,ones)-3*conv_1(rho,ones))+b*(conv_3(rho,ones)-3*conv_1(rho,ones))^2)` | 36 | a=299.999305, b=0 | 34.172211 | 6453532.000000 | 3226783.000000 | no | 120 | 12122 | 13.131 | 3524.1 |
| Triangular | `(1)*exp(a*(conv_3(rho,ones)-3*conv_1(rho,ones))+b*(conv_3(rho,ones)-3*conv_1(rho,ones))^2)` | 27 | a=-207.7942781, b=14690.28913 | 7.041351 | 9.262589 | 8.151970 | yes | 120 | 12123 | 50.550 | 3842.1 |
| Del Castillo | `(exp(0.0054952026563039776)*exp(-2.1218116018455078*conv_3(rho,ones)-256.86922950791177*(conv_3(rho,ones)-3*conv_1(rho,ones))))*exp(a*(conv_3(rho,ones)-3*conv_1(rho,ones))+b*(conv_3(rho,ones)-3*conv_1(rho,ones))^2)` | 47 | a=-0.2062989687, b=19997.85724 | 7.222209 | 5.442559 | 6.332384 | yes | 120 | 12124 | 59.013 | 4397.5 |

Aggregate completed fit runtime: 525.627 seconds.
`test_evaluated = false`.
