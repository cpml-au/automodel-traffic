# Attempt 2 training

Template: `g_inc*exp(a*SqrtP0(rho)+b*rho)`.

All meta-3 incumbent coefficients were fixed. The protected square root
was executed as `jnp.sqrt(jnp.maximum(rho, 0))`, corresponding to GP
`SqrtP0(rho)`. New constants were fitted on full I80 train times 0--63
using two Powell restarts, at most 60 evaluations per
restart, and bounds `[-5, 5]`.

| Baseline | Full expression | Nodes | Parameters | E_rho | E_v | E_data | Feasible | Success | Evaluations | Seed | Fit time (s) | RSS (MB) |
|---|---|---:|---|---:|---:|---:|:---:|:---:|---:|---:|---:|---:|
| Greenshields | `(exp(0.0231229711605-0.0883173972597*rho-0.15008122102*rho^2)*exp(-293.458109123*(conv_3(rho,ones)-3*conv_1(rho,ones))))*exp(a*SqrtP0(rho)+b*rho)` | 35 | a=-0.06577808748, b=0.2182737932 | 7.666165 | 4.897528 | 6.281847 | yes | yes | 100 | 12120 | 42.580 | 1169.3 |
| IDM | `(exp(-46.9914359751*(conv_3(rho,ones)-3*conv_1(rho,ones))))*exp(a*SqrtP0(rho)+b*rho)` | 24 | a=-0.3316562184, b=0.101825096 | 6.943656 | 5.975074 | 6.459365 | yes | no | 100 | 12121 | 265.276 | 3360.4 |
| Weidmann | `(exp(0.136544052144*rho*(1-rho/0.80612097)))*exp(a*SqrtP0(rho)+b*rho)` | 20 | a=-0.03443763445, b=0.001243791988 | 8.270763 | 6.149541 | 7.210152 | yes | no | 99 | 12122 | 14.131 | 3392.7 |
| Triangular | `(1)*exp(a*SqrtP0(rho)+b*rho)` | 11 | a=-0.1925413287, b=0.06052141686 | 7.732718 | 9.963214 | 8.847966 | yes | no | 99 | 12123 | 14.395 | 3501.4 |
| Del Castillo | `(exp(0.0054952026563)*exp(-2.12181160185*conv_3(rho,ones)-256.869229508*(conv_3(rho,ones)-3*conv_1(rho,ones))))*exp(a*SqrtP0(rho)+b*rho)` | 31 | a=-0.002138861055, b=0.006610388422 | 7.242300 | 5.485480 | 6.363890 | yes | yes | 71 | 12124 | 23.762 | 3678.4 |

Aggregate fit runtime: 360.143 seconds.
`test_evaluated = false`.
