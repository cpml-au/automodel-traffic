# Attempt 3 training

Template: `g_inc*exp(a*SqrtP0(rho)+b*rho^2)`.

All meta-3 incumbent coefficients were fixed. The protected square root
was executed as `jnp.sqrt(jnp.maximum(rho, 0))`, corresponding to GP
`SqrtP0(rho)`. New constants were fitted on full I80 train times 0--63
using two Powell restarts, at most 60 evaluations per
restart, and bounds `[-5, 5]`.

| Baseline | Full expression | Nodes | Parameters | E_rho | E_v | E_data | Feasible | Success | Evaluations | Seed | Fit time (s) | RSS (MB) |
|---|---|---:|---|---:|---:|---:|:---:|:---:|---:|---:|---:|---:|
| Greenshields | `(exp(0.0231229711605-0.0883173972597*rho-0.15008122102*rho^2)*exp(-293.458109123*(conv_3(rho,ones)-3*conv_1(rho,ones))))*exp(a*SqrtP0(rho)+b*rho^2)` | 37 | a=0.05041564411, b=0.08354968132 | 7.606708 | 4.959421 | 6.283064 | yes | yes | 89 | 12130 | 31.921 | 1019.8 |
| IDM | `(exp(-46.9914359751*(conv_3(rho,ones)-3*conv_1(rho,ones))))*exp(a*SqrtP0(rho)+b*rho^2)` | 26 | a=-0.3554653767, b=0.3235312489 | 7.004910 | 5.766846 | 6.385878 | yes | no | 100 | 12131 | 264.723 | 3309.1 |
| Weidmann | `(exp(0.136544052144*rho*(1-rho/0.80612097)))*exp(a*SqrtP0(rho)+b*rho^2)` | 22 | a=-0.3770427067, b=1.507134031 | 7.447149 | 5.397537 | 6.422343 | yes | no | 120 | 12132 | 28.368 | 3491.0 |
| Triangular | `(1)*exp(a*SqrtP0(rho)+b*rho^2)` | 13 | a=-0.2865864788, b=0.6055425796 | 7.838460 | 9.442585 | 8.640522 | yes | no | 101 | 12133 | 18.061 | 3695.4 |
| Del Castillo | `(exp(0.0054952026563)*exp(-2.12181160185*conv_3(rho,ones)-256.869229508*(conv_3(rho,ones)-3*conv_1(rho,ones))))*exp(a*SqrtP0(rho)+b*rho^2)` | 33 | a=-0.002138861055, b=0.04694894837 | 7.263279 | 5.461671 | 6.362475 | yes | yes | 85 | 12134 | 34.510 | 4008.1 |

Aggregate fit runtime: 377.583 seconds.
`test_evaluated = false`.
