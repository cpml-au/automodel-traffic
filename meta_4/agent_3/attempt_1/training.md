# Attempt 1 training

Template: `g_inc*exp(a*SqrtP0(rho))`.

All meta-3 incumbent coefficients were fixed. The protected square root
was executed as `jnp.sqrt(jnp.maximum(rho, 0))`, corresponding to GP
`SqrtP0(rho)`. New constants were fitted on full I80 train times 0--63
using two Powell restarts, at most 60 evaluations per
restart, and bounds `[-5, 5]`.

| Baseline | Full expression | Nodes | Parameters | E_rho | E_v | E_data | Feasible | Success | Evaluations | Seed | Fit time (s) | RSS (MB) |
|---|---|---:|---|---:|---:|---:|:---:|:---:|---:|---:|---:|---:|
| Greenshields | `(exp(0.0231229711605-0.0883173972597*rho-0.15008122102*rho^2)*exp(-293.458109123*(conv_3(rho,ones)-3*conv_1(rho,ones))))*exp(a*SqrtP0(rho))` | 31 | a=0.05848173354 | 7.568363 | 5.003437 | 6.285900 | yes | yes | 56 | 12110 | 20.435 | 935.8 |
| IDM | `(exp(-46.9914359751*(conv_3(rho,ones)-3*conv_1(rho,ones))))*exp(a*SqrtP0(rho))` | 20 | a=-0.2911344862 | 6.964084 | 6.071343 | 6.517714 | yes | no | 86 | 12111 | 228.601 | 3307.8 |
| Weidmann | `(exp(0.136544052144*rho*(1-rho/0.80612097)))*exp(a*SqrtP0(rho))` | 16 | a=-0.0338490444 | 8.271723 | 6.150653 | 7.211188 | yes | no | 104 | 12112 | 19.173 | 3469.4 |
| Triangular | `(1)*exp(a*SqrtP0(rho))` | 7 | a=-0.1464189334 | 7.741199 | 10.041154 | 8.891176 | yes | yes | 83 | 12113 | 18.895 | 3687.3 |
| Del Castillo | `(exp(0.0054952026563)*exp(-2.12181160185*conv_3(rho,ones)-256.869229508*(conv_3(rho,ones)-3*conv_1(rho,ones))))*exp(a*SqrtP0(rho))` | 27 | a=-0.002141597792 | 7.236516 | 5.491701 | 6.364108 | yes | yes | 51 | 12114 | 19.886 | 3872.5 |

Aggregate fit runtime: 306.991 seconds.
`test_evaluated = false`.
