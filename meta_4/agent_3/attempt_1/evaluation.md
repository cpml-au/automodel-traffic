# Attempt 1 evaluation

Validation uses full I80 times 64--107. Test times 108--179 were not
evaluated. Fitness is `E_data + 0.01*total_tree_nodes`.

| Baseline | Full expression | Parameters | E_rho | E_v | E_data | Fitness | Meta-3 fitness | Change | Finite/feasible | Runtime (s) | RSS (MB) |
|---|---|---|---:|---:|---:|---:|---:|---:|:---:|---:|---:|
| Greenshields | `(exp(0.0231229711605-0.0883173972597*rho-0.15008122102*rho^2)*exp(-293.458109123*(conv_3(rho,ones)-3*conv_1(rho,ones))))*exp(a*SqrtP0(rho))` | a=0.05848173354 | 8.997259 | 4.802628 | 6.899943 | 7.209943 | 6.953732 | +0.256211 | yes | 0.598 | 952.9 |
| IDM | `(exp(-46.9914359751*(conv_3(rho,ones)-3*conv_1(rho,ones))))*exp(a*SqrtP0(rho))` | a=-0.2911344862 | 6.079375 | 8.883239 | 7.481307 | 7.681307 | 5.417018 | +2.264289 | yes | 0.894 | 3312.3 |
| Weidmann | `(exp(0.136544052144*rho*(1-rho/0.80612097)))*exp(a*SqrtP0(rho))` | a=-0.0338490444 | 5.823793 | 7.025075 | 6.424434 | 6.584434 | 6.322293 | +0.262141 | yes | 0.237 | 3478.0 |
| Triangular | `(1)*exp(a*SqrtP0(rho))` | a=-0.1464189334 | 6.484667 | 9.158576 | 7.821621 | 7.891621 | 6.457489 | +1.434132 | yes | 0.198 | 3693.6 |
| Del Castillo | `(exp(0.0054952026563)*exp(-2.12181160185*conv_3(rho,ones)-256.869229508*(conv_3(rho,ones)-3*conv_1(rho,ones))))*exp(a*SqrtP0(rho))` | a=-0.002141597792 | 6.098007 | 4.954267 | 5.526136 | 5.796136 | 5.730221 | +0.065915 | yes | 0.464 | 3883.0 |

Wave-speed bound: `row-wise sum(abs(full flux Jacobian)); includes off-diagonal coupling`.
`test_evaluated = false`.
