# Attempt 3 evaluation

Validation uses full I80 times 64--107. Test times 108--179 were not
evaluated. Fitness is `E_data + 0.01*total_tree_nodes`.

| Baseline | Full expression | Parameters | E_rho | E_v | E_data | Fitness | Meta-3 fitness | Change | Finite/feasible | Runtime (s) | RSS (MB) |
|---|---|---|---:|---:|---:|---:|---:|---:|:---:|---:|---:|
| Greenshields | `(exp(0.0231229711605-0.0883173972597*rho-0.15008122102*rho^2)*exp(-293.458109123*(conv_3(rho,ones)-3*conv_1(rho,ones))))*exp(a*SqrtP0(rho)+b*rho^2)` | a=0.05041564411, b=0.08354968132 | 9.533124 | 4.860028 | 7.196576 | 7.566576 | 6.953732 | +0.612844 | yes | 0.715 | 1031.4 |
| IDM | `(exp(-46.9914359751*(conv_3(rho,ones)-3*conv_1(rho,ones))))*exp(a*SqrtP0(rho)+b*rho^2)` | a=-0.3554653767, b=0.3235312489 | 5.752972 | 7.752353 | 6.752663 | 7.012663 | 5.417018 | +1.595645 | yes | 0.947 | 3312.9 |
| Weidmann | `(exp(0.136544052144*rho*(1-rho/0.80612097)))*exp(a*SqrtP0(rho)+b*rho^2)` | a=-0.3770427067, b=1.507134031 | 14.499782 | 8.329763 | 11.414772 | 11.634772 | 6.322293 | +5.312479 | yes | 0.301 | 3498.1 |
| Triangular | `(1)*exp(a*SqrtP0(rho)+b*rho^2)` | a=-0.2865864788, b=0.6055425796 | 7.111139 | 8.056026 | 7.583583 | 7.713583 | 6.457489 | +1.256094 | yes | 0.198 | 3699.1 |
| Del Castillo | `(exp(0.0054952026563)*exp(-2.12181160185*conv_3(rho,ones)-256.869229508*(conv_3(rho,ones)-3*conv_1(rho,ones))))*exp(a*SqrtP0(rho)+b*rho^2)` | a=-0.002138861055, b=0.04694894837 | 6.356644 | 4.780971 | 5.568808 | 5.898808 | 5.730221 | +0.168586 | yes | 0.488 | 4017.9 |

Wave-speed bound: `row-wise sum(abs(full flux Jacobian)); includes off-diagonal coupling`.
`test_evaluated = false`.
