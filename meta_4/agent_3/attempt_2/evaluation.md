# Attempt 2 evaluation

Validation uses full I80 times 64--107. Test times 108--179 were not
evaluated. Fitness is `E_data + 0.01*total_tree_nodes`.

| Baseline | Full expression | Parameters | E_rho | E_v | E_data | Fitness | Meta-3 fitness | Change | Finite/feasible | Runtime (s) | RSS (MB) |
|---|---|---|---:|---:|---:|---:|---:|---:|:---:|---:|---:|
| Greenshields | `(exp(0.0231229711605-0.0883173972597*rho-0.15008122102*rho^2)*exp(-293.458109123*(conv_3(rho,ones)-3*conv_1(rho,ones))))*exp(a*SqrtP0(rho)+b*rho)` | a=-0.06577808748, b=0.2182737932 | 10.367278 | 5.036567 | 7.701922 | 8.051922 | 6.953732 | +1.098190 | yes | 0.668 | 1183.8 |
| IDM | `(exp(-46.9914359751*(conv_3(rho,ones)-3*conv_1(rho,ones))))*exp(a*SqrtP0(rho)+b*rho)` | a=-0.3316562184, b=0.101825096 | 5.944986 | 8.182260 | 7.063622 | 7.303622 | 5.417018 | +1.886604 | yes | 0.896 | 3364.9 |
| Weidmann | `(exp(0.136544052144*rho*(1-rho/0.80612097)))*exp(a*SqrtP0(rho)+b*rho)` | a=-0.03443763445, b=0.001243791988 | 5.825033 | 7.016444 | 6.420738 | 6.620738 | 6.322293 | +0.298445 | yes | 0.252 | 3399.1 |
| Triangular | `(1)*exp(a*SqrtP0(rho)+b*rho)` | a=-0.1925413287, b=0.06052141686 | 6.490715 | 9.227747 | 7.859231 | 7.969231 | 6.457489 | +1.511742 | yes | 0.230 | 3505.1 |
| Del Castillo | `(exp(0.0054952026563)*exp(-2.12181160185*conv_3(rho,ones)-256.869229508*(conv_3(rho,ones)-3*conv_1(rho,ones))))*exp(a*SqrtP0(rho)+b*rho)` | a=-0.002138861055, b=0.006610388422 | 6.157804 | 4.891837 | 5.524820 | 5.834820 | 5.730221 | +0.104599 | yes | 0.478 | 3685.9 |

Wave-speed bound: `row-wise sum(abs(full flux Jacobian)); includes off-diagonal coupling`.
`test_evaluated = false`.
