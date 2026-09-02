# Meta-4 agent 2 summary

This lineage augmented each fixed meta-3 incumbent with repaired Hodge-star
and convolution features. All comparisons use full validation only; test was
not evaluated.

| FD | Best attempt | Expression | Parameters | Validation E_data | Fitness | Meta-3 fitness | Change |
|---|---:|---|---|---:|---:|---:|---:|
| Greenshields | 2 | `(exp(0.0231229711605-0.0883173972597*rho-0.15008122102*rho^2)*exp(-293.458109123*(conv_3-3*conv_1)))*exp(a*hquad*(conv_3-3*conv_1))` | a=-30813.75332 | 6.625660 | 7.055660 | 6.953732 | +0.101927 |
| IDM | 2 | `(exp(-46.9914359751*(conv_3-3*conv_1)))*exp(a*hquad*(conv_3-3*conv_1))` | a=-119537.3883 | 4.728745 | 5.048745 | 5.417018 | -0.368274 |
| Weidmann | 3 | `(exp(0.136544052144*rho*(1-rho/0.80612097)))*exp(a*(conv_3-3*conv_1)+b*hquad)` | a=-299.9991404, b=-1757.434601 | 45.303795 | 45.603795 | 6.322293 | +39.281502 |
| Triangular | 3 | `(1)*exp(a*(conv_3-3*conv_1)+b*hquad)` | a=50.98755103, b=-3.328582147 | 7.335152 | 7.545152 | 6.457489 | +1.087663 |
| Del Castillo | 2 | `(exp(0.0054952026563)*exp(-2.12181160185*conv_3-256.869229508*(conv_3-3*conv_1)))*exp(a*hquad*(conv_3-3*conv_1))` | a=-54288.2137 | 5.525867 | 5.915867 | 5.730221 | +0.185645 |

Lineage selection: retain the meta-3 incumbent for Greenshields, Weidmann,
Triangular, and Del Castillo. Adopt attempt 2 for IDM, whose 32-node fitness is
5.048745 versus 5.417018 at meta 3 (change -0.368274). Its Powell fit exhausted
the 120-evaluation total budget, so the root review should independently
re-evaluate it. Attempts 1 and 2 returned infeasible/nonfinite Weidmann endpoints
and are rejected; their diagnostics remain in the result files.

Registered live Hodge names: `St_oneP0`, `St_oneP1`, `St_oneD0`, `St_oneD1`.
Nonlocal speed bound: `row-wise sum(abs(full flux Jacobian)); includes all off-diagonal Hodge/convolution coupling`.
`test_evaluated = false`.
