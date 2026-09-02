# Meta-5 agent 2 summary

Density-gated DEC convolution contrasts were appended to every fixed global meta-4 incumbent.
All fits used full train and full validation only; the held-out test was never evaluated.

| FD | Best attempt | Expression | Parameters | Nodes | Validation E_data | Fitness | Incumbent fitness | Change |
|---|---:|---|---|---:|---:|---:|---:|---:|
| Greenshields | 2 | `(exp(0.0231229711605-0.0883173972597*rho-0.15008122102*rho^2)*exp(-293.458109123*C))*exp(a*rho^2*C)` | a=-390.1454983 | 41 | 6.625639 | 7.035639 | 6.953732 | +0.081907 |
| IDM | 1 | `(exp(-46.9914359751*C)*exp(-119537.388349*hquad*C))*exp(a*rho*C)` | a=124.395271 | 47 | 4.664610 | 5.134610 | 5.048745 | +0.085865 |
| Weidmann | -- | no feasible candidate; retain incumbent | -- | 10 | -- | 6.322293 | 6.322293 | +0.000000 |
| Triangular | 1 | `(1)*exp(a*rho*C)` | a=-921.9701998 | 16 | 5.067508 | 5.227508 | 6.457489 | -1.229981 |
| Del Castillo | 1 | `(exp(0.0054952026563)*exp(-2.12181160185*conv_3-256.869229508*C))*exp(a*rho*C)` | a=994.5579335 | 36 | 6.597411 | 6.957411 | 5.730221 | +1.227190 |

Lineage winners over the incumbent: triangular.
Nonlocal speed bound: `row-wise sum(abs(full flux Jacobian)); includes off-diagonal convolution coupling`.
Selection required homogeneous nonlocal feasibility and finite full simulations.
Ten candidates passed; five pathological Weidmann/Triangular/Del Castillo endpoints were rejected.
`test_evaluated = false`.
