# Attempt 2 evaluation

All values use the full I80 prediction validation split (times 64-107).
The held-out test split was not evaluated. Fitness is validation
`E_data + 0.01 * tree_nodes`; lower is better.

| Baseline | Expression | Constants | E_rho | E_v | E_data | Fitness | Finite/feasible | Runtime (s) | Peak RSS (MB) |
|---|---|---|---:|---:|---:|---:|:---:|---:|---:|
| Greenshields | `exp(c0 + a*rho)` | c0=0.02312297116, a=-0.08831739726 | 9.347294 | 7.450416 | 8.398855 | 8.458855 | yes | 0.145 | 2282.8 |
| IDM | `exp(c0 + a*rho)` | c0=-0.1678685824, a=-0.09076265524 | 6.133255 | 7.747911 | 6.940583 | 7.000583 | yes | 0.557 | 2821.7 |
| Weidmann | `exp(c0 + a*rho)` | c0=-0.07011210945, a=0.2297073667 | 8.870292 | 7.269213 | 8.069752 | 8.129752 | yes | 0.178 | 3096.2 |
| Triangular | `exp(c0 + a*rho)` | c0=-0.1299587479, a=0.1091785095 | 7.801054 | 7.845758 | 7.823406 | 7.883406 | yes | 0.152 | 3347.2 |
| Del Castillo | `exp(c0 + a*rho)` | c0=0.005495202656, a=-0.09005362272 | 6.036612 | 7.543319 | 6.789966 | 6.849966 | yes | 0.279 | 3716.6 |

Complexity penalty: 0.06.
`test_evaluated = false`.
