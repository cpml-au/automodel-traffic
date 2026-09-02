# Attempt 1 evaluation

All values use the full I80 prediction validation split (times 64-107).
The held-out test split was not evaluated. Fitness is validation
`E_data + 0.01 * tree_nodes`; lower is better.

| Baseline | Expression | Constants | E_rho | E_v | E_data | Fitness | Finite/feasible | Runtime (s) | Peak RSS (MB) |
|---|---|---|---:|---:|---:|---:|:---:|---:|---:|
| Greenshields | `exp(c0)` | c0=0.02263538121 | 11.638790 | 7.659224 | 9.649007 | 9.669007 | yes | 0.347 | 830.2 |
| IDM | `exp(c0)` | c0=-0.1883932724 | 6.279280 | 6.138362 | 6.208821 | 6.228821 | yes | 0.716 | 1444.2 |
| Weidmann | `exp(c0)` | c0=-0.01971739213 | 5.790349 | 7.430845 | 6.610597 | 6.630597 | yes | 0.150 | 1641.1 |
| Triangular | `exp(c0)` | c0=-0.09737332019 | 6.599504 | 8.417008 | 7.508256 | 7.528256 | yes | 0.155 | 1838.6 |
| Del Castillo | `exp(c0)` | c0=0.005495202656 | 6.802947 | 6.364965 | 6.583956 | 6.603956 | yes | 0.205 | 2053.2 |

Complexity penalty: 0.02.
`test_evaluated = false`.
