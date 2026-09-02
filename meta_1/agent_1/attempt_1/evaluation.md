# Attempt 1 evaluation

All values below use the full I80 prediction validation split. The test
split was not accessed. Fitness is validation data error plus the stated
`0.01 * tree_nodes` complexity penalty.

| Baseline | Expression | Constants | rho error | velocity error | data error | fitness | Finite | Runtime (s) | Peak RSS (MB) |
|---|---|---|---:|---:|---:|---:|:---:|---:|---:|
| Greenshields | `exp(a*rho)` | a=-0.05943973795 | 10.017130 | 7.653740 | 8.835435 | 8.875435 | yes | 0.400 | 825.9 |
| IDM | `exp(a*rho)` | a=-0.3034516108 | 6.905651 | 8.497226 | 7.701438 | 7.741438 | yes | 0.514 | 1266.9 |
| Weidmann | `exp(a*rho)` | a=0.1633405395 | 7.510491 | 6.709445 | 7.109968 | 7.149968 | yes | 0.166 | 1436.6 |
| Triangular | `exp(a*rho)` | a=-0.1937875339 | 6.503874 | 8.996302 | 7.750088 | 7.790088 | yes | 0.150 | 1622.4 |
| Del Castillo | `exp(a*rho)` | a=-0.08049199 | 6.038365 | 7.402493 | 6.720429 | 6.760429 | yes | 0.270 | 1878.6 |

Complexity penalty: 0.04.
