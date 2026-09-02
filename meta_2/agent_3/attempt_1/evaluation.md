# Attempt 1 evaluation

All values use the full I80 prediction validation split (times 64--107).
The held-out test split was not evaluated. Fitness is validation data
error plus `0.01 * tree_nodes`.

| Baseline | Expression | Constants | rho error | velocity error | data error | fitness | Feasible | Validation runtime (s) | Peak RSS (MB) |
|---|---|---|---:|---:|---:|---:|:---:|---:|---:|
| Greenshields | `1 + a*rho` | a=-0.05878239279 | 10.009269 | 7.653008 | 8.831139 | 8.881139 | yes | 0.426 | 903.2 |
| IDM | `1 + a*rho` | a=-0.2733786877 | 6.897743 | 8.376746 | 7.637244 | 7.687244 | yes | 0.549 | 1505.6 |
| Weidmann | `1 + a*rho` | a=0.1637536228 | 7.376661 | 6.638988 | 7.007825 | 7.057825 | yes | 0.148 | 1683.4 |
| Triangular | `1 + a*rho` | a=-0.1803255837 | 6.495402 | 8.901711 | 7.698556 | 7.748556 | yes | 0.157 | 1872.5 |
| Del Castillo | `1 + a*rho` | a=-0.0790741062 | 6.037877 | 7.406733 | 6.722305 | 6.772305 | yes | 0.231 | 2198.3 |

Complexity penalty: 0.05.

`test_evaluated = false`.
