# Attempt 2 evaluation

All values use the full I80 prediction validation split (times 64--107).
The held-out test split was not evaluated. Fitness is validation data
error plus `0.01 * tree_nodes`.

| Baseline | Expression | Constants | rho error | velocity error | data error | fitness | Feasible | Validation runtime (s) | Peak RSS (MB) |
|---|---|---|---:|---:|---:|---:|:---:|---:|---:|
| Greenshields | `(1 + a*rho)/(1 + b*rho)` | a=-0.05905643164, b=-0.00012482376 | 10.005164 | 7.653353 | 8.829259 | 8.939259 | yes | 0.449 | 946.5 |
| IDM | `(1 + a*rho)/(1 + b*rho)` | a=1.803268255, b=2.551713544 | 6.910261 | 9.460093 | 8.185177 | 8.295177 | yes | 0.592 | 1622.9 |
| Weidmann | `(1 + a*rho)/(1 + b*rho)` | a=0.159189116, b=-0.00881146449 | 7.467092 | 6.688443 | 7.077767 | 7.187767 | yes | 0.202 | 1881.4 |
| Triangular | `(1 + a*rho)/(1 + b*rho)` | a=0.197528937, b=0.4247429861 | 6.512419 | 9.105995 | 7.809207 | 7.919207 | yes | 0.166 | 2157.7 |
| Del Castillo | `(1 + a*rho)/(1 + b*rho)` | a=-0.07953900713, b=-0.0002736367539 | 6.037746 | 7.411696 | 6.724721 | 6.834721 | yes | 0.273 | 2490.7 |

Complexity penalty: 0.11.

`test_evaluated = false`.
