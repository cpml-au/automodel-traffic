# Attempt 3 evaluation

All values use the full I80 prediction validation split (times 64--107).
The held-out test split was not evaluated. Fitness is validation data
error plus `0.01 * tree_nodes`.

| Baseline | Expression | Constants | rho error | velocity error | data error | fitness | Feasible | Validation runtime (s) | Peak RSS (MB) |
|---|---|---|---:|---:|---:|---:|:---:|---:|---:|
| Greenshields | `(1 + a*rho + b*rho^2)/(1 + c*rho)` | a=-0.05905643164, b=-0.04344204673, c=0 | 9.441969 | 7.682182 | 8.562076 | 8.732076 | yes | 0.427 | 960.2 |
| IDM | `(1 + a*rho + b*rho^2)/(1 + c*rho)` | a=0.6804113862, b=3.488739361, c=2.933020238 | 9.340754 | 6.780552 | 8.060653 | 8.230653 | yes | 0.516 | 1642.1 |
| Weidmann | `(1 + a*rho + b*rho^2)/(1 + c*rho)` | a=3.508096828, b=2.270760695, c=3.989828642 | 10.653431 | 8.080384 | 9.366907 | 9.536907 | yes | 0.204 | 1934.6 |
| Triangular | `(1 + a*rho + b*rho^2)/(1 + c*rho)` | a=0.4679734922, b=3.576157952, c=2.132359604 | 13.180256 | 7.910118 | 10.545187 | 10.715187 | yes | 0.160 | 2229.8 |
| Del Castillo | `(1 + a*rho + b*rho^2)/(1 + c*rho)` | a=0.3448258363, b=-0.3734323616, c=0.2615828572 | 6.086178 | 7.907075 | 6.996626 | 7.166626 | yes | 0.242 | 2588.1 |

Complexity penalty: 0.17.

`test_evaluated = false`.
