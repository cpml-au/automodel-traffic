# Attempt 3 evaluation

All values below use the full I80 prediction validation split. The test
split was not accessed. Fitness is validation data error plus the stated
`0.01 * tree_nodes` complexity penalty.

| Baseline | Expression | Constants | rho error | velocity error | data error | fitness | Finite | Runtime (s) | Peak RSS (MB) |
|---|---|---|---:|---:|---:|---:|:---:|---:|---:|
| Greenshields | `exp(a*rho + b*rho^2 + c*rho^3)` | a=0.1462883697, b=-0.8855922779, c=0.939213679 | 8.631692 | 7.458929 | 8.045311 | 8.205311 | yes | 0.152 | 3136.9 |
| IDM | `exp(a*rho + b*rho^2 + c*rho^3)` | a=-0.303666787, b=0.1732332682, c=0 | 6.788599 | 7.888529 | 7.338564 | 7.498564 | yes | 0.564 | 3483.3 |
| Weidmann | `exp(a*rho + b*rho^2 + c*rho^3)` | a=0.1633383861, b=0.3654981168, c=0 | 11.939810 | 10.048405 | 10.994107 | 11.154107 | yes | 1.716 | 3665.5 |
| Triangular | `exp(a*rho + b*rho^2 + c*rho^3)` | a=-0.1937132569, b=0, c=0 | 6.503795 | 8.995251 | 7.749523 | 7.909523 | yes | 0.174 | 3840.2 |
| Del Castillo | `exp(a*rho + b*rho^2 + c*rho^3)` | a=-0.08001789927, b=-0.06951187409, c=0 | 6.091519 | 8.360629 | 7.226074 | 7.386074 | yes | 0.273 | 4096.1 |

Complexity penalty: 0.16.
