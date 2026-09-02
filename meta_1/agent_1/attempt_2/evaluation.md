# Attempt 2 evaluation

All values below use the full I80 prediction validation split. The test
split was not accessed. Fitness is validation data error plus the stated
`0.01 * tree_nodes` complexity penalty.

| Baseline | Expression | Constants | rho error | velocity error | data error | fitness | Finite | Runtime (s) | Peak RSS (MB) |
|---|---|---|---:|---:|---:|---:|:---:|---:|---:|
| Greenshields | `exp(a*rho + b*rho^2)` | a=-0.05944921716, b=0 | 10.016876 | 7.653763 | 8.835320 | 8.935320 | yes | 0.150 | 2027.2 |
| IDM | `exp(a*rho + b*rho^2)` | a=-0.9796330922, b=1.451517799 | 6.467944 | 8.669990 | 7.568967 | 7.668967 | yes | 0.559 | 2387.6 |
| Weidmann | `exp(a*rho + b*rho^2)` | a=0.1633383861, b=0.3654981168 | 11.939810 | 10.048405 | 10.994107 | 11.094107 | yes | 0.168 | 2556.3 |
| Triangular | `exp(a*rho + b*rho^2)` | a=-0.1937875339, b=0 | 6.503874 | 8.996302 | 7.750088 | 7.850088 | yes | 0.162 | 2736.5 |
| Del Castillo | `exp(a*rho + b*rho^2)` | a=0.1435023183, b=-0.4919838371 | 6.153291 | 8.183041 | 7.168166 | 7.268166 | yes | 0.301 | 2972.3 |

Complexity penalty: 0.10.
