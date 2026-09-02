# Attempt 3 evaluation

Full validation times 64--107; fitness is `E_data + 0.01*total_nodes`.
Nonlocal speed bound: `row-wise sum(abs(full flux Jacobian)); includes all off-diagonal Hodge/convolution coupling`.
Registered Hodge names: `St_oneP0, St_oneP1, St_oneD0, St_oneD1`.

| FD | Parameters | E_rho | E_v | E_data | Fitness | Meta-3 fitness | Change | Finite/feasible | Runtime (s) | RSS (MB) |
|---|---|---:|---:|---:|---:|---:|---:|:---:|---:|---:|
| Greenshields | a=-6.362556798, b=26.72142449 | 10.712700 | 5.068594 | 7.890647 | 8.340647 | 6.953732 | +1.386914 | yes | 0.625 | 1263.7 |
| IDM | a=0.09895592915, b=-45.27079703 | 6.130311 | 6.950236 | 6.540274 | 6.880274 | 5.417018 | +1.463256 | yes | 0.767 | 3487.3 |
| Weidmann | a=-299.9991404, b=-1757.434601 | 5.911703 | 84.695885 | 45.303795 | 45.603795 | 6.322293 | +39.281502 | yes | 0.320 | 3517.7 |
| Triangular | a=50.98755103, b=-3.328582147 | 7.997720 | 6.672584 | 7.335152 | 7.545152 | 6.457489 | +1.087663 | yes | 0.291 | 3816.9 |
| Del Castillo | a=-0.2062989687, b=3.00290988 | 6.325887 | 4.784229 | 5.555058 | 5.965058 | 5.730221 | +0.234837 | yes | 0.490 | 4110.9 |

All candidates use `is_nonlocal_feasible`.
`test_evaluated = false`.
