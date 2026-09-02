# Attempt 2 evaluation

Full validation times 64--107 only. Fitness is
`E_data + 0.01*full_tree_nodes`; lower is better.
Wave-speed bound: `row-wise sum(abs(full flux Jacobian)); includes all off-diagonal Hodge/convolution coupling`.

| FD | Parameters | E_rho | E_v | E_data | Fitness | Incumbent fitness | Change | Finite/feasible | Runtime (s) | RSS (MB) |
|---|---|---:|---:|---:|---:|---:|---:|:---:|---:|---:|
| Greenshields | a=-102.8410728, b=-200.000003 | 8.230750 | 5.102823 | 6.666787 | 7.256787 | 6.953732 | +0.303054 | yes | 0.760 | 1465.3 |
| IDM | a=246.9610852, b=-44303.4881 | 5.125805 | 4.122959 | 4.624382 | 5.284382 | 5.048745 | +0.235638 | yes | 0.829 | 3672.6 |
| Weidmann | a=0, b=0 | 5.923585 | 6.521000 | 6.222293 | 6.662293 | 6.322293 | +0.340000 | no | 0.375 | 3672.6 |
| Triangular | a=1598.514331, b=-382259.7217 | 6.805105 | 7.112442 | 6.958774 | 7.308774 | 6.457489 | +0.851285 | yes | 0.305 | 3814.7 |
| Del Castillo | a=1999.641094, b=-452328.8146 | nan | nan | nan | nan | 5.730221 | +nan | no | 0.396 | 4364.4 |

Registered repaired Hodge variants: `St_oneP0, St_oneP1, St_oneD0, St_oneD1`.
All candidates use `is_nonlocal_feasible`.
`test_evaluated = false`.
