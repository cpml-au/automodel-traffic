# Attempt 3 evaluation

Full validation times 64--107 only. Fitness is
`E_data + 0.01*full_tree_nodes`; lower is better.
Wave-speed bound: `row-wise sum(abs(full flux Jacobian)); includes all off-diagonal Hodge/convolution coupling`.

| FD | Parameters | E_rho | E_v | E_data | Fitness | Incumbent fitness | Change | Finite/feasible | Runtime (s) | RSS (MB) |
|---|---|---:|---:|---:|---:|---:|---:|:---:|---:|---:|
| Greenshields | a=0 | 8.259530 | 5.147935 | 6.703732 | 7.183732 | 6.953732 | +0.230000 | yes | 0.687 | 666.4 |
| IDM | a=0 | 5.279587 | 4.177902 | 4.728745 | 5.278745 | 5.048745 | +0.230000 | yes | 0.732 | 1479.4 |
| Weidmann | a=0 | 5.923585 | 6.521000 | 6.222293 | 6.552293 | 6.322293 | +0.230000 | no | 0.298 | 1538.1 |
| Triangular | a=0 | 6.485486 | 6.409492 | 6.447489 | 6.687489 | 6.457489 | +0.230000 | yes | 0.279 | 1572.9 |
| Del Castillo | a=0 | 6.115417 | 4.925029 | 5.520223 | 5.960223 | 5.730221 | +0.230001 | yes | 0.471 | 1614.0 |

Registered repaired Hodge variants: `St_oneP0, St_oneP1, St_oneD0, St_oneD1`.
All candidates use `is_nonlocal_feasible`.
`test_evaluated = false`.
