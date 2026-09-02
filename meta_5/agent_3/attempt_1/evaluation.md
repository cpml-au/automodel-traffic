# Attempt 1 evaluation

Full validation times 64--107 only. Fitness is
`E_data + 0.01*full_tree_nodes`; lower is better.
Wave-speed bound: `row-wise sum(abs(full flux Jacobian)); includes all off-diagonal Hodge/convolution coupling`.

| FD | Parameters | E_rho | E_v | E_data | Fitness | Incumbent fitness | Change | Finite/feasible | Runtime (s) | RSS (MB) |
|---|---|---:|---:|---:|---:|---:|---:|:---:|---:|---:|
| Greenshields | a=48.62308613, b=-42852.50609 | 7.958491 | 5.195994 | 6.577243 | 7.127243 | 6.953732 | +0.173510 | yes | 0.691 | 1439.5 |
| IDM | a=86.08646158, b=-29600.21354 | 5.041477 | 4.227371 | 4.634424 | 5.254424 | 5.048745 | +0.205680 | yes | 0.999 | 3633.9 |
| Weidmann | a=0, b=0 | 5.923585 | 6.521000 | 6.222293 | 6.622293 | 6.322293 | +0.300000 | no | 0.682 | 623.4 |
| Triangular | a=50.99860897, b=0 | 8.445928 | 6.881437 | 7.663682 | 7.973682 | 6.457489 | +1.516193 | yes | 0.866 | 914.1 |
| Del Castillo | a=71.14073361, b=-54261.81513 | 5.713920 | 5.167340 | 5.440630 | 5.950630 | 5.730221 | +0.220409 | yes | 0.536 | 3852.5 |

Registered repaired Hodge variants: `St_oneP0, St_oneP1, St_oneD0, St_oneD1`.
All candidates use `is_nonlocal_feasible`.
`test_evaluated = false`.
