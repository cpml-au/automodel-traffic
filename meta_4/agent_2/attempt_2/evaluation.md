# Attempt 2 evaluation

Full validation times 64--107; fitness is `E_data + 0.01*total_nodes`.
Nonlocal speed bound: `row-wise sum(abs(full flux Jacobian)); includes all off-diagonal Hodge/convolution coupling`.
Registered Hodge names: `St_oneP0, St_oneP1, St_oneD0, St_oneD1`.

| FD | Parameters | E_rho | E_v | E_data | Fitness | Meta-3 fitness | Change | Finite/feasible | Runtime (s) | RSS (MB) |
|---|---|---:|---:|---:|---:|---:|---:|:---:|---:|---:|
| Greenshields | a=-30813.75332 | 8.138717 | 5.112603 | 6.625660 | 7.055660 | 6.953732 | +0.101927 | yes | 0.695 | 1440.2 |
| IDM | a=-119537.3883 | 5.279587 | 4.177902 | 4.728745 | 5.048745 | 5.417018 | -0.368274 | yes | 0.998 | 3582.9 |
| Weidmann | a=999999.9785 | 4796891.500000 | inf | inf | inf | 6.322293 | +inf | no | 0.235 | 3585.1 |
| Triangular | a=53932.07703 | 9.137575 | 7.136757 | 8.137166 | 8.327166 | 6.457489 | +1.869677 | yes | 0.331 | 3853.6 |
| Del Castillo | a=-54288.2137 | 6.262347 | 4.789387 | 5.525867 | 5.915867 | 5.730221 | +0.185645 | yes | 0.497 | 4481.8 |

All candidates use `is_nonlocal_feasible`.
`test_evaluated = false`.
