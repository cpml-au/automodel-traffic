# Attempt 1 evaluation

Full validation times 64--107; fitness is `E_data + 0.01*total_nodes`.
Nonlocal speed bound: `row-wise sum(abs(full flux Jacobian)); includes all off-diagonal Hodge/convolution coupling`.
Registered Hodge names: `St_oneP0, St_oneP1, St_oneD0, St_oneD1`.

| FD | Parameters | E_rho | E_v | E_data | Fitness | Meta-3 fitness | Change | Finite/feasible | Runtime (s) | RSS (MB) |
|---|---|---:|---:|---:|---:|---:|---:|:---:|---:|---:|
| Greenshields | a=-102.8027683 | 8.231526 | 5.103027 | 6.667277 | 7.087277 | 6.953732 | +0.133544 | yes | 0.703 | 1398.2 |
| IDM | a=-443.270414 | 5.525958 | 4.407512 | 4.966735 | 5.276735 | 5.417018 | -0.140283 | yes | 0.935 | 3572.4 |
| Weidmann | a=999.9989421 | 40.237061 | 813272730659985140859161005281771520.000000 | 406636365329992570429580502640885760.000000 | 406636365329992570429580502640885760.000000 | 6.322293 | +406636365329992570429580502640885760.000000 | no | 0.232 | 3578.1 |
| Triangular | a=190.7181361 | 8.870175 | 7.045199 | 7.957687 | 8.137687 | 6.457489 | +1.680198 | yes | 0.363 | 3723.2 |
| Del Castillo | a=-181.0737941 | 6.384772 | 4.740941 | 5.562857 | 5.942857 | 5.730221 | +0.212635 | yes | 0.505 | 4319.4 |

All candidates use `is_nonlocal_feasible`.
`test_evaluated = false`.
