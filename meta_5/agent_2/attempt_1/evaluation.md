# Attempt 1 evaluation

Full validation times 64--107; fitness includes every tree node. Test was not evaluated.
Nonlocal speed bound: `row-wise sum(abs(full flux Jacobian)); includes off-diagonal convolution coupling`.

| FD | Parameters | E_rho | E_v | E_data | Fitness | Incumbent fitness | Change | Finite/feasible | Runtime (s) | RSS (MB) |
|---|---|---:|---:|---:|---:|---:|---:|:---:|---:|---:|
| Greenshields | a=1119.626208 | 9.643871 | 6.750902 | 8.197387 | 8.597387 | 6.953732 | +1.643654 | yes | 0.684 | 1198.2 |
| IDM | a=124.395271 | 5.153134 | 4.176086 | 4.664610 | 5.134610 | 5.048745 | +0.085865 | yes | 0.987 | 3478.1 |
| Weidmann | a=2999.998768 | 6397.433594 | inf | inf | inf | 6.322293 | +inf | no | 0.217 | 3480.0 |
| Triangular | a=-921.9701998 | 5.216815 | 4.918202 | 5.067508 | 5.227508 | 6.457489 | -1.229981 | yes | 0.330 | 3824.7 |
| Del Castillo | a=994.5579335 | 7.091896 | 6.102927 | 6.597411 | 6.957411 | 5.730221 | +1.227190 | yes | 0.487 | 4215.8 |

`test_evaluated = false`.
