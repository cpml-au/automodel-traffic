# Attempt 3 evaluation

Full validation times 64--107; fitness includes every tree node. Test was not evaluated.
Nonlocal speed bound: `row-wise sum(abs(full flux Jacobian)); includes off-diagonal convolution coupling`.

| FD | Parameters | E_rho | E_v | E_data | Fitness | Incumbent fitness | Change | Finite/feasible | Runtime (s) | RSS (MB) |
|---|---|---:|---:|---:|---:|---:|---:|:---:|---:|---:|
| Greenshields | a=1119.680194, b=0 | 9.644303 | 6.750977 | 8.197640 | 8.747640 | 6.953732 | +1.793908 | yes | 0.748 | 1489.9 |
| IDM | a=124.0956041, b=0 | 5.153337 | 4.175947 | 4.664642 | 5.284642 | 5.048745 | +0.235898 | yes | 1.050 | 3688.4 |
| Weidmann | a=2999.998768, b=0 | 6397.435547 | inf | inf | inf | 6.322293 | +inf | no | 0.269 | 3696.2 |
| Triangular | a=1887.817768, b=-6170.114358 | 6.410837 | 7.950436 | 7.180636 | 7.490636 | 6.457489 | +1.033148 | yes | 0.421 | 3775.4 |
| Del Castillo | a=994.4396655, b=0 | 7.090786 | 6.102780 | 6.596783 | 7.106783 | 5.730221 | +1.376561 | yes | 0.560 | 4087.7 |

`test_evaluated = false`.
