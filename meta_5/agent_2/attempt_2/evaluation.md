# Attempt 2 evaluation

Full validation times 64--107; fitness includes every tree node. Test was not evaluated.
Nonlocal speed bound: `row-wise sum(abs(full flux Jacobian)); includes off-diagonal convolution coupling`.

| FD | Parameters | E_rho | E_v | E_data | Fitness | Incumbent fitness | Change | Finite/feasible | Runtime (s) | RSS (MB) |
|---|---|---:|---:|---:|---:|---:|---:|:---:|---:|---:|
| Greenshields | a=-390.1454983 | 8.138682 | 5.112597 | 6.625639 | 7.035639 | 6.953732 | +0.081907 | yes | 0.668 | 1422.8 |
| IDM | a=0.0003736235817 | 5.279587 | 4.177901 | 4.728745 | 5.208745 | 5.048745 | +0.160000 | yes | 0.987 | 3608.6 |
| Weidmann | a=14999.99855 | 72.226974 | 376427274829824.000000 | 188213637414912.000000 | 188213637414912.250000 | 6.322293 | +188213637414905.937500 | no | 0.236 | 3613.5 |
| Triangular | a=14999.99855 | 77.235657 | 731237217439055872.000000 | 365618608719527936.000000 | 365618608719527936.000000 | 6.457489 | +365618608719527936.000000 | no | 0.210 | 3616.1 |
| Del Castillo | a=4497.764819 | nan | nan | nan | nan | 5.730221 | +nan | no | 0.531 | 3859.1 |

`test_evaluated = false`.
