# Attempt 3 training

Expression: `exp(a*rho + b*rho^2 + c*rho^3)` (16 tree nodes).

All values below use the full I80 prediction training split. Fits used
two Powell starts, 30 evaluations per start, and parameter bounds `[-4, 4]`.

| Baseline | Constants | rho error | velocity error | data error | Feasible | Evaluations | Optimizer status | Fit runtime (s) | Peak RSS (MB) |
|---|---|---:|---:|---:|:---:|---:|---|---:|---:|
| Greenshields | a=0.1462883697, b=-0.8855922779, c=0.939213679 | 7.416229 | 6.713172 | 7.064701 | yes | 60 | False: Maximum number of function evaluations has been exceeded. | 9.687 | 3131.5 |
| IDM | a=-0.303666787, b=0.1732332682, c=0 | 7.019094 | 7.513859 | 7.266477 | yes | 60 | False: Maximum number of function evaluations has been exceeded. | 32.415 | 3474.1 |
| Weidmann | a=0.1633383861, b=0.3654981168, c=0 | 7.768281 | 6.277887 | 7.023084 | yes | 60 | False: Maximum number of function evaluations has been exceeded. | 11.787 | 3659.0 |
| Triangular | a=-0.1937132569, b=0, c=0 | 7.765450 | 10.349955 | 9.057702 | yes | 60 | False: Maximum number of function evaluations has been exceeded. | 10.330 | 3840.2 |
| Del Castillo | a=-0.08001789927, b=-0.06951187409, c=0 | 7.192105 | 6.789798 | 6.990952 | yes | 60 | False: Maximum number of function evaluations has been exceeded. | 15.915 | 4085.2 |

Attempt wall time: 80.137 seconds.
