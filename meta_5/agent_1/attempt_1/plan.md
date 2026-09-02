# Attempt 1 plan

Start from each current global incumbent in `automodel/final_candidates.json`,
with every incumbent coefficient fixed, and fit the even nonlinear contrast
factor

`g = g_inc * exp(a*C^2)`, where
`C = conv_3(rho,ones) - 3*conv_1(rho,ones)`.

Squaring tests whether contrast magnitude matters independently of queue-front
sign. Only `a` is fitted on full train times 0--63 using two deterministic
Powell starts and 60 evaluations per start. A preflight `[-1e6,1e6]` range
overflowed `exp(a*C^2)` in every full simulation; using the observed maximum
homogeneous `C^2 = 0.001442`, the accepted run narrows this to `[-2e4,2e4]` so
the exponent stays numerically resolvable. Full validation times 64--107
determine selection.
Fitness counts the 14-node increment plus the complete fixed incumbent. The
nonlocal homogeneous feasibility gate and full row-absolute-Jacobian speed
bound apply. Test times 108--179 remain untouched.
