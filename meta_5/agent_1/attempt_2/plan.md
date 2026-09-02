# Attempt 2 plan

Hold every current global-incumbent coefficient fixed and combine signed and
even contrast responses:

`g = g_inc * exp(a*C + b*C^2)`, where
`C = conv_3(rho,ones) - 3*conv_1(rho,ones)`.

Attempt 1 showed that an even response can improve feasible Triangular
validation fitness, while its apparent Weidmann improvement was non-monotone.
The added linear term can distinguish front direction and may restore a valid
velocity response while retaining useful magnitude curvature. Fit `a` in
`[-300,300]` and `b` in the accepted squared-term range `[-2e4,2e4]`, using two
deterministic Powell starts and 60 evaluations per start on full train times
0--63. Selection uses full validation times 64--107 and the complete 26-node
increment plus incumbent tree. The corrected nonlocal feasibility check and
full row-absolute-Jacobian speed bound apply. Test times 108--179 are untouched.
