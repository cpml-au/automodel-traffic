# Attempt 3 plan

Hold the current global incumbent fixed and fit the odd nonlinear response

`g = g_inc * exp(a*C^3)`, where
`C = conv_3(rho,ones) - 3*conv_1(rho,ones)`.

Attempt 1's even square improved feasible Triangular fitness modestly, while
attempt 2's signed linear-plus-square response produced a large Triangular gain.
A cubic preserves contrast direction like the useful linear branch but
suppresses small residuals and emphasizes queue fronts. It is also a
one-coefficient nonlinear alternative to the 26-node hybrid. In the available
grammar the cube is `C*SquareP0(C)`, so repeated contrast subtrees give a
24-node increment. Fit `a` in `[-1e6,1e6]`: with maximum measured homogeneous
`|C|^3` near `5.48e-5`, this keeps the exponent within about 54.8. Two
deterministic Powell starts use 60 evaluations per start on full train times
0--63. Full validation times 64--107, the full tree penalty, corrected nonlocal
feasibility, and full Jacobian speed bound determine selection. The explicit
finite zero start is retained if Powell returns an invalid/worse endpoint. Test
times 108--179 remain untouched.
