# Attempt 2 plan

Freeze the selected meta-3 incumbent separately for each FD, including all of
its fitted coefficients, and append exactly one positive convolution contrast:

`g = g_inc * exp(a*(conv_3(rho,ones)-4*conv_1(rho,ones)))`.

This probes whether stronger local subtraction than the incumbent window-3
contrast better represents queue fronts. Only `a` is fitted on the full
training interval. Selection uses full validation, the complete expression tree
(incumbent included), the nonlocal homogeneous-state feasibility check, and the
full row-absolute-Jacobian speed bound. The held-out test is untouched.
