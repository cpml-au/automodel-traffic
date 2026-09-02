# Attempt 1 plan

Freeze the selected meta-3 incumbent separately for each FD, including all of
its fitted coefficients, and append exactly one positive convolution contrast:

`g = g_inc * exp(a*(conv_3(rho,ones)-2*conv_1(rho,ones)))`.

This probes whether reducing the local subtraction relative to the successful
level-cancelling window-3 contrast transfers better. Only `a` is fitted on the
full training interval. Selection uses full validation, the complete expression
tree (incumbent included), the nonlocal homogeneous-state feasibility check,
and the full row-absolute-Jacobian speed bound. The held-out test is untouched.
