# Attempt 3 plan

Freeze the selected meta-3 incumbent separately for each FD, including all of
its fitted coefficients, and append a residual factor on the same contrast:

`g = g_inc * exp(a*(conv_3(rho,ones)-3*conv_1(rho,ones)))`.

For Greenshields, IDM, and Del Castillo this is a residual refinement of a
coefficient already selected in meta 3; for Weidmann and Triangular it is a new
one-coefficient contrast. Only `a` is fitted on the full training interval.
Selection uses full validation and counts the complete unsimplified tree,
including the fixed incumbent. Nonlocal feasibility and the full row-absolute-
Jacobian speed bound are enforced. The held-out test is untouched.
