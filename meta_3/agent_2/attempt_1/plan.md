# Attempt 1 plan

Test the repaired Hodge-star pair with the positive multiplier
`g = exp(a*St_oneD1(St_oneP0(rho)))`. Its exact typed-GP equivalent is
`ExpP0(MFP0(St_oneD1(St_oneP0(rho)), a))` (6 tree nodes). DCTKit's diagonal
star and inverse star make this a useful registration/control case: on this
one-dimensional mesh the two stars should recover density up to roundoff.

Fit `a` independently for all five fixed calibrated FDs on the complete I80
training split, then score the complete chronological validation split. Use the
shared Powell fitter with two deterministic restarts, 45 evaluations per
restart, bound `a in [-5,5]`, and seed `8110 + baseline_index`. Use the shared
homogeneous-state nonlocal feasibility check because Hodge factors depend on
the mesh metric. Do not evaluate the test split.
