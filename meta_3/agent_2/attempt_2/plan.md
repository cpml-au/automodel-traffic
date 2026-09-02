# Attempt 2 plan

Move the nonlinearity between the repaired Hodge stars:
`g = exp(a*St_oneD1(SquareD1(St_oneP0(rho))))`. The exact typed-GP equivalent
is `ExpP0(MFP0(St_oneD1(SquareD1(St_oneP0(rho))), a))` (7 tree nodes). Unlike
the double-star control, this feature retains one diagonal mesh Hodge factor
after mapping the squared dual one-cochain back to primal nodes.

Fit `a` independently for all five fixed calibrated FDs on the complete I80
training split, then score the complete chronological validation split. Use two
deterministic Powell restarts, 45 evaluations per restart, bound
`a in [-2500,2500]`, and seed `8120 + baseline_index`. Apply
`is_nonlocal_feasible` because the mesh metric and downstream boundary states
vary. Do not evaluate the test split.
