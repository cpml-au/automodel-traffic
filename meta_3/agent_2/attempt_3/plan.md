# Attempt 3 plan

Combine both repaired Hodge-star features in one positive multiplier:
`g = exp(a*St_oneD1(St_oneP0(rho)) + b*St_oneD1(SquareD1(St_oneP0(rho))))`.
Its exact typed-GP equivalent is
`ExpP0(AddCP0(MFP0(St_oneD1(St_oneP0(rho)), a), MFP0(St_oneD1(SquareD1(St_oneP0(rho))), b)))`
(13 tree nodes). This tests whether ordinary density response and the
mesh-weighted quadratic response are complementary.

Fit `a,b` independently for all five fixed calibrated FDs on the complete I80
training split, then score the complete chronological validation split. Use two
deterministic Powell restarts, 45 evaluations per restart, bounds
`a in [-5,5]`, `b in [-2500,2500]`, and seed `8130 + baseline_index`. Apply
`is_nonlocal_feasible` because the mesh metric and downstream boundary states
vary. Do not evaluate the test split.
