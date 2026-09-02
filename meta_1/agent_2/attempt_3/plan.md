# Attempt 3 plan

Add a cubic-density term to the anchored exponent:

`g(rho) = exp((a*rho + b*rho^2 + c*rho^3)*(1-rho/r_j))`.

This is the most flexible prescribed lineage while remaining strictly positive
and exactly one at each baseline's physical jam density. Obtain `r_j` from
`physical_density_limit`. Fit `a,b,c` separately on the full I80 training split
with `fit_candidate`, two deterministic Powell restarts, 30 evaluations per
restart, bounds `[-6, 6]`, and seed
`2100 + 3*10 + baseline_index`. Evaluate only on the full validation split and
rank with `E_data + 0.01*21`. Log errors, constants, optimizer diagnostics,
runtime/RSS, and feasibility. Never request or inspect test data.
