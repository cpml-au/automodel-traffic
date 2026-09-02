# Attempt 2 plan

Extend the anchored exponent with one quadratic-density term:

`g(rho) = exp((a*rho + b*rho^2)*(1-rho/r_j))`.

The extra term can change the multiplier's interior curvature while retaining
strict positivity and the exact jam anchor. Obtain baseline-specific `r_j` from
`physical_density_limit`. Fit `a,b` separately on full I80 training data using
`fit_candidate`, two deterministic Powell restarts, 30 evaluations per restart,
both bounds `[-6, 6]`, and seed `2100 + 2*10 + baseline_index`. Evaluate only on
full validation after fitting and rank with `E_data + 0.01*15`. Record component
errors, constants, optimizer diagnostics, runtime/RSS, and feasibility. Never
request the held-out test split.
