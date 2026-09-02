# Attempt 1 plan

Fit the minimal prescribed positive, jam-anchored correction

`g(rho) = exp(a*rho*(1-rho/r_j))`

separately for all five fixed basic fundamental diagrams. For every baseline,
`r_j` is obtained from `physical_density_limit(baseline)`. Fit `a` on the full
I80 training split with `fit_candidate`, two deterministic Powell restarts,
30 evaluations per restart, bounds `[-6, 6]`, and seed
`2100 + 1*10 + baseline_index`. Evaluate exactly once on the full validation
split after fitting and rank with `E_data + 0.01*10`. Check strict multiplier
positivity, finite velocity, non-negative velocity, and non-increasing velocity
on the physical baseline domain. Do not request or inspect test results.
