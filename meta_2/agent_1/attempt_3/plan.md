# Attempt 3 plan

- Candidate: `g(rho) = exp(c0 + a*rho + b*rho^2)`, adding curvature to the
  intercept-bearing exponential-linear multiplier from attempt 2.
- Fit `c0, a, b` independently for all five calibrated baselines on the complete
  I80 prediction training split.
- Use two Powell starts, at most 45 function evaluations per start, bounds
  `c0 in [-1, 1]`, `a,b in [-5, 5]`, and seed `4130 + baseline_index`.
- Apply the shared positivity, finiteness, nonnegative-velocity, and
  non-increasing-velocity feasibility checks.
- Evaluate fitted constants once on the complete validation split. Never access
  or evaluate the held-out test split.
- Selection fitness: validation `E_data + 0.01 * 12` tree nodes.

