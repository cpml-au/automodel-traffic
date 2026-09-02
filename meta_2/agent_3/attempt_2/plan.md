# Attempt 2 plan

- Candidate: `g(rho) = (1 + a*rho)/(1 + b*rho)` (11 tree nodes).
- The numerator and denominator share the identity anchor at zero density;
  their ratio adds saturating curvature while remaining pointwise.
- Fit `a,b` independently for all five fixed calibrated baselines using the
  full I80 prediction training split (times 0--63).
- Use two deterministic Powell starts, at most 45 objective evaluations per
  start, bounds `[-0.9, 4.0]` for each parameter, and seed
  `6120 + baseline_index`.
- Reject non-finite or non-positive multipliers, near-zero/non-positive
  denominators, and corrected velocities that are negative or increase on the
  baseline's physical domain.
- Score exactly once on the full validation split (times 64--107) after fitting.
  Never evaluate the held-out test split.
- Selection fitness: validation data error plus `0.01 * 11`.
