# Attempt 1 plan

- Candidate: `g(rho) = 1 + a*rho` (5 tree nodes).
- This is the lowest-node direct affine multiplier and includes identity at
  `a = 0`.
- Fit `a` independently for all five fixed calibrated baselines using the full
  I80 prediction training split (times 0--63).
- Use two deterministic Powell starts, at most 45 objective evaluations per
  start, bounds `[-0.9, 4.0]`, and seed `6110 + baseline_index`.
- Reject non-finite or non-positive multipliers, near-zero denominators, and
  corrected velocities that are negative or increase on the baseline's
  physical domain.
- Score exactly once on the full validation split (times 64--107) after fitting.
  Never evaluate the held-out test split.
- Selection fitness: validation data error plus `0.01 * 5`.
