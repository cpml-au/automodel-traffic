# Attempt 1 plan

- Structure: `g(rho) = exp(a*rho/(1+rho))`.
- Hypothesis: a single saturating log-multiplier can shift capacity and speed
  without unbounded high-density amplification and exactly preserves `g(0)=1`.
- Fit each of the five fixed I80 prediction baselines independently on all train
  times with two deterministic Powell starts, 30 evaluations per restart, and
  `a in [-5, 5]`.
- Evaluate the fitted model once on the full validation interval. Never inspect
  or evaluate the held-out test interval.
- Complexity convention: eight expression-tree nodes (all operators and leaves
  in `exp(div(mul(a,rho),add(1,rho)))`).
