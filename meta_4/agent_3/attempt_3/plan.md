# Attempt 3 plan

Fit all five fixed meta-3 incumbents multiplied by
`exp(a*SqrtP0(rho) + b*rho^2)`. The executable protected square root is
`sqrt(max(rho, 0))`, equivalent to GP `SqrtP0` on a P0 density cochain.

Use full I80 train times 0--63 and full validation times 64--107, two
deterministic Powell restarts, 60 evaluations per restart, coefficient bounds
`[-5, 5]`, and seeds `12130 + baseline_index`. Count all incumbent nodes and
the twelve-node added factor. Use `is_nonlocal_feasible`, and never access held-
out test times 108--179.
