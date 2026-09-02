# Attempt 1 plan

Fit all five fixed meta-3 incumbents multiplied by
`exp(a*SqrtP0(rho))`. In executable JAX, `SqrtP0(rho)` is protected as
`sqrt(max(rho, 0))`; this is the GP `SqrtP0` equivalent for nonnegative P0
density cochains.

Use full I80 train times 0--63 and full validation times 64--107, two
deterministic Powell restarts, 60 evaluations per restart, coefficient bounds
`[-5, 5]`, and seeds `12110 + baseline_index`. The complete tree includes the
fixed incumbent plus the six-node added product/exponential factor. Evaluate
homogeneous-state feasibility with `is_nonlocal_feasible`, because some fixed
incumbents contain convolutions. Do not evaluate test times 108--179.
