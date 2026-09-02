# Attempt 3 plan

Augment every fixed global incumbent with `exp(a*rho*C+b*rho^2*C)`, using a
separate repeated typed subtree for each `C`. The exact typed factor is in
`model.py`; it has 29 nodes and the incumbent attachment makes 30 added nodes.

Fit on full train times 0--63 with two deterministic Powell starts, 60
evaluations per start, bounds `a in [-3000,3000]`, `b in [-15000,15000]`, and
seed `13130+baseline_index`. Evaluate only full validation times 64--107 with
the nonlocal feasibility check and corrected full-Jacobian speed. Never test.
