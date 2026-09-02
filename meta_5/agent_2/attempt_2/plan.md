# Attempt 2 plan

Augment every fixed global incumbent with `exp(a*rho^2*C)`, where
`C=conv_3(rho,ones)-3*conv_1(rho,ones)`. The typed factor is
`ExpP0(MFP0(CMulP0(SquareP0(rho),SubCP0(conv_3P0(rho,ones),MFP0(conv_1P0(rho,ones),three))),a))`.
It has 15 nodes; the incumbent attachment makes 16 added nodes.

Fit on full train times 0--63 with two deterministic Powell starts, 60
evaluations per start, bounds `[-15000,15000]`, and seed
`13120+baseline_index`. Evaluate only full validation times 64--107 with the
nonlocal feasibility check and corrected full-Jacobian speed bound. Never test.
