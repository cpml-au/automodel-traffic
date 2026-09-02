# Attempt 1 plan

Refine every fixed global incumbent from `automodel/final_candidates.json` with
`exp(a*C + b*hquad*C)`, where
`C = conv_3(rho,ones)-3*conv_1(rho,ones)` and
`hquad = St_oneD1(SquareD1(St_oneP0(rho)))`. This jointly permits the ordinary
contrast and a repaired-Hodge quadratic-density gate. The full tree count is
the incumbent count plus the 29-node typed factor and one `CMulP0` attachment.

Fit only `a,b` on full I80 train times 0--63 with two deterministic Powell
starts, at most 60 evaluations per start, bounds `a in [-500,500]` and
`b in [-1e6,1e6]`, and seeds `13110 + baseline_index`. Select on full
validation times 64--107 using `E_data + 0.01*full_tree_nodes`. Use
`is_nonlocal_feasible` and the full-Jacobian row-sum speed bound. Never evaluate
test times 108--179.

For numerical conditioning, Powell uses dimensionless internal coordinates
`alpha=a/100` and `beta=b/100000`; reports contain the effective `a,b`.
