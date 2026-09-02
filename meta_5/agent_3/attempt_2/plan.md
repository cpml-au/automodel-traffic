# Attempt 2 plan

Refine every fixed global incumbent with `exp(a*hlin*C + b*hquad*C)`, where
`hlin = St_oneD1(St_oneP0(rho))`,
`hquad = St_oneD1(SquareD1(St_oneP0(rho)))`, and `C` is the level-cancelling
three-/one-cell convolution contrast. The full tree includes the incumbent,
33-node typed factor, and one attachment node.

Fit only `a,b` on full I80 train times 0--63 with two deterministic Powell
starts, at most 60 evaluations per start, bounds `a in [-2000,2000]` and
`b in [-1e6,1e6]`, and seeds `13120 + baseline_index`. Evaluate full
validation times 64--107 only. Use corrected nonlocal feasibility and
full-Jacobian wave speeds. Test times 108--179 remain untouched.

Powell uses internal coordinates `alpha=a/500` and `beta=b/100000`; reports
contain the effective `a,b` without changing the candidate expression.
