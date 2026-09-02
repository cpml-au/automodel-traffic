# Attempt 3 plan

Attempts 1 and 2 showed modest raw validation improvements from quadratic-Hodge
contrast gating for Greenshields, IDM, and Del Castillo, but neither full tree
beat an incumbent. The ordinary/linear companion terms added complexity, and
attempt 2 was unstable for Del Castillo. A one-parameter centered gate is the
parsimonious refinement:

`g_inc * exp(a * (79*hquad - rho^2) * C)`.

Here `hquad = St_oneD1(SquareD1(St_oneP0(rho)))` and
`C = conv_3(rho,ones)-3*conv_1(rho,ones)`. A direct pre-fit mesh audit found
`hquad=rho^2/79` at homogeneous interior nodes, so `79*hquad-rho^2` removes the
ordinary homogeneous quadratic scale and retains metric/boundary departures.
The typed factor has 22 nodes and one attachment node; complete incumbent trees
remain included.

Fit `a` on full train times 0--63 with two deterministic Powell starts, at most
60 evaluations per start, effective bounds `[-1e6,1e6]`, internal coordinate
`alpha=a/100000`, and seeds `13130 + baseline_index`. Evaluate full validation
times 64--107 with corrected nonlocal feasibility and full-Jacobian wave speeds.
Never evaluate test times 108--179.
