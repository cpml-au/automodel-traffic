# Meta-3 agent 2 summary

All 15 requested Hodge-star fits completed on full train/validation data. The held-out test split was not evaluated.

| FD | Best attempt | Expression | Parameters | Validation fitness | Meta-2 | Change |
|---|---:|---|---|---:|---:|---:|
| Greenshields | 2 | `exp(a*St_oneD1(SquareD1(St_oneP0(rho))))` | a=-14.60026637 | 8.380293 | 7.627391 | +0.752902 |
| IDM | 2 | `exp(a*St_oneD1(SquareD1(St_oneP0(rho))))` | a=-37.11162586 | 7.306958 | 5.955012 | +1.351946 |
| Weidmann | 1 | `exp(a*St_oneD1(St_oneP0(rho)))` | a=-0.0001305520795 | 6.541131 | 6.322293 | +0.218838 |
| Triangular | 2 | `exp(a*St_oneD1(SquareD1(St_oneP0(rho))))` | a=-2.086540996 | 6.522889 | 6.457489 | +0.065400 |
| Del Castillo | 1 | `exp(a*St_oneD1(St_oneP0(rho)))` | a=-0.08033418289 | 6.778627 | 6.603956 | +0.174671 |

All tree sizes were compiled against the repaired live typed grammar. Registered variants: `St_oneP0`, `St_oneP1`, `St_oneD0`, `St_oneD1`.

`test_evaluated = false`.
