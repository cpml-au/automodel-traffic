# Attempt 1 evaluation

The fitted constants were frozen and evaluated once on the complete I80
prediction validation split (times 64--107). The held-out test split was not
evaluated. Fitness is `validation data error + 0.01*8`, so the complexity charge
is 0.08.

| Baseline | Validation density error | Validation velocity error | Validation data error | Validation fitness | Peak RSS (MB) | Feasible |
|---|---:|---:|---:|---:|---:|:---:|
| Greenshields | 10.473258 | 7.642451 | 9.057855 | 9.137855 | 891.609 | yes |
| IDM | 6.904508 | 8.908321 | 7.906415 | 7.986415 | 1323.715 | yes |
| Weidmann | 6.543904 | 6.305277 | 6.424590 | 6.504590 | 1535.223 | yes |
| Triangular | 6.515769 | 9.196405 | 7.856087 | 7.936087 | 1721.805 | yes |
| Del Castillo | 6.086124 | 7.134638 | 6.610381 | 6.690381 | 1932.727 | yes |

The simplest saturating family transferred best for Weidmann and Del Castillo.
Its IDM and Triangular corrections lowered training error but generalized poorly
relative to their identity baselines, making them weak candidates for those FDs.
Peak RSS is the process high-water mark and therefore grows cumulatively as the
five baselines are processed sequentially.
