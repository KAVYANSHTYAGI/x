# Failure-mode suite (mode=exact)

## f1: f1_collapse  (277s)

| objective | min_rel_loss_on_path | argmin_h | opt_from_truth_h | opt_from_truth_trans_err | proper_on_path |
|---|---|---|---|---|---|
| CMax variance (t_ref=0) | -85.712 | +16.000 | +0.023 | +0.173 | False |
| CMax multi-ref {0,.5,1} | -27.672 | +16.000 | +0.001 | +0.132 | False |
| MPCM focus (|grad| L1, rand t_ref) | +0.165 | +0.000 | +0.004 | +0.239 | True |
| Shiba R_D (lambda=1) | -85.256 | +16.000 | +0.023 | +0.170 | False |
| Shiba R_A (lambda=1) | -85.690 | +16.000 | +0.023 | +0.171 | False |
| EMin | -0.638 | +16.000 | +0.938 | +1.956 | False |
| ST-PPP | -0.895 | -16.000 | -0.081 | +2.495 | False |
| Zhu timestamp | -0.977 | -16.000 | +2.251 | +2.407 | False |
| Taming timestamp | -0.978 | -16.000 | +2.180 | +1.674 | False |
| ATS sliced (ours) | +0.174 | +0.000 | -0.008 | +0.417 | True |

## f2: f2_reference_bias  (65s)

| objective | rel_bias@h*=0.1 | rel_bias@h*=0.25 | rel_bias@h*=0.5 |
|---|---|---|---|
| CMax variance (t_ref=0) | +0.034 | +0.029 | +0.021 |
| CMax multi-ref {0,.5,1} | -0.135 | -0.035 | -0.024 |
| MPCM focus (|grad| L1, rand t_ref) | -0.112 | -0.045 | -0.034 |
| Shiba R_D (lambda=1) | +0.032 | +0.026 | +0.020 |
| Shiba R_A (lambda=1) | +0.034 | +0.027 | +0.021 |
| EMin | -0.000 | +0.017 | +0.016 |
| ST-PPP | -0.080 | +0.001 | +0.013 |
| Zhu timestamp | +0.030 | +0.006 | +0.018 |
| Taming timestamp | +0.031 | +0.007 | +1.500 |
| ATS sliced (ours) | +0.070 | -0.014 | -0.022 |

## f3: f3_lambda  (663s)

| regulariser | lambda | translation_scene_spurious_h | zoom_scene_rel_bias |
|---|---|---|---|
| R_div | +0.000 | +0.016 | +0.027 |
| R_div | +0.010 | +0.016 | +0.027 |
| R_div | +0.100 | +0.015 | +0.026 |
| R_div | +1.000 | +0.016 | +0.025 |
| R_div | +10.000 | +0.015 | +0.015 |
| R_div | +100.000 | +0.016 | -0.098 |
| R_det | +0.000 | +0.015 | +0.027 |
| R_det | +0.010 | +0.016 | +0.026 |
| R_det | +0.100 | +0.015 | +0.026 |
| R_det | +1.000 | +0.015 | +0.026 |
| R_det | +10.000 | +0.015 | +0.019 |
| R_det | +100.000 | +0.015 | -0.046 |
| ATS sliced (ours) | none | -0.003 | -0.060 |

## f4: f4_metrics  (6s)

| candidate | EPE | FWL (higher=better) | RSAT (lower=better) | held-out ATS (higher=better) |
|---|---|---|---|---|
| truth | +0.000 | +4.028 | +0.712 | -8.075 |
| collapsed (h=+8) | +128700.133 | +361.934 | +0.049 | -10.397 |
| collapsed (h=-8) | +46.706 | +0.005 | +0.049 | -10.409 |
| zero motion | +25.495 | +1.000 | +1.000 | -8.837 |
| truth + 4px | +4.000 | +3.187 | +0.878 | -8.219 |
