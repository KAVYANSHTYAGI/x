#!/usr/bin/env bash
# Sequential evidence runs (CPU-friendly).  Results in outputs/.
set -u
PY=${PY:-python}
mkdir -p outputs
for sc in zoom affine; do
  for o in mpcm ats_exact; do
    m=0; [ "$o" = ats_exact ] && m=6
    $PY scripts/experiments/diag_dense_drift.py --objective $o --scene $sc --anchor_margin $m 2>&1 | grep -v Warn > outputs/diag_${sc}_${o}.txt
  done
done
for o in gradmag cmax_var; do
  $PY scripts/experiments/diag_dense_drift.py --objective $o --scene translation 2>&1 | grep -v Warn > outputs/diag_translation_${o}.txt
done
$PY scripts/experiments/run_suite.py --mode exact 2>&1 | grep -v Warn > outputs/suite_exact.log
$PY scripts/experiments/exp_dsec_modelbased.py --windows 5,20,36 2>&1 | grep -v Warn > outputs/dsec_modelbased.log
echo ALL_DONE >> outputs/dsec_modelbased.log
