# GPU runbook: the decisive experiments

## 0. Environment
```bash
conda create -n atl python=3.10 -y && conda activate atl
pip install torch --index-url https://download.pytorch.org/whl/cu121   # match your CUDA
pip install -e ".[dev]" && pip install pykeops                         # pykeops: fast KNN for the baseline
pytest -q                                   # MUST pass; the CUDA kernels run here for the first time
python bench/bench_ops.py --device cuda --events 1000000                # size n_anchors / max_events / tau
```
**If a CUDA test fails, stop and fix it before training.** The CPU versions are verified; the CUDA versions have never
run.

## 1. Data
```bash
# DSEC (official layout works directly): https://dsec.ifi.uzh.ch
data/dsec/train/<seq>/<seq>_events_left/{events.h5,rectify_map.h5}
data/dsec/train/<seq>/<seq>_optical_flow_forward_event/*.png  +  <seq>_optical_flow_forward_timestamps.txt
data/dsec/test/<seq>/...   +  test timestamp CSVs (test_forward_optical_flow_timestamps.zip)
```
Training uses MotionPriorCMax's 24 sequences with windows at every second image timestamp (`train_windows: images`).
That needs `images/timestamps.txt` per sequence. Otherwise use `sliding`.

## 2. Step 0: reproduce the baseline through our pipeline
```bash
python scripts/tools/convert_mpcm_checkpoint.py <official DSEC checkpoint> runs/mpcm_official/best.pt
python scripts/tools/dsec_submit.py --ckpt runs/mpcm_official/best.pt --test_root data/dsec/test \
    --timestamps <csv dir> --out submissions/mpcm_official     # upload: expect EPE ~3.20
```
Gate: our evaluation pipeline reproduces their number.

## 3. Step 1: retrain the baseline with our port of their loss
`torchrun --nproc_per_node=2 scripts/train.py --config configs/dsec_mpcm.yaml`
(official setting: total batch 14, Adam 1e-4).

Gate: validation EPE within ~0.1 of the official checkpoint's.

## 4. Step 2: ours (only the loss differs)
`torchrun --nproc_per_node=2 scripts/train.py --config configs/dsec_ats.yaml`

Watch `runs/dsec_ats/log.jsonl`.
* If validation EPE stalls early (basin, `docs/ATTACK_SURFACE.md` W2), try:
  * `--set anneal=[8.0,1.0,0.4]`;
  * fine-tune from the baseline: load `runs/dsec_mpcm/best.pt` into the model before training (add a
    `init_ckpt` hook in `Trainer.__init__`).

## 5. Step 3: ablations (validation set)
* **Ours:**
  * `loss.smooth_anchors=false`;
  * `loss.dequantize=false`;
  * `loss.visibility=false`;
  * `loss.anchor_margin=0`;
  * no annealing;
  * `loss.n_anchors` ∈ {20k, 60k, 120k};
  * `loss.tau` ∈ {0.25, 0.5, 1}.
* **Baseline without its patches:**
  * `loss.smooth_weight=0`;
  * `loss.num_knn=1`;
  * `loss.t_ref=0.0`.

  Expect large degradation for theirs (their paper: up to 7.26 EPE).

## 6. Step 4: EVIMO2 (continuous time, TEPE)
1. Clone the official repo.
2. Run `atl.models.bflow_adapter.check_against_official` once.
3. Wrap their RAFT-spline output with `trajectories_from_bflow` and train with `ats_sliced` (Bezier prior is not
   linear at the event).
4. Evaluate with `atl.data.evimo2.evaluate_tracks`. Target TEPE < 6.14.

## 7. Report
* DSEC test server: EPE / AE / 1PE / 3PE, plus a per-sequence table.
* Event-pixel vs event-free-pixel EPE (`ATTACK_SURFACE.md` Exp 1).
* Timings.
