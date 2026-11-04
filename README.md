# atl — Anchored Transport Likelihood for event-camera motion

**Target paper:** MotionPriorCMax — Hamann, Wang, Asmanis, Chaney, Gallego, Daniilidis,
*Motion-prior Contrast Maximization for Dense Continuous-Time Motion Estimation*, ECCV 2024
(code: github.com/tub-rip/MotionPriorCMax, MIT).
It is the self-supervised state of the art for dense, continuous-time event motion.

**Numbers to beat**

| Benchmark | Metric | MotionPriorCMax |
|---|---|---|
| DSEC-Flow test | EPE / AE / %Out | 3.20 / 8.53 / 15.21 |
| EVIMO2 300 ms | TEPE / TAE / %Out | 6.14 / 16.98 / 0.254 |
| DSEC-Flow test (secondary target, Taming CMax ICCV'23) | EPE | 2.33 |

**Thesis.** Every loss in the contrast-maximisation family scores events *after moving them into a reference frame*.
Such scores are not proper: their optimum is not the true motion. That is why each method needs its own patch:
random or multiple reference times, KNN soft assignment, TV / divergence / deformation regularisers with tuned
weights, or sparse ground truth for "anchoring" (STSC, CVPR'26).

**Our objective.** The *anchored transport score* scores every **raw** event under the other events transported to
its own timestamp. It is proper and needs no reference time and no regulariser weight. It contains the exact
change-of-variables term that the regularisers approximate. Theory: `../research_notes/PROPER_ALIGNMENT_MATH.md`.

## Layout

```
csrc/            C++/OpenMP + CUDA kernels (sm_80/sm_90; 22-48 registers, no spills)
  atl_cpu.cpp    splat (bilinear / Gaussian), bilinear gather, anchored pair sums (spatial hash), voxel grid
  atl_cuda.cu    same loops, one thread per event / anchor, float atomics
atl/ops          autograd wrappers, JIT build (atl/ops/_ext.py), pure-torch references (tests, fallback)
atl/motion       parametric fields (translation, zoom, Sim2, affine, planar, 3-D rotation, dense) with the classic
                 linear warp and the exact flow (augmented matrix exponential / RK4); MotionPriorCMax pixel
                 trajectories (polynomial / Bezier, fixed-point inversion)
atl/objectives   baselines: CMax variance, multi-reference, gradient magnitude, Shiba R_D / R_A, EMin, ST-PPP,
                 Zhu'19 timestamp, Taming CMax timestamp, faithful MotionPriorCMax loss (mpcm.py);
                 ours: AnchoredExact (pair kernel), AnchoredSliced (any motion); metrics FWL, RSAT, held-out score
atl/data         DSEC (official + MotionPriorCMax layouts, flow PNG I/O, benchmark writer), synthetic DVS generator
atl/models       U-Net matched to MotionPriorCMax's DSEC network
atl/train        trainer with pluggable loss (DDP-ready); configs/dsec_mpcm.yaml vs configs/dsec_ats.yaml
atl/suite        failure-mode suite F1-F4 (scripts/run_suite.py); dense test F6 (scripts/exp_dense_trajectories.py)
tests/           ops vs references (values + grads), objectives, motion, I/O
```

## Attack plan

1. **Failure suite** (synthetic, exact ground truth): each baseline objective is run through the same tasks.
   * F1 collapse;
   * F2 reference-time bias;
   * F3 regulariser-weight dilemma;
   * F4 unsupervised metrics fooled by collapse;
   * F6 dense trajectories (the target's representation) optimised from the ground truth.
2. **Model-based on real data.** DSEC ground-truth windows, per-window dense-flow optimisation under each
   objective (a MultiCM-style comparison).
3. **Learning, controlled.** Identical U-Net, data and schedule; only the loss changes. Evaluated on DSEC
   (test-server submission via `atl.data.dsec.write_flow_png`) and on EVIMO2 continuous-time tracks.
4. **Ablations of the target's patches under our loss** (they should become unnecessary):
   * random t_ref;
   * KNN-32;
   * TV λ.

## Quick start

```
pip install -e .                                   # extension JIT-compiles on first use (CUDA if available)
pytest -q                                          # ops, objectives, I/O
python scripts/scan_collapse.py exact              # one-screen collapse comparison
python scripts/run_suite.py --mode exact           # F1-F4 -> outputs/suite_exact.md
python scripts/exp_dense_trajectories.py           # F6
python scripts/train.py --config configs/dsec_mpcm.yaml   # baseline
python scripts/train.py --config configs/dsec_ats.yaml    # ours (only the loss differs)
```
