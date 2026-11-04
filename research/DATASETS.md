# Big, real event datasets for the point-process pretraining project

Replaces DVS128 Gesture / N-Caltech101 / toy simulations as the main evidence. Figures come from the papers
and project pages found in this session. **Verify size, format and licence on each download page before
planning storage or publishing results**; entries marked *(unverified)* could not be confirmed.

## Tier A: pretraining corpus (long, real, high-resolution, several sensor generations)

| Dataset | Sensor / resolution | Scale | Format | Labels | Notes |
|---|---|---|---|---|---|
| [Prophesee Gen1 Automotive](https://arxiv.org/abs/2001.08499) | ATIS Gen1, 304×240 | > 39 h driving | `.dat` (u32 ts + u32 packed x,y,p) | car / pedestrian boxes | Access needs agreement to Prophesee T&C |
| [Prophesee 1 Mpx Automotive](https://www.prophesee.ai/2020/11/24/automotive-megapixel-event-based-dataset/) | Gen4, 1280×720 | 14–15 h, 25 M boxes | `.dat` + numpy labels | road-user boxes | Highest-volume real HD driving set |
| [eTraM](https://arxiv.org/abs/2403.19976) | EVK4 HD, 1280×720 | ~10 h, 2 M boxes | *(format unverified)* | 8 traffic classes | **Static camera**: motion comes from the scene, not ego-motion, a different rate regime |
| [M3ED](https://registry.opendata.aws/m3ed/) | stereo 1280×720 *(Prophesee model unverified)* | 70 seq, ~3.4 h | HDF5 on AWS Open Data | depth, pose, semantics | Car, legged robot, drone: three motion regimes |
| [EventKitchen](https://arxiv.org/abs/2608.04865) (2026) | stereo Gen4 | 5.5 h egocentric | *(unverified)* | RGB, depth, IMU | Hand–object interaction, indoor lighting |
| [DSEC](https://udel.edu/~ghuang/icra21-vins-workshop/papers/05-Gehrig_DSEC.pdf) | Gen3.1 640×480 stereo | 53 seq, ~1 h | HDF5 (Blosc) | disparity, flow, detection (DSEC-Det) | Small, but the standard driving benchmark |

**About 73 h of real streams from three sensor generations** (ATIS Gen1, Gen3.1, Gen4/IMX636 class), indoor
and outdoor, ego-moving and static cameras. That is two to three orders of magnitude more events than DVS
Gesture + N-Caltech101, and it gives the cross-sensor axis (R3) for free.

## Tier B: downstream evaluation with real shifts

| Dataset | Sensor | Scale | Why it matters |
|---|---|---|---|
| [N-ImageNet](https://github.com/82magnolia/n_imagenet) | event camera moving over a monitor *(model unverified)* | ~1.3 M train / 50 k val, 1000 classes, ~400 GB; mini = 100 classes | **9 extra validation sets with real camera-trajectory (5) and brightness (4) changes**. This is the real version of the synthetic speed / threshold shifts in CE-GNN |
| [HARDVS](https://arxiv.org/abs/2211.09648) | DAVIS346, 346×260 | 107,646 recordings, 300 activities, 5–10 s | Large, genuinely temporal, multi-view / illumination / motion |
| [DailyDVS-200](https://arxiv.org/abs/2407.05106) | *(unverified)* | > 22,000 sequences, 200 actions, 47 people, 14 attributes | Attribute-stratified robustness |
| [EvDET200K](https://arxiv.org/abs/2412.06647) | EVK4-HD 1280×720 | 10,054 clips (2–5 s), 10 classes, ~200 k boxes | HD detection under weather and day/night |
| [EventVOT](https://openaccess.thecvf.com/content/CVPR2024/papers/Wang_Event_Stream-based_Visual_Object_Tracking_A_High-Resolution_Benchmark_Dataset_and_CVPR_2024_paper.pdf) | EVK4-HD | 1,141 videos | HD tracking: a streaming, time-critical task |
| [PEOD](https://arxiv.org/abs/2511.08140) | EVK4 1280×720 + aligned RGB | 340 k boxes | Hard conditions with pixel-aligned frames |
| [E-POSE](https://www.nature.com/articles/s41597-025-04536-5) | *(unverified)* | 306 sequences, ~1.5 B events | Pose regression |
| [EventDrive](https://arxiv.org/abs/2606.18242) (CVPR 2026) | re-annotates M3ED + DSEC + PKU-DAVIS-SOD | 471 k event–frame–language samples | Only if you add a language readout |

## Recommended protocol for the ICML submission

1. **Pretrain** on Tier A (start with 1Mpx + eTraM + M3ED: three motion regimes, one sensor class), then add
   Gen1 and DSEC as held-out sensors for the cross-sensor result.
2. **Probe / fine-tune** on N-ImageNet (full or mini), HARDVS and DailyDVS-200.
3. **Real shift evaluation**: N-ImageNet's 9 robustness sets replace synthetic time-rescaling. Report accuracy,
   the time-rescaling KS calibration, and temporal information gain on each.
4. Keep DVS Gesture / N-Caltech101 only as a sanity table for comparability with prior work.

## Storage and I/O (with evkit)

* EVB stores 13 B/event. Prophesee EVT3 is about 1.7–2 B/event (measured 1.70 on bursty synthetic data).
  For the TB-scale sets, keep the raw `.raw`/`.dat` files and decode on the fly (the decoder runs at roughly
  200 Mev/s on 4 cores here; it scales with cores). EVB is for the smaller sets and for packed shards.
* Conversion:
  * `evkit-convert in.raw out.evb` for `.raw` / `.dat` files.
  * `tools/convert_events.py h5 ... --preset dsec` for HDF5 (DSEC, M3ED).
  * `tools/convert_events.py npz-pack` for N-ImageNet-style per-sample `.npz` files. This writes packed shards with a sample index, so millions of short recordings don't each become a file.
* Licences differ: the Prophesee sets require accepting their T&C; for the others, check each project page (not verified here). Record them before redistribution.
