"""Train dense continuous-time motion with a pluggable loss.  Example:
    python scripts/train.py --config configs/dsec_mpcm.yaml          # baseline (MotionPriorCMax loss)
    python scripts/train.py --config configs/dsec_ats.yaml           # ours
    torchrun --nproc_per_node=2 scripts/train.py --config configs/dsec_ats.yaml
Overrides: --set key=value (YAML-parsed), e.g. --set batch_size=4 --set loss.name=ats_exact"""
import argparse

import yaml

from atl.train.trainer import TrainConfig, Trainer

ap = argparse.ArgumentParser()
ap.add_argument("--config", required=True)
ap.add_argument("--set", action="append", default=[])
a = ap.parse_args()
d = yaml.safe_load(open(a.config))
for kv in a.set:
    k, v = kv.split("=", 1)
    tgt = d
    *path, last = k.split(".")
    for p in path:
        tgt = tgt.setdefault(p, {})
    tgt[last] = yaml.safe_load(v)
Trainer(TrainConfig(**d)).train()
