"""Self-supervised training of dense continuous-time motion with a pluggable alignment loss.

The *only* difference between the baseline run (MotionPriorCMax loss) and ours is ``loss.name`` in the config;
network (U-Net, 15 voxel bins), trajectory prior (degree-1 polynomial on DSEC), data, optimiser and schedule are
shared.  Launch single-GPU with ``python scripts/train.py --config ...`` or multi-GPU with
``torchrun --nproc_per_node=2 scripts/train.py --config ...``.
"""
from __future__ import annotations

import json
import math
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

import torch
import torch.distributed as dist

from ..eval.flow_metrics import Accumulator, flow_errors
from ..models.unet import UNet
from ..motion import PixelTrajectories
from ..objectives import build as build_objective
from ..ops import voxel_grid


@dataclass
class TrainConfig:
    # data
    dataset: str = "dsec"                 # "dsec" | "synthetic"
    data_root: str = "data/dsec/train"
    train_windows: str = "images"         # dsec: "images" (MotionPriorCMax) | "gt" | "sliding"
    val_root: Optional[str] = None        # defaults to data_root
    max_events: Optional[int] = None      # per sample for the LOSS (network input always uses all events)
    num_workers: int = 8
    # model
    num_bins: int = 15
    K: int = 1
    basis: str = "polynomial"
    base_channels: int = 64
    # loss
    loss: dict = field(default_factory=lambda: {"name": "mpcm"})
    # optimisation (MotionPriorCMax DSEC: Adam, lr 1e-4, total batch 14 on 2 GPUs, 100 epochs in the public config)
    batch_size: int = 7                   # per GPU
    lr: float = 1e-4
    epochs: int = 100
    grad_clip: float = 10.0
    weight_decay: float = 0.0
    schedule: str = "constant"            # "constant" | "cosine"
    # bandwidth annealing for anchored losses (start, end, fraction of training)
    anneal: Optional[list] = None
    # bookkeeping
    out_dir: str = "runs/default"
    val_every: int = 1
    log_every: int = 50
    seed: int = 0
    max_steps: Optional[int] = None       # smoke tests

    @staticmethod
    def from_yaml(path) -> "TrainConfig":
        import yaml
        d = yaml.safe_load(open(path))
        return TrainConfig(**d)


def _ddp():
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ and not dist.is_initialized():
        dist.init_process_group("nccl" if torch.cuda.is_available() else "gloo")
    if dist.is_initialized():
        return dist.get_rank(), dist.get_world_size()
    return 0, 1


def build_datasets(cfg: TrainConfig):
    if cfg.dataset == "dsec":
        from ..data.dsec import DSECFlowDataset, TRAIN_SEQS, VAL_SEQS
        train = DSECFlowDataset(cfg.data_root, TRAIN_SEQS, windows=cfg.train_windows, max_events=None)
        val = DSECFlowDataset(cfg.val_root or cfg.data_root, VAL_SEQS, windows="gt")
        return train, val, (480, 640)
    if cfg.dataset == "synthetic":
        from ..data.synthetic_dataset import SyntheticFlowDataset
        tr = SyntheticFlowDataset(n=64, seed=0)
        va = SyntheticFlowDataset(n=8, seed=1000)
        return tr, va, (tr.H, tr.W)
    raise ValueError(cfg.dataset)


def _collate_fn(cfg, H, W):
    from ..data.dsec import collate as dsec_collate

    def f(items):
        b = dsec_collate(items, H=H, W=W)
        b["H"], b["W"] = H, W
        return b
    return f


class Trainer:
    def __init__(self, cfg: TrainConfig):
        self.cfg = cfg
        self.rank, self.world = _ddp()
        self.dev = torch.device(f"cuda:{int(os.environ.get('LOCAL_RANK', 0))}" if torch.cuda.is_available() else "cpu")
        torch.manual_seed(cfg.seed + self.rank)
        self.train_ds, self.val_ds, (self.H, self.W) = build_datasets(cfg)
        self.model = UNet(cfg.num_bins, 2 * cfg.K, cfg.base_channels).to(self.dev)
        if self.world > 1:
            self.model = torch.nn.parallel.DistributedDataParallel(
                self.model, device_ids=[self.dev.index] if self.dev.type == "cuda" else None)
        self.loss_fn = build_objective(**cfg.loss)
        self.opt = torch.optim.Adam(self.model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        sampler = torch.utils.data.distributed.DistributedSampler(self.train_ds) if self.world > 1 else None
        self.train_dl = torch.utils.data.DataLoader(
            self.train_ds, batch_size=cfg.batch_size, shuffle=sampler is None, sampler=sampler,
            num_workers=cfg.num_workers, collate_fn=_collate_fn(cfg, self.H, self.W), drop_last=True,
            persistent_workers=cfg.num_workers > 0)
        self.val_dl = torch.utils.data.DataLoader(self.val_ds, batch_size=1, shuffle=False,
                                                  num_workers=min(cfg.num_workers, 4),
                                                  collate_fn=_collate_fn(cfg, self.H, self.W))
        self.out = Path(cfg.out_dir)
        if self.rank == 0:
            self.out.mkdir(parents=True, exist_ok=True)
            json.dump(asdict(cfg), open(self.out / "config.json", "w"), indent=1)
        self.step = 0
        self.total_steps = cfg.max_steps or cfg.epochs * max(len(self.train_dl), 1)
        self.best = math.inf

    # ------------------------------------------------------------------------------------------------
    def motion_from_batch(self, batch):
        ev = batch["events"].to(self.dev)
        vox = voxel_grid(ev.xy, ev.t, ev.p, ev.b, ev.batch_size, self.cfg.num_bins, self.H, self.W)
        coeff = self.model(vox)
        return ev, PixelTrajectories(coeff, K=self.cfg.K, basis_kind=self.cfg.basis)

    def _lr_at(self):
        if self.cfg.schedule == "cosine":
            return self.cfg.lr * 0.5 * (1 + math.cos(math.pi * min(self.step / self.total_steps, 1.0)))
        return self.cfg.lr

    def _anneal(self):
        a = self.cfg.anneal
        if a and hasattr(self.loss_fn, "set_bandwidth"):
            h0, h1, frac = a
            u = min(1.0, self.step / max(frac * self.total_steps, 1))
            self.loss_fn.set_bandwidth(h0 + (h1 - h0) * u)

    def train(self):
        cfg = self.cfg
        log = open(self.out / "log.jsonl", "a") if self.rank == 0 else None
        t0 = time.time()
        for epoch in range(cfg.epochs):
            if hasattr(self.train_dl.sampler, "set_epoch"):
                self.train_dl.sampler.set_epoch(epoch)
            self.model.train()
            for batch in self.train_dl:
                for g in self.opt.param_groups:
                    g["lr"] = self._lr_at()
                self._anneal()
                ev, motion = self.motion_from_batch(batch)
                loss_ev = ev.subsample(cfg.max_events * ev.batch_size) if cfg.max_events else ev
                out = self.loss_fn(loss_ev, motion)
                self.opt.zero_grad(set_to_none=True)
                out["loss"].backward()
                if cfg.grad_clip:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), cfg.grad_clip)
                self.opt.step()
                self.step += 1
                if log and self.step % cfg.log_every == 0:
                    rec = {"step": self.step, "epoch": epoch, "loss": float(out["loss"].detach()),
                           "lr": self._lr_at(), "sec": time.time() - t0}
                    rec.update({k: float(v) for k, v in out.items() if k != "loss" and torch.is_tensor(v) and v.numel() == 1})
                    log.write(json.dumps(rec) + "\n"); log.flush()
                    print(rec, flush=True)
                if cfg.max_steps and self.step >= cfg.max_steps:
                    break
            if (epoch + 1) % cfg.val_every == 0 or (cfg.max_steps and self.step >= cfg.max_steps):
                res = self.validate()
                if self.rank == 0:
                    rec = {"step": self.step, "epoch": epoch, **{f"val/{k}": v for k, v in res.items()}}
                    log.write(json.dumps(rec) + "\n"); log.flush()
                    print(rec, flush=True)
                    self.save("last.pt")
                    if res.get("EPE", math.inf) < self.best:
                        self.best = res["EPE"]
                        self.save("best.pt")
            if cfg.max_steps and self.step >= cfg.max_steps:
                break
        if log:
            log.close()

    @torch.no_grad()
    def validate(self):
        self.model.eval()
        acc = Accumulator()
        for batch in self.val_dl:
            if "flow" not in batch:
                continue
            ev, motion = self.motion_from_batch(batch)
            pred = motion.flow_field(ev.batch_size, self.H, self.W)
            acc.add(flow_errors(pred, batch["flow"].to(self.dev), batch["valid"].to(self.dev)))
        self.model.train()
        return acc.result()

    def save(self, name):
        m = self.model.module if hasattr(self.model, "module") else self.model
        torch.save({"model": m.state_dict(), "step": self.step, "config": asdict(self.cfg)}, self.out / name)
