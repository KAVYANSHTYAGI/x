"""Convert real event datasets into EVB shards (the mmap-able, time-indexed training store).

Raw sensor files are decoded by the C++ decoders; only HDF5 needs Python (h5py + hdf5plugin for the
Blosc filter used by DSEC).  Many short recordings (N-ImageNet, HARDVS, ...) are PACKED into one EVB per
shard with an index (sample -> event range, label) so pretraining does not open millions of files.

  # .raw (EVT2/EVT3, e.g. EVK4/IMX636 recordings, eTraM/1Mpx raw) or .dat (Gen1 / 1Mpx td.dat)
  python tools/convert_events.py raw  in.raw  out.evb --sensor IMX636
  # HDF5 with x/y/t/p datasets (presets verified against each dataset's README before use)
  python tools/convert_events.py h5   events.h5 out.evb --preset dsec
  python tools/convert_events.py h5   data.h5   out.evb --keys prophesee/left/x prophesee/left/y prophesee/left/t prophesee/left/p
  # directory of N-ImageNet style .npz files (key 'event_data' with fields x, y, t, p) -> packed shards
  python tools/convert_events.py npz-pack ROOT out_dir --width 640 --height 480 --shard-events 200000000
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "python"))
import evkit  # noqa: E402

PRESETS = {
    # DSEC: events/{x,y,t,p}, t in us relative to t_offset (check the DSEC README for your download)
    "dsec": dict(keys=["events/x", "events/y", "events/t", "events/p"], t_offset="t_offset", width=640, height=480),
}


def write(path, t, x, y, p, W, H, sensor):
    o = np.argsort(t, kind="stable")
    evkit.write_evb(path, torch.from_numpy(t[o].astype(np.int64)), torch.from_numpy(x[o].astype(np.int16)),
                    torch.from_numpy(y[o].astype(np.int16)), torch.from_numpy(p[o].astype(np.uint8)), W, H, sensor)


def cmd_raw(a):
    d = evkit.decode_file(a.input, a.threads, a.format)
    W = a.width or d["width"]
    H = a.height or d["height"]
    if W <= 0 or H <= 0:
        sys.exit("sensor size missing from header: pass --width/--height")
    evkit.C.write_evb(a.output, d["t"], d["x"], d["y"], d["p"], W, H, a.sensor)
    print(f"{a.output}: {len(d['t'])} events")


def cmd_h5(a):
    import h5py
    try:
        import hdf5plugin  # noqa: F401  (registers Blosc etc.)
    except ImportError:
        print("warning: hdf5plugin not installed; compressed datasets (e.g. DSEC) will fail to read")
    pr = PRESETS.get(a.preset, {})
    keys = a.keys or pr.get("keys")
    with h5py.File(a.input, "r") as f:
        x, y, t, p = (np.asarray(f[k]) for k in keys)
        off = int(np.asarray(f[pr["t_offset"]])) if pr.get("t_offset") and pr["t_offset"] in f else 0
    t = (t.astype(np.int64) * a.t_scale).astype(np.int64) + off
    write(a.output, t, x, y, p, a.width or pr.get("width"), a.height or pr.get("height"), a.sensor or a.preset or "")
    print(f"{a.output}: {len(t)} events")


def cmd_npz_pack(a):
    files = sorted(glob.glob(os.path.join(a.input, "**", "*.npz"), recursive=True))
    classes = sorted({os.path.basename(os.path.dirname(f)) for f in files})
    cls_id = {c: i for i, c in enumerate(classes)}
    os.makedirs(a.output, exist_ok=True)
    shard, buf, index, base, n_in = 0, [], [], 0, 0
    def flush():
        nonlocal shard, buf, index, base, n_in
        if not buf:
            return
        t = np.concatenate([b[0] for b in buf]); x = np.concatenate([b[1] for b in buf])
        y = np.concatenate([b[2] for b in buf]); p = np.concatenate([b[3] for b in buf])
        path = os.path.join(a.output, f"shard_{shard:05d}.evb")
        evkit.write_evb(path, torch.from_numpy(t), torch.from_numpy(x.astype(np.int16)),
                        torch.from_numpy(y.astype(np.int16)), torch.from_numpy(p.astype(np.uint8)), a.width, a.height, a.sensor)
        json.dump({"shard": os.path.basename(path), "samples": index}, open(path + ".index.json", "w"))
        print(f"{path}: {len(t)} events, {len(index)} samples")
        shard, buf, index, base, n_in = shard + 1, [], [], 0, 0
    gap_us = 1_000_000                       # samples are laid out on one time axis, 1 s apart (never mixed)
    for f in files:
        ev = np.load(f)[a.key]
        t = ev["t"].astype(np.int64)
        t = t - t.min()
        index.append({"file": os.path.relpath(f, a.input), "label": cls_id[os.path.basename(os.path.dirname(f))],
                      "begin": n_in, "end": n_in + len(t), "t0": base})
        buf.append((t + base, ev["x"], ev["y"], ev["p"]))
        n_in += len(t)
        base += int(t.max()) + gap_us
        if n_in >= a.shard_events:
            flush()
    flush()
    json.dump({"classes": classes}, open(os.path.join(a.output, "classes.json"), "w"))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("raw"); r.add_argument("input"); r.add_argument("output")
    r.add_argument("--threads", type=int, default=0); r.add_argument("--format", default="")
    h = sub.add_parser("h5"); h.add_argument("input"); h.add_argument("output")
    h.add_argument("--preset", default=None, choices=list(PRESETS)); h.add_argument("--keys", nargs=4, default=None)
    h.add_argument("--t-scale", type=float, default=1.0, help="multiply raw t to get microseconds")
    n = sub.add_parser("npz-pack"); n.add_argument("input"); n.add_argument("output")
    n.add_argument("--key", default="event_data"); n.add_argument("--shard-events", type=int, default=200_000_000)
    for s in (r, h, n):
        s.add_argument("--width", type=int, default=0); s.add_argument("--height", type=int, default=0)
        s.add_argument("--sensor", default="")
    a = ap.parse_args()
    {"raw": cmd_raw, "h5": cmd_h5, "npz-pack": cmd_npz_pack}[a.cmd](a)


if __name__ == "__main__":
    main()
