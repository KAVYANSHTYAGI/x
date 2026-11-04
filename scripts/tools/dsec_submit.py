"""DSEC-Flow test-set inference -> benchmark submission folder (zip and upload to the DSEC website).
    python scripts/tools/dsec_submit.py --ckpt runs/dsec_ats/best.pt --test_root data/dsec/test \
        --timestamps data/dsec/test_forward_optical_flow_timestamps --out submissions/ats
Layout written: <out>/<sequence>/<file_index:06d>.png  (16-bit, R = x*128 + 2^15, G = y*128 + 2^15, B = 1)."""
import argparse
from pathlib import Path

import torch

from atl.data.dsec import TEST_SEQS, DSECFlowDataset, collate, write_flow_png
from atl.models.unet import UNet
from atl.motion import PixelTrajectories
from atl.ops import voxel_grid

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", required=True)
ap.add_argument("--test_root", required=True)
ap.add_argument("--timestamps", required=True, help="directory with <seq>.csv files")
ap.add_argument("--out", required=True)
a = ap.parse_args()

dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ck = torch.load(a.ckpt, map_location="cpu")
cfg = ck.get("config", {})
nb, K = cfg.get("num_bins", 15), cfg.get("K", 1)
net = UNet(nb, 2 * K).to(dev).eval()
net.load_state_dict(ck["model"])
ds = DSECFlowDataset(a.test_root, TEST_SEQS, windows="test", test_csv_dir=a.timestamps)
dl = torch.utils.data.DataLoader(ds, batch_size=1, num_workers=4, collate_fn=collate)
n = 0
with torch.no_grad():
    for batch in dl:
        ev = batch["events"].to(dev)
        vox = voxel_grid(ev.xy, ev.t, ev.p, ev.b, 1, nb, 480, 640)
        flow = PixelTrajectories(net(vox), K=K, basis_kind=cfg.get("basis", "polynomial")).flow_field(1, 480, 640)[0]
        write_flow_png(Path(a.out) / batch["seqs"][0] / f"{batch['file_index'][0]:06d}.png", flow.cpu().numpy())
        n += 1
print(f"wrote {n} flow maps to {a.out}; zip the sequence folders and upload to the DSEC benchmark")
