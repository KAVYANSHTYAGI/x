"""Load an official MotionPriorCMax DSEC U-Net checkpoint (Lightning .ckpt or extracted weights) into atl's UNet.

Step zero of the comparison: evaluate their released weights through *our* data/eval path and check that we
reproduce their validation numbers before training anything.
    python scripts/tools/convert_mpcm_checkpoint.py weights/<their>.ckpt runs/mpcm_official/best.pt
"""
import re
import sys

import torch

from atl.models.unet import UNet

src, dst = sys.argv[1], sys.argv[2]
sd = torch.load(src, map_location="cpu")
sd = sd.get("state_dict", sd)
out = {}
for k, v in sd.items():
    k = re.sub(r"^model\.", "", k)
    if not re.match(r"^(inc|down\d|up\d|outc)\.", k):
        continue                                            # basis network etc.
    k = re.sub(r"^down(\d)\.maxpool_conv\.1\.double_conv\.", r"d\1.net.1.net.", k)
    k = re.sub(r"^up(\d)\.conv\.double_conv\.", r"u\1.conv.net.", k)
    k = re.sub(r"^up(\d)\.up\.", r"u\1.up.", k)
    k = re.sub(r"^inc\.double_conv\.", "inc.net.", k)
    k = re.sub(r"^outc\.conv\.", "outc.", k)
    out[k] = v
n_out = out["outc.weight"].shape[0]
in_ch = out["inc.net.0.weight"].shape[1]
net = UNet(in_ch, n_out)
missing, unexpected = net.load_state_dict(out, strict=False)
print(f"in={in_ch} out={n_out}  missing={missing}  unexpected={unexpected}")
torch.save({"model": net.state_dict(), "step": 0, "config": {"num_bins": in_ch, "K": n_out // 2}}, dst)
print("saved", dst)
