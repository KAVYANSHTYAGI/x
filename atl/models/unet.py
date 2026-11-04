"""Networks.  ``UNet`` reproduces MotionPriorCMax's DSEC model (milesial-style 5-level U-Net, 64..1024 channels,
transposed-conv upsampling, BatchNorm) so that loss comparisons are architecture-matched.  It maps a voxel grid
(B, num_bins, H, W) to a coefficient grid (B, 2K, H, W) consumed by :class:`atl.motion.PixelTrajectories`."""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class DoubleConv(nn.Module):
    """(3x3 conv -> BN -> ReLU) x 2."""
    def __init__(self, cin, cout, cmid=None):
        super().__init__()
        cmid = cmid or cout
        self.net = nn.Sequential(nn.Conv2d(cin, cmid, 3, padding=1, bias=False), nn.BatchNorm2d(cmid),
                                 nn.ReLU(inplace=True), nn.Conv2d(cmid, cout, 3, padding=1, bias=False),
                                 nn.BatchNorm2d(cout), nn.ReLU(inplace=True))

    def forward(self, x):
        return self.net(x)


class Down(nn.Module):
    """2x max-pool followed by :class:`DoubleConv`."""
    def __init__(self, cin, cout):
        super().__init__()
        self.net = nn.Sequential(nn.MaxPool2d(2), DoubleConv(cin, cout))

    def forward(self, x):
        return self.net(x)


class Up(nn.Module):
    """2x transposed-conv upsampling, skip concatenation, :class:`DoubleConv`."""
    def __init__(self, cin, cout):
        super().__init__()
        self.up = nn.ConvTranspose2d(cin, cin // 2, 2, stride=2)
        self.conv = DoubleConv(cin, cout)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        dy, dx = x2.size(2) - x1.size(2), x2.size(3) - x1.size(3)
        x1 = F.pad(x1, [dx // 2, dx - dx // 2, dy // 2, dy - dy // 2])
        return self.conv(torch.cat([x2, x1], 1))


class UNet(nn.Module):
    """5-level U-Net (base..16*base channels): voxel grid (B, in_ch, H, W) -> coefficient grid (B, out_ch, H, W)."""
    def __init__(self, in_ch: int = 15, out_ch: int = 2, base: int = 64):
        super().__init__()
        c = [base, base * 2, base * 4, base * 8, base * 16]
        self.inc = DoubleConv(in_ch, c[0])
        self.d1, self.d2, self.d3, self.d4 = Down(c[0], c[1]), Down(c[1], c[2]), Down(c[2], c[3]), Down(c[3], c[4])
        self.u1, self.u2, self.u3, self.u4 = Up(c[4], c[3]), Up(c[3], c[2]), Up(c[2], c[1]), Up(c[1], c[0])
        self.outc = nn.Conv2d(c[0], out_ch, 1)
        for m in self.modules():                       # MotionPriorCMax utils.initialize_weights (kaiming)
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        x1 = self.inc(x); x2 = self.d1(x1); x3 = self.d2(x2); x4 = self.d3(x3); x5 = self.d4(x4)
        x = self.u1(x5, x4); x = self.u2(x, x3); x = self.u3(x, x2); x = self.u4(x, x1)
        return self.outc(x)
