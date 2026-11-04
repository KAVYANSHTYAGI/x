# Third-party notices

## MotionPriorCMax (MIT)

Source: https://github.com/tub-rip/MotionPriorCMax (commit `d871fd82def48182deec5849fc3a1212df1acdae`).

The following files re-implement or port parts of that code for an architecture- and loss-matched comparison:
* `atl/objectives/mpcm.py`: MotionPriorCMax loss, from `src/losses/focus.py` and `src/utils/loss.py`;
* `atl/models/unet.py`: U-Net, from `src/models/unet/`;
* `atl/motion/trajectories.py`: tile-trajectory layout and channel order, from `src/modules/trajectory_net.py` and
  `src/utils/trajectories.py`;
* `atl/data/dsec.py`: training splits, from `src/loader/dsec/loader.py`;
* `atl/data/evimo2.py`: data layout, from `src/loader/evimo2/datasubset.py`;
* `atl/models/bflow_adapter.py`: interface to `src/models/raft_spline/`.

The official repository is not vendored; `atl/models/bflow_adapter.py` imports it from a local clone when needed.

```
MIT License

Copyright (c) 2024 TU Berlin - Robotic Interactive Perception

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## Datasets

Each dataset is used under its own licence and is not redistributed here:
* DSEC (Gehrig et al., RA-L 2021): https://dsec.ifi.uzh.ch
* EVIMO2 (Burner et al.)
