# Tests

Run with `pytest -q`. There are 20 tests:
* `test_ops.py`: kernels vs pure-torch references, values and gradients; runs on CUDA too when available;
* `test_objectives.py`: motion, objectives, properness, I/O.
