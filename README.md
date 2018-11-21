# About this fork

This is a Python 3 port of the original SVCCA code onto the GPU. This is
realized with the help of [Cupy](https://github.com/cupy/cupy/)]. Conversion
functions for [PyTorch](https://github.com/pytorch/pytorch) tensors are
provided.

---

It should be noted that this code will **not** run faster than the CPU
implementation, since the runtime is dominated by calls to SVD. The `gesvd`
routine used by Cupy is slower (but more precise) than `gsedd` (which is e.g. in
MAGMA), but I doubt that would help much.

SVD with Cupy is about 40 times slower than with Numpy.
