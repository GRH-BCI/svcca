'''Reimplementation of stuff from numpy which is not available in torch.

.. moduleauthor:: Rasmus Diederichsen
'''
import torch
import numpy as np


def fftfreq(n, d=1.0):
    if not np.issubdtype(type(n), np.integer) and type(n) not in [int, torch.uint8, torch.int8,
                                                                  torch.short, torch.int, torch.long]:
        raise ValueError(f'n should be an integer (is {type(n)})')
    n = int(n)
    d = float(d)
    val = 1.0 / (n * d)
    results = torch.empty(n, dtype=torch.float32)
    N = (n - 1) // 2 + 1
    p1 = torch.arange(0, N, dtype=torch.float32)
    results[:N] = p1
    p2 = torch.arange(-(n // 2), 0, dtype=torch.float32)
    results[N:] = p2
    return results * val


def cov(m, y=None):

    if m.ndimension() > 2:
        raise ValueError("m has more than 2 dimensions")

    if y.ndimension() > 2:
        raise ValueError('y has more than 2 dimensions')

    X = m
    if X.shape[0] == 0:
        return torch.tensor([]).reshape(0, 0)
    if y is not None:
        X = torch.cat((X, y), dim=0)

    ddof = 1

    avg = torch.mean(X, dim=1)

    fact = X.shape[1] - ddof

    if fact <= 0:
        import warnings
        warnings.warn("Degrees of freedom <= 0 for slice",
                      RuntimeWarning, stacklevel=2)
        fact = 0.0

    X -= avg[:, None]
    X_T = X.t()
    c = dot(X, X_T)
    c *= 1. / fact
    return c.squeeze()


def flatnonzero(tensor):
    return torch.squeeze(torch.nonzero(tensor.view(-1)))


def dot(a, b, out=None):
    if a.ndimension() < 1 or b.ndimension() < 1:
        raise ValueError('Torch matmul does not work with scalars.')
    if a.ndimension() > 2 and b.ndimension() > 2:
        raise ValueError('Torch matmul with multidimensional matrices currently unsupported.')
    return torch.matmul(a, b, out=out)

import pycuda.autoinit
from pycuda.gpuarray import GPUArray
from pycuda.driver import PointerHolderBase

class Holder(PointerHolderBase):

    def __init__(self, tensor):
        super().__init__()
        self.tensor = tensor
        self.gpudata = tensor.data_ptr()

    def get_pointer(self):
        return self.tensor.data_ptr()

    # without an __index__ method, arithmetic calls to the GPUArray backed by this pointer fail
    # not sure why, this needs to return some integer, apparently
    def __index__(self):
        return self.gpudata


def torch_dtype_to_numpy(dtype):
    dtype_map = {
        # signed integers
        torch.int8: np.int8,
        torch.int16: np.int16,
        torch.short: np.int16,
        torch.int32: np.int32,
        torch.int: np.int32,
        torch.int64: np.int64,
        torch.long: np.int64,

        # unsinged inters
        torch.uint8: np.uint8,

        # floating point
        torch.float: np.float32,
        torch.float32: np.float32,
        torch.float16: np.float16,
        torch.half: np.float16,
        torch.float64: np.float64,
        torch.double: np.float64
    }

    from pycuda.compyte.dtypes import dtype_to_ctype
    if dtype not in dtype_map:
        raise ValueError(f'{dtype} has no PyTorch equivalent')
    else:
        candidate = dtype_map[dtype]
        # we can raise exception early by checking of the type can be used with pycuda. Otherwise
        # we realize it only later when using the array
        try:
            _ = dtype_to_ctype(candidate)
        except ValueError:
            raise ValueError(f'{dtype} cannot be used in pycuda')
        else:
            return candidate


def tensor_to_gpuarray(tensor):
    if not tensor.is_cuda:
        raise ValueError('Cannot convert CPU tensor to GPUArray (call `cuda()` on it)')
    else:
        array = GPUArray(tensor.shape, dtype=torch_dtype_to_numpy(tensor.dtype),
                         gpudata=Holder(tensor))
        return array
