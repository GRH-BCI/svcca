'''Reimplementation of stuff from numpy which is not available in torch.

.. moduleauthor:: Rasmus Diederichsen
'''
import torch
import numpy as np


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


def torch_dtype_to_numpy(dtype):
    '''Convert a torch ``dtype`` to an equivalent numpy ``dtype``, if it is also available in
    pycuda.

    Parameters
    ----------
    dtype   :   np.dtype

    Returns
    -------
    torch.dtype

    Raises
    ------
    ValueError
        If there is not PyTorch equivalent, or the equivalent would not work with pycuda
    '''
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


def numpy_dtype_to_torch(dtype):
    '''Convert numpy ``dtype`` to torch ``dtype``. The first matching one will be returned, if there
    are synonyms.

    Parameters
    ----------
    dtype   :   torch.dtype

    Returns
    -------
    np.dtype
    '''
    for dtype_t, dtype_n in dtype_map.items():
        if dtype_n == dtype_t:
            return dtype_t


import cupy
import torch.utils.dlpack


def tensor_to_cupy(t):
    '''Convert a PyTorch :class:`~torch.Tensor` object to a :mod:`cupy` array.
    Returns
    -------
    cupy.ndarray
    '''
    return cupy.fromDlpack(torch.utils.dlpack.to_dlpack(t))


def cupy_place(arr, mask, vals):
    n = mask.sum()
    if len(vals) < n:
        reps = cupy.ceil(n / len(vals))
        vals = cupy.repeat(vals, reps, axis=0)
    arr[mask] = vals[:n]
