'''Reimplementation of stuff from numpy which is not available in torch.

.. moduleauthor:: Rasmus Diederichsen
'''
import torch
import numpy as np
import reikna
import reikna.cluda as cuda
from reikna.fft import FFT
from reikna.cluda import dtypes
from reikna.core import Annotation, Type, Transformation, Parameter
import pycuda.autoinit
from pycuda.driver import PointerHolderBase
import warnings
from pycuda.compyte.dtypes import dtype_to_ctype
from skcuda.misc import subtract, init as skcuinit
from skcuda.linalg import dot as sdot

skcuinit()


def fftfreq_torch(n, d=1.0):
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


def fftfreq_pycuda(n, d=1.0):
    if not np.issubdtype(type(n), np.integer):
        raise ValueError(f'n should be an integer (is {type(n)})')
    n = int(n)
    d = float(d)
    val = 1.0 / (n * d)
    results = pycuda.gpuarray.empty((n,), dtype=np.float32)
    N = (n - 1) // 2 + 1
    p1 = pycuda.gpuarray.arange(0, N, dtype=np.float32)
    results[:N] = p1
    p2 = pycuda.gpuarray.arange(-(n // 2), 0, dtype=np.float32)
    results[N:] = p2
    return results * val


def cat_pycuda(a, b):

    assert a.ndim == b.ndim == 2, 'Only 2d inputs supported for now.'
    assert a.shape[1] == b.shape[1], '2nd dimension must have same size.'

    rows_a = a.shape[0]
    rows_b = b.shape[0]
    cols = a.shape[1]
    ret = pycuda.gpuarray.empty((rows_a + rows_b, cols), dtype=a.dtype)
    ret[:rows_a, :] = a
    ret[rows_b:, :] = b

    return ret


def cov_pycuda(m, y=None, context=pycuda.autoinit.context):

    if m.ndim > 2:
        raise ValueError("m has more than 2 dimensions")

    if y.ndim > 2:
        raise ValueError('y has more than 2 dimensions')

    X = m
    if X.shape[0] == 0:
        return pycuda.gpuarray.empty([0], dtype=m.dtype)
    if y is not None:
        X = cat_pycuda(X, y)

    ddof = 1

    avg = pycuda_mean(X, axis=1, context=context)

    fact = X.shape[1] - ddof

    if fact <= 0:
        warnings.warn("Degrees of freedom <= 0 for slice",
                      RuntimeWarning, stacklevel=2)
        fact = 0.0

    X = subtract(X, avg[:, None])
    c = sdot(X, X, transb='T')
    c *= 1. / fact
    return c.squeeze()


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


class Holder(PointerHolderBase):

    def __init__(self, tensor):
        super().__init__()
        self.tensor = tensor
        self.gpudata = tensor.data_ptr()

    def get_pointer(self):
        return self.tensor.data_ptr()

    def __int__(self):
        return self.__index__()

    # without an __index__ method, arithmetic calls to the GPUArray backed by this pointer fail
    # not sure why, this needs to return some integer, apparently
    def __index__(self):
        return self.gpudata


# dict to map between torch and numpy dtypes
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


def tensor_to_gpuarray(tensor, context=pycuda.autoinit.context):
    '''Convert a :class:`torch.Tensor` to a :class:`pycuda.gpuarray.GPUArray`. The underlying
    storage will be shared, so that modifications to the array will reflect in the tensor object.

    Parameters
    ----------
    tensor  :   torch.Tensor

    Returns
    -------
    pycuda.gpuarray.GPUArray

    Raises
    ------
    ValueError
        If the ``tensor`` does not live on the gpu
    '''
    if not tensor.is_cuda:
        raise ValueError('Cannot convert CPU tensor to GPUArray (call `cuda()` on it)')
    else:
        thread = cuda.cuda_api().Thread(context)
        return reikna.cluda.cuda.Array(thread, tensor.shape,
                                       dtype=torch_dtype_to_numpy(tensor.dtype),
                                       base_data=Holder(tensor))


def gpuarray_to_tensor(gpuarray, context=pycuda.autoinit.context):
    '''Convert a :class:`pycuda.gpuarray.GPUArray` to a :class:`torch.Tensor`. The underlying
    storage will NOT be shared, since a new copy must be allocated.

    Parameters
    ----------
    gpuarray  :   pycuda.gpuarray.GPUArray

    Returns
    -------
    torch.Tensor
    '''
    shape = gpuarray.shape
    dtype = gpuarray.dtype
    out_dtype = numpy_dtype_to_torch(dtype)
    out = torch.zeros(shape, dtype=out_dtype).cuda()
    gpuarray_copy = tensor_to_gpuarray(out, context=context)
    byte_size = gpuarray.itemsize * gpuarray.size
    pycuda.driver.memcpy_dtod(gpuarray_copy.gpudata, gpuarray.gpudata, byte_size)
    return out


####################################################################################################
#  Taken from https://github.com/fjarri/reikna/blob/develop/examples/demo_real_to_complex_fft.py   #
####################################################################################################


def get_complex_trf(arr):
    complex_dtype = dtypes.complex_for(arr.dtype)
    return Transformation(
        [Parameter('output', Annotation(Type(complex_dtype, arr.shape), 'o')),
         Parameter('input', Annotation(arr, 'i'))],
         """
         ${output.store_same}(
             COMPLEX_CTR(${output.ctype})(
                 ${input.load_same},
                 0));
         """)


def fft2(array_gpu, axes=None, context=pycuda.autoinit.context):
    complex_transf = get_complex_trf(array_gpu)
    thread = cuda.cuda_api().Thread(context)

    fft = FFT(complex_transf.output, axes=axes)
    fft.parameter.input.connect(complex_transf, complex_transf.output,
                                new_input=complex_transf.input)
    cfft = fft.compile(thread)

    result_gpu = thread.array(array_gpu.shape, np.complex64)
    cfft(result_gpu, array_gpu)
    return result_gpu


def pycuda_mean(array_gpu, axis=None, context=pycuda.autoinit.context):
    # number of elements in the meaned dimensions
    n = array_gpu.size if not axis else array_gpu.shape[axis]
    reduction = reikna.algorithms.Reduce(array_gpu,
                                         reikna.algorithms.predicate_sum(array_gpu.dtype),
                                         axes=[axis])
    thread = cuda.cuda_api().Thread(context)
    creduction = reduction.compile(thread)
    result = thread.empty_like(reduction.parameter.output)
    creduction(result, array_gpu)
    return result / n
