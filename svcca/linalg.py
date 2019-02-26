'''This module defines a class which contains reimplementations of some operations defined
differently in numpy, cupy and PyTorch. It can be used to dispatch dynamically to the different
frameworks depending on what kind of input data is passed.'''

import numpy
import warnings
has_torch = False
has_cupy = False

try:
    import cupy
    has_torch = True
except ImportError:
    warnings.warn('Cupy not found.')

try:
    import torch
    has_torch = True
except ImportError:
    warnings.warn('PyTorch not found.')


class Linalg(object):
    '''Dear reader,
    You may be wondering why the implementations of this class's methods's are a bit wonky. Like,
    why there's always three cases in the ifs, despite the fact that they have the same content for
    Numpy an Cupy. The reason is that I always need to check the Numpy case first, because numpy can
    always be imported. I do not want to risk accidentally evaluating expressions involving Cupy
    (not sure if `or` shortcuts in Python), before being certain that I am not dealing with a Numpy
    array and Cupy or PyTorch are not installed.'''

    def __init__(self):
        self._method_getter = None
        self.overloads = {
            'eigh': Linalg.eigh,
            'place': Linalg.place,
            'cov': Linalg.cov,
            'dot': Linalg.dot,
            'svd': Linalg.svd,
            'pinv': Linalg.pinv,
            'conj': Linalg.conj,
            'transpose': Linalg.transpose,
            'mean': Linalg.mean,
            'fft2': Linalg.fft2,
            'ifft2': Linalg.ifft2,
            'fftfreq': Linalg.fftfreq,
            'flatnonzero': Linalg.flatnonzero,
            'add_normal': Linalg.add_normal,
            'sum': Linalg.sum
        }

    @staticmethod
    def transpose(arg):
        '''Plain matrix transposition.'''
        if isinstance(arg, numpy.ndarray):
            return arg.T
        elif isinstance(arg, cupy.ndarray):
            return arg.T
        else:
            return arg.t()

    @staticmethod
    def conj(arr):
        '''(Complex) conjugation'''
        if isinstance(arr, numpy.ndarray):
            return arr.conj()
        elif isinstance(arr, torch.Tensor):
            return arr
        else:
            return arr.conj()

    @staticmethod
    def sum(array, axis=0, keepdims=False):
        '''Summation over dimennsions.'''
        if isinstance(array, numpy.ndarray):
            return numpy.sum(array, axis=axis, keepdims=keepdims)
        elif isinstance(array, torch.Tensor):
            return array.sum(dim=axis, keepdim=keepdims)
        else:
            return cupy.sum(array, axis=axis, keepdims=keepdims)

    @staticmethod
    def add_normal(array, multiplier):
        '''Add normal-distributed noise to some array'''
        if isinstance(array, numpy.ndarray):
            return array + numpy.random.normal(size=array.shape) * multiplier
        elif isinstance(array, torch.Tensor):
            return array + torch.randn_like(array) * multiplier
        else:
            return cupy + cupy.random.normal(size=array.shape) * multiplier

    @staticmethod
    def svd(arr, full_matrices=True, compute_uv=True):
        if isinstance(arr, numpy.ndarray):
            return numpy.linalg.svd(arr, full_matrices=full_matrices, compute_uv=compute_uv)
        elif isinstance(arr, torch.Tensor):
            return torch.svd(arr, some=not full_matrices, compute_uv=compute_uv)
        else:
            return cupy.linalg.svd(arr, full_matrices=full_matrices, compute_uv=compute_uv)

    @staticmethod
    def fft2(arr, axes):
        if isinstance(arr, numpy.ndarray):
            return numpy.fft.fft2(arr.astype('complex64'), axes=axes)
        elif isinstance(arr, torch.Tensor):
            raise ValueError('Complex Fourier does not work with PyTorch')
        else:
            return cupy.fft.fft2(arr.astype('complex64'), axes=axes)

    @staticmethod
    def ifft2(*args):
        if isinstance(args[0], numpy.ndarray):
            return numpy.fft.ifft2(*args)
        elif isinstance(args[0], torch.Tensor):
            raise ValueError('Complex Inverse Fourier does not work with PyTorch')
        else:
            return cupy.fft.ifft2(*args)

    @staticmethod
    def fftfreq(*args):
        if isinstance(args[0], numpy.ndarray):
            return numpy.fft.fftfreq(*args)
        elif isinstance(args[0], torch.Tensor):
            raise ValueError('FFTfreq does not work with PyTorch')
        else:
            return cupy.fft.fftfreq(*args)

    @staticmethod
    def flatnonzero(*args):
        if isinstance(args[0], numpy.ndarray):
            return numpy.flatnonzero(*args)
        elif isinstance(args[0], torch.Tensor):
            raise ValueError('Flatnonzero does not work with PyTorch')
        else:
            return cupy.flatnonzero(*args)

    @staticmethod
    def mean(*args, **kwargs):
        axis     = kwargs.get('axis', None)
        keepdims = kwargs.get('keepdims', False)
        if isinstance(args[0], numpy.ndarray):
            return numpy.mean(*args, axis=axis, keepdims=keepdims)
        elif isinstance(args[0], cupy.ndarray):
            return cupy.mean(*args, axis=axis, keepdims=keepdims)
        else:
            if axis is None:
                return torch.mean(*args)
            else:
                return torch.mean(*args, axis, keepdim=keepdims)

    @staticmethod
    def place(arr, mask, vals):
        if isinstance(arr, numpy.ndarray):
            return numpy.place(arr, mask, vals)
        elif isinstance(arr, cupy.ndarray):
            return Linalg.cupy_place(arr, mask, vals)
        else:
            return Linalg.torch_place(arr, mask, vals)

    @staticmethod
    def cupy_place(arr, mask, vals):
        n = mask.sum()
        vals = vals.flatten()
        if len(vals) < n:
            reps = cupy.ceil(n / len(vals))
            vals = cupy.repeat(vals, int(reps), axis=0)
            arr[mask] = vals[:n]

    @staticmethod
    def torch_place(arr, mask, vals):
        n = mask.sum()
        vals = vals.flatten()
        if vals.numel() < n:
            reps = torch.ceil(n / vals.numel())
            vals = torch.repeat(vals, reps)
            arr[mask] = vals[:n]

    @staticmethod
    def cov(m, y=None):

        if isinstance(m, numpy.ndarray):
            return numpy.cov(m, y)
        elif isinstance(m, torch.Tensor):
            if m.ndimension() > 2:
                raise ValueError("m has more than 2 dimensions")

            if y is not None and y.ndimension() > 2:
                raise ValueError('y has more than 2 dimensions')

            X = m
            if X.shape[0] == 0:
                return torch.tensor([], device=torch.cuda.current_device()).reshape(0, 0)
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
            c = Linalg.dot(X, X_T)
            c *= 1. / fact
            return c.squeeze()
        else:
            return cupy.cov(m, y)

    @staticmethod
    def dot(a, b, out=None):
        if isinstance(a, numpy.ndarray):
            return numpy.dot(a, b, out=out)
        elif isinstance(a, torch.Tensor):
            if a.ndimension() < 1 or b.ndimension() < 1:
                raise ValueError('Torch matmul does not work with scalars.')
            if a.ndimension() > 2 and b.ndimension() > 2:
                raise ValueError('Torch matmul with multidimensional matrices currently unsupported.')
            return torch.matmul(a.to(dtype=torch.float32), b.to(dtype=torch.float32), out=out)
        else:
            return cupy.dot(a, b, out=out)

    @staticmethod
    def eigh(array):
        if isinstance(arry, numpy.ndarray):
            w, v = numpy.linalg.eigh(array)
        if isinstance(array, torch.Tensor):
            w, v = torch.symeig(array, eigenvectors=True, upper=False)
        elif isinstance(array, cupy.ndarray):
            w, v = cupy.linalg.eigh(array)
        return w, v

    @staticmethod
    def pinv(array):
        if isinstance(a, numpy.ndarray):
            return numpy.linalg.pinv(array)
        elif isinstance(array, torch.Tensor):
            return torch.pinverse(array)
        else:
            return cupy.linalg.pinv(array)

    @staticmethod
    def method_exists(name):
        exists = [False, False, False]
        if has_torch and hasattr(torch, name):
            exists[0] = True
        if hasattr(numpy, name) or hasattr(numpy.linalg, name):
            exists[1] = True
        if has_cupy and (hasattr(cupy, name) or hasattr(cupy.linalg, name)):
            exists[2] = True

        return all(exists)

    @staticmethod
    def get_numpy(name):
        if hasattr(numpy, name):
            return getattr(numpy, name)
        else:
            return getattr(numpy.linalg, name)

    @staticmethod
    def get_cupy(name):
        if not has_cupy:
            raise ValueError('Cupy not loaded. Probably because it isn\'t installed.')
        if hasattr(cupy, name):
            return getattr(cupy, name)
        else:
            return getattr(cupy.linalg, name)

    @staticmethod
    def get_torch(name):
        if not has_torch:
            raise ValueError('PyTorch not loaded. Probably because it isn\'t installed.')
        if hasattr(torch, name):
            return getattr(torch, name)

    def __getattr__(self, name):

        if not Linalg.method_exists(name):
            raise ValueError(f'`{name}()` is not available in all frameworks')

        def wrapped(*args, **kwargs):
            if name in self.overloads:
                return self.overloads[name](*args, **kwargs)
            elif isinstance(args[0], numpy.ndarray):
                self._method_getter = Linalg.get_numpy
            elif isinstance(args[0], torch.Tensor):
                self._method_getter = Linalg.get_torch
            elif isinstance(args[0], cupy.ndarray):
                self._method_getter = Linalg.get_cupy
            try:
                # TODO: Check if TypeError is still thrown if torch is not imported
                return self._method_getter(name)(*args,
                                                 **kwargs,
                                                 device=torch.cuda.current_device())
            except TypeError:
                return self._method_getter(name)(*args, **kwargs)

        return wrapped


import sys
# make class instance act like a module
sys.modules[__name__] = Linalg()
