'''This module defines a class which contains reimplementations of some operations defined
differently in numpy, cupy and PyTorch. It can be used to dispatch dynamically to the different
frameworks depending on what kind of input data is passed.'''

import numpy
import warnings
has_torch = False
has_cupy = False

try:
    import cupy
    has_cupy = True
except ImportError:
    pass

try:
    import torch
    has_torch = True
except ImportError:
    pass


class NumpyOverloads:
    @staticmethod
    def transpose(array):
        return array.T

    @staticmethod
    def conj(array):
        return array.conj()

    @staticmethod
    def sum(array, axis=0, keepdims=False):
        return array.sum(axis=axis, keepdims=keepdims)

    @staticmethod
    def add_normal(array, multiplier):
        return array + numpy.random.normal(size=array.shape) * multiplier

    @staticmethod
    def svd(array, full_matrices=True, compute_uv=True):
        return numpy.linalg.svd(array, full_matrices=full_matrices, compute_uv=compute_uv)

    @staticmethod
    def fft2(array, axes):
        return numpy.fft.fft2(array.astype('complex64'), axes=axes)

    @staticmethod
    def ifft2(*args):
        return numpy.fft.ifft2(*args)

    @staticmethod
    def fftfreq(*args):
        return numpy.fft.fftfreq(*args)


class CupyOverloads:
    @staticmethod
    def transpose(array):
        return array.T

    @staticmethod
    def conj(array):
        return array.conj()

    @staticmethod
    def sum(array, axis=0, keepdims=False):
        return array.sum(axis=axis, keepdims=keepdims)

    @staticmethod
    def add_normal(array, multiplier):
        return array + cupy.random.normal(size=array.shape) * multiplier

    @staticmethod
    def svd(array, full_matrices=True, compute_uv=True):
        return cupy.linalg.svd(array, full_matrices=full_matrices, compute_uv=compute_uv)

    @staticmethod
    def fft2(array, axes):
        return cupy.fft.fft2(array.astype('complex64'), axes=axes)

    @staticmethod
    def ifft2(*args):
        cupy.fft.ifft2(*args)

    @staticmethod
    def fftfreq(*args):
        return cupy.fft.fftfreq(*args)

    @staticmethod
    def flatnonzero(*args):
        return cupy.flatnonzero(*args)

    @staticmethod
    def mean(*args, **kwargs):
        axis = kwargs.get('axis', None)
        keepdims = kwargs.get('keepdims', False)
        return cupy.mean(*args, axis=axis, keepdims=keepdims)

    @staticmethod
    def place(array, mask, vals):
        n = mask.sum()
        vals = vals.flatten()
        if len(vals) < n:
            reps = cupy.ceil(n / len(vals))
            vals = cupy.repeat(vals, int(reps), axis=0)
            arr[mask] = vals[:n]

    @staticmethod
    def cov(m, y=None):
        return cupy.cov(m, y)

    @staticmethod
    def dot(a, b, out=None):
        return cupy.dot(a, b, out=out)

    @staticmethod
    def eigh(array):
        w, v = cupy.linalg.eigh(array)
        return w, v

    @staticmethod
    def pinv(array):
        return cupy.linalg.pinv(array)


class PyTorchOverloads:
    @staticmethod
    def transpose(array):
        return array.T()

    @staticmethod
    def conj(array):
        return array

    @staticmethod
    def sum(array, axis=0, keepdims=False):
        return array.sum(dim=axis, keepdim=keepdims)

    @staticmethod
    def add_normal(array, multiplier):
        return array + torch.randn_like(array) * multiplier

    @staticmethod
    def svd(array, full_matrices=True, compute_uv=True):
        return torch.svd(array, some=not full_matrices, compute_uv=compute_uv)

    @staticmethod
    def fft2(array, axes):
        raise ValueError('Complex Fourier does not work with PyTorch')

    @staticmethod
    def ifft2(*args):
        raise ValueError('Complex Inverse Fourier does not work with PyTorch')

    @staticmethod
    def fftfreq(*args):
        raise ValueError('FFTfreq does not work with PyTorch')

    @staticmethod
    def flatnonzero(*args):
        raise ValueError('Flatnonzero does not work with PyTorch')

    @staticmethod
    def mean(*args, **kwargs):
        axis = kwargs.get('axis', None)
        keepdims = kwargs.get('keepdims', False)
        if axis is None:
            return torch.mean(*args)
        else:
            return torch.mean(*args, axis, keepdim=keepdims)

    @staticmethod
    def place(array, mask, vals):
        n = mask.sum()
        vals = vals.flatten()
        if vals.numel() < n:
            reps = torch.ceil(n / vals.numel())
            vals = torch.repeat(vals, reps)
            arr[mask] = vals[:n]

    @staticmethod
    def cov(m, y=None):
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

    @staticmethod
    def dot(a, b, out=None):
        if a.ndimension() < 1 or b.ndimension() < 1:
            raise ValueError('Torch matmul does not work with scalars.')
        if a.ndimension() > 2 and b.ndimension() > 2:
            raise ValueError('Torch matmul with multidimensional '
                             'matrices currently unsupported.')
        return torch.matmul(a.to(dtype=torch.float32), b.to(dtype=torch.float32), out=out)

    @staticmethod
    def eigh(array):
        w, v = torch.symeig(array, eigenvectors=True, upper=False)
        return w, v

    @staticmethod
    def pinv(array):
        return torch.pinverse(array)


class Linalg(object):
    '''Dear reader,
    You may be wondering why the implementations of this class's methods's are a bit wonky. Like,
    why there's always three cases in the ifs, despite the fact that they have the same content for
    Numpy an Cupy. The reason is that I always need to check the Numpy case first, because numpy can
    always be imported. I do not want to risk accidentally evaluating expressions involving Cupy
    (not sure if `or` shortcuts in Python), before being certain that I am not dealing with a Numpy
    array and Cupy or PyTorch are not installed.'''

    def __init__(self):
        self._overloads = {
            'numpy': NumpyOverloads,
            'cupy': CupyOverloads,
            'pytorch': PyTorchOverloads,
        }
        self._backend = 'numpy'

    def use_backend(self, backend):
        assert backend in ('numpy', 'pytorch', 'cupy')
        assert has_torch or backend != 'pytorch'
        assert has_cupy or backend != 'cupy'
        self._backend = backend

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
        def wrapped(*args, **kwargs):
            if hasattr(self._overloads[self._backend], name):
                return getattr(self._overloads[self._backend], name)(*args, **kwargs)

            method_getter = {
                'pytorch': Linalg.get_torch,
                'cupy': Linalg.get_cupy,
                'numpy': Linalg.get_numpy,
            }[self._backend]

            if self._backend == 'pytorch':
                try:
                    return method_getter(name)(*args,
                                               **kwargs,
                                               device=torch.cuda.current_device())
                except TypeError:
                    return method_getter(name)(*args, **kwargs)
            else:
                return method_getter(name)(*args, **kwargs)


        return wrapped


import sys
# make class instance act like a module
sys.modules[__name__] = Linalg()
