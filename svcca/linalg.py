import numpy, cupy, torch


class Linalg(object):

    def __init__(self):
        self._method_getter = None
        self.overloads = {
            'eigh': Linalg.eigh,
            'place': Linalg.place,
            'cov': Linalg.cov,
            'dot': Linalg.dot,
            'asarray': Linalg.asarray,
            'svd': Linalg.svd,
            'pinv': Linalg.pinv,
            'conj': Linalg.conj,
            'transpose': Linalg.transpose,
            'mean': Linalg.mean,
            'fft2': Linalg.fft2,
            'ifft2': Linalg.ifft2,
            'fftfreq': Linalg.fftfreq,
            'flatnonzero': Linalg.flatnonzero,
        }

    @staticmethod
    def asarray(arg, module):
        if module == 'numpy':
            return numpy.asarray(arg)
        elif module == 'cupy':
            return cupy.asarray(arg)
        else:
            return torch.Tensor(arg, device=torch.cuda.current_device())

    @staticmethod
    def transpose(arg):
        if isinstance(arg, numpy.ndarray) or isinstance(arg, cupy.ndarray):
            return arg.T
        else:
            return arg.t()

    @staticmethod
    def conj(arr):
        if isinstance(arr, torch.Tensor):
            return arr
        else:
            return arr.conj()

    @staticmethod
    def svd(arr, full_matrices=True, compute_uv=True):
        if isinstance(arr, torch.Tensor):
            return torch.svd(arr, some=not full_matrices, compute_uv=compute_uv)
        elif isinstance(arr, cupy.ndarray):
            return cupy.linalg.svd(arr, full_matrices=full_matrices, compute_uv=compute_uv)
        else:
            return numpy.linalg.svd(arr, full_matrices=full_matrices, compute_uv=compute_uv)

    @staticmethod
    def fft2(arr, axes):
        if isinstance(arr, torch.Tensor):
            raise ValueError('Complex Fourier does not work with PyTorch')
        elif isinstance(arr, cupy.ndarray):
            return cupy.fft.fft2(arr.astype('complex64'), axes=axes)
        else:
            return numpy.fft.fft2(arr.astype('complex64'), axes=axes)

    @staticmethod
    def ifft2(*args):
        if isinstance(args[0], torch.Tensor):
            raise ValueError('Complex Inverse Fourier does not work with PyTorch')
        elif isinstance(args[0], cupy.ndarray):
            return cupy.fft.ifft2(*args)
        else:
            return numpy.fft.ifft2(*args)

    @staticmethod
    def fftfreq(*args):
        if isinstance(args[0], torch.Tensor):
            raise ValueError('FFTfreq does not work with PyTorch')
        elif isinstance(args[0], cupy.ndarray):
            return cupy.fft.fftfreq(*args)
        else:
            return numpy.fft.fftfreq(*args)

    @staticmethod
    def flatnonzero(*args):
        if isinstance(args[0], torch.Tensor):
            raise ValueError('Flatnonzero does not work with PyTorch')
        elif isinstance(args[0], cupy.ndarray):
            return cupy.flatnonzero(*args)
        else:
            return numpy.flatnonzero(*args)

    @staticmethod
    def mean(*args, **kwargs):
        axis = kwargs.get('axis', None)
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
        if isinstance(arr, cupy.ndarray):
            return Linalg.cupy_place(arr, mask, vals)
        elif isinstance(arr, torch.Tensor):
            return Linalg.torch_place(arr, mask, vals)
        else:
            return numpy.place(arr, mask, vals)

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

        if isinstance(m, torch.Tensor):
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
        elif isinstance(m, numpy.ndarray):
            return numpy.cov(m, y)
        else:
            return cupy.cov(m, y)



    @staticmethod
    def dot(a, b, out=None):
        if isinstance(a, torch.Tensor):
            if a.ndimension() < 1 or b.ndimension() < 1:
                raise ValueError('Torch matmul does not work with scalars.')
            if a.ndimension() > 2 and b.ndimension() > 2:
                raise ValueError('Torch matmul with multidimensional matrices currently unsupported.')
            return torch.matmul(a.to(dtype=torch.float32), b.to(dtype=torch.float32), out=out)
        elif isinstance(a, cupy.ndarray):
            return cupy.dot(a, b, out=out)
        else:
            return numpy.dot(a, b, out=out)

    @staticmethod
    def eigh(array):
        if isinstance(array, torch.Tensor):
            w, v = torch.symeig(array, eigenvectors=True, upper=False)
        elif isinstance(array, cupy.ndarray):
            w, v = cupy.linalg.eigh(array)
        else:
            w, v = numpy.linalg.eigh(array)
        return w, v

    @staticmethod
    def pinv(array):
        if isinstance(array, torch.Tensor):
            return torch.pinverse(array)
        elif isinstance(array, cupy.ndarray):
            return cupy.linalg.pinv(array)
        else:
            return numpy.linalg.pinv(array)

    @staticmethod
    def method_exists(name):
        exists = [False, False, False]
        if hasattr(torch, name):
            exists[0] = True
        if hasattr(numpy, name) or hasattr(numpy.linalg, name):
            exists[1] = True
        if hasattr(cupy, name) or hasattr(cupy.linalg, name):
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
        if hasattr(cupy, name):
            return getattr(cupy, name)
        else:
            return getattr(cupy.linalg, name)

    @staticmethod
    def get_torch(name):
        if hasattr(torch, name):
            return getattr(torch, name)


    def __getattr__(self, name):

        if not Linalg.method_exists(name):
            raise ValueError(f'`{name}()` is not available in all frameworks')

        def wrapped(*args, **kwargs):
            if name in self.overloads:
                return self.overloads[name](*args, **kwargs)
            elif isinstance(args[0], torch.Tensor):
                self._method_getter = Linalg.get_torch
            elif isinstance(args[0], cupy.ndarray):
                self._method_getter = Linalg.get_cupy
            elif isinstance(args[0], numpy.ndarray):
                self._method_getter = Linalg.get_numpy
            try:
                return self._method_getter(name)(*args, **kwargs, device=torch.cuda.current_device())
            except TypeError:
                return self._method_getter(name)(*args, **kwargs)

        return wrapped

import sys
sys.modules[__name__] = Linalg()
