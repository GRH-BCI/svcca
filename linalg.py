import numpy, cupy, torch


# class ProxyTensor(object):

#     def __init__(self, arg):
#         self._init_arg = arg

#     def __add__(self, other):
#         if isinstance(other, numpy.ndarray):
#             return numpy.asarray(self._init_arg) + other
#         if isinstance(other, numpy.ndarray):
#             return numpy.asarray(self._init_arg) + other


class Linalg(object):

    def __init__(self):
        self._method_getter = None
        self.overloads = {
            'eigh': Linalg.eigh,
            'place': Linalg.place,
            'cov': Linalg.cov,
            'dot': Linalg.dot,
            'asarray': Linalg.asarray,
            'pinv': Linalg.pinv,
            'conj': Linalg.conj,
            'transpose': Linalg.transpose,
            'mean': Linalg.mean,
        }

    @staticmethod
    def asarray(arg, module):
        if module == 'numpy':
            return numpy.asarray(arg)
        elif module == 'cupy':
            return cupy.asarray(arg)
        else:
            return torch.Tensor(arg)

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
        if len(vals) < n:
            reps = cupy.ceil(n / len(vals))
            vals = cupy.repeat(vals, reps, axis=0)
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
            return torch.matmul(a, b, out=out)
        elif isinstance(a, cupy.ndarray):
            return cupy.dot(a, b, out=out)
        else:
            return numpy.dot(a, b, out=out)

    @staticmethod
    def eigh(array):
        if isinstance(array, torch.Tensor):
            w, v = torch.eig(array, eigenvectors=True)
            w = w[:, 0].sort()[0]
            return w, v
        elif isinstance(array, cupy.ndarray):
            return cupy.linalg.eigh(array)
        else:
            return numpy.linalg.eigh(array)

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
            return self._method_getter(name)(*args, **kwargs)

        return wrapped

import sys
sys.modules[__name__] = Linalg()
