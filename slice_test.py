import utils, numpy as np, pycuda.autoinit, pycuda

array = pycuda.gpuarray.to_gpu(np.array([1,1,0,1,0]))

from pycuda.elementwise import ElementwiseKernel
n_nonzero = pycuda.gpuarray.sum(array)
nonzero = pycuda.gpuarray.zeros((n_nonzero.get(),), dtype=array.dtype)
