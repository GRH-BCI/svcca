from pprint import pprint as print
import numpy as np
import pycuda.autoinit
import pycuda
np.random.seed(1)

grid_cache = {}


# def make_redundant(ft, desired_shape, dim=-1):
#     h_desired, w_desired = desired_shape
#     h, w = ft.shape
#     out = pycuda.gpuarray.empty(desired_shape, dtype=np.complex64)
#     out[:h, :w] = ft
#     if desired_shape not in grid_cache:
#         X, Y = np.meshgrid(np.arange(h_desired), np.arange(w, w_desired))
#         __import__('ipdb').set_trace()
#         grid_cache[desired_shape] = (pycuda.gpuarray.to_gpu(X), pycuda.gpuarray.to_gpu(Y))

#     I, J = grid_cache[desired_shape]
#     for i in range(h_desired):
#         for j in range(ft.shape[1], w_desired):
#             out[i, j] = out[-i, -j].conj()
#     return out

reference = np.around(np.fft.fft2(array))
__import__('ipdb').set_trace()


# I, J = np.meshgrid(-np.arange(N), -np.arange(N), indexing='ij')
# print(np.around(ft))

# ft_gpu = pycuda.gpuarray.empty((N, N // 2 + 1), dtype=np.complex64)
# plan = fft.Plan((N, N), np.float32, np.complex64)
# fft.fft(array_gpu, ft_gpu, plan)
