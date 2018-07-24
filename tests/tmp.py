import pycuda.autoinit
from pycuda.gpuarray import GPUArray

import pycuda
import time
import numpy as np

N = 1
t0 = time.time()
for n in range(N):
    pycuda.gpuarray.to_gpu(np.fft.fftfreq(13))
print((time.time() - t0) / N)

t0 = time.time()
for n in range(N):
    np.fft.fftfreq(13)
print((time.time() - t0) / N)
