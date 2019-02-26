import numpy as np
import cupy
import time
np.random.seed(0)

k = 1000
array = np.random.rand(k, k) + np.random.rand(k, k) * 1j
array_gpu = cupy.asarray(array)

time_cpu = 0

n = 20
for i in range(n):
    t0 = time.time()
    np.linalg.svd(cupy.asnumpy(array_gpu))
    t1 = time.time()
    time_cpu += (t1 - t0)

time_gpu = 0
for i in range(n):
    t0 = time.time()
    cupy.linalg.svd(array_gpu)
    t1 = time.time()
    time_gpu += (t1 - t0)

print('CPU: ', time_cpu / n)
print('GPU: ', time_gpu / n)
print(f'ratio: {time_gpu / time_cpu}')
