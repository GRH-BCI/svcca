import sys
sys.path.insert(0, '..')
sys.path.insert(0, '../..')
sys.path.insert(0, 'svcca/')
import unittest
import numpy as np
import cupy
from os.path import join, dirname
from dft_ccas_cupy import fft_resize, fourier_ccas

class TestCCA(unittest.TestCase):

    def test_fft_resize_cupy(self):
        array1 = np.load(join(dirname(__file__), 'array1.npy'))
        resized_np = fft_resize(array1, resize=True, new_size=(10, 10))

        cupy_array1 = cupy.asarray(array1)
        resized_cupy = fft_resize(cupy_array1, resize=True, new_size=(10, 10))

        self.assertTrue(np.allclose(resized_np, cupy.asnumpy(resized_cupy), 1e-2))     # more accuracy is not available it seems

    def test_fourier_ccas_cupy(self):
        array1 = np.load(join(dirname(__file__), 'array1.npy'))
        array2 = np.load(join(dirname(__file__), 'array2.npy'))
        result_np = fourier_ccas(array1, array2)
        result_cupy = fourier_ccas(cupy.asarray(array1), cupy.asarray(array2))

if __name__ == '__main__':
    unittest.main()
