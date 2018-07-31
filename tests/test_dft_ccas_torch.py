import sys
sys.path.insert(0, '..')
sys.path.insert(0, '../..')
sys.path.insert(0, 'svcca/')
import unittest
import numpy as np
import torch
from os.path import join, dirname
from dft_ccas import fft_resize, fourier_ccas
from dft_ccas_torch import fft_resize_torch, fourier_ccas_pycuda
import utils
import pycuda
pycuda.driver.init()
device = pycuda.driver.Device(0)


class TestCCA(unittest.TestCase):

    def test_fft_resize_torch(self):
        array1 = np.load(join(dirname(__file__), 'array1.npy'))
        resized_np = fft_resize(array1, resize=True, new_size=(10, 10))

        ctx = device.make_context()
        tensor1 = torch.tensor(array1).cuda()
        resized_torch = fft_resize_torch(tensor1, resize=True, new_size=(10, 10), context=ctx)

        self.assertTrue(np.allclose(resized_np, resized_torch.get(), 1e-2))     # more accuracy is not available it seems
        ctx.pop()

    def test_fourier_ccas_pycuda(self):
        ctx = device.make_context()
        array1 = np.load(join(dirname(__file__), 'array1.npy'))
        array2 = np.load(join(dirname(__file__), 'array2.npy'))
        # result_np = fourier_ccas(array1, array2)
        result_pycuda = fourier_ccas_pycuda(torch.tensor(array1).cuda(), torch.tensor(array2).cuda())

if __name__ == '__main__':
    unittest.main()
