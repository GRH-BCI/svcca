import sys
sys.path.insert(0, '..')
sys.path.insert(0, '../..')
sys.path.insert(0, 'svcca/')
import unittest
import numpy as np
import torch
from os.path import join, dirname
from dft_ccas import fft_resize
from dft_ccas_torch import fft_resize_torch
import utils
import pycuda


class TestCCA(unittest.TestCase):

    def test_fft_resize_torch(self):
        array1 = np.load(join(dirname(__file__), 'array1.npy'))
        resized_np = fft_resize(array1, resize=True, new_size=(10, 10))

        pycuda.driver.init()
        device = pycuda.driver.Device(0)
        ctx = device.make_context()
        tensor1 = torch.tensor(array1).cuda()
        resized_torch = fft_resize_torch(tensor1, resize=True, new_size=(10, 10))

        ctx.pop()
        self.assertTrue(np.allclose(resized_np, resized_torch.get(), 1e-2))     # more accuracy is not available it seems


if __name__ == '__main__':
    unittest.main()
