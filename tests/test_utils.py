import sys
sys.path.insert(0, '..')
sys.path.insert(0, '../..')
sys.path.insert(0, 'svcca/')
import unittest
import numpy as np
import utils
import torch


class TestUtils(unittest.TestCase):

    def test_fftfreq(self):
        test_ints = sorted([10, 212312, 2, 23523, 1, 35456, 45, 654, 945, 2312313])
        for i in test_ints:
            res_np = np.fft.fftfreq(i)
            res_torch = utils.fftfreq(i).numpy()
            self.assertTrue(np.allclose(res_np, res_torch), msg=f'Failure for {i}')

    def test_cov(self):

        array1 = np.random.rand(50, 10)
        array2 = np.random.rand(50, 10)

        res_np = np.cov(array1, array2)
        res_torch = utils.cov(torch.tensor(array1), torch.tensor(array2)).numpy()
        self.assertTrue(np.allclose(res_np, res_torch))

    def test_flatnonzero(self):
        ndims = np.arange(1, 4)
        for ndim in ndims:
            dims = [np.random.randint(1, 100) for i in range(ndim)]
            index_array = np.random.randint(0, 1, size=dims)
            array = np.random.rand(*dims)
            array[index_array] = 0
            tensor = torch.tensor(array)
            res_np = np.flatnonzero(array)
            res_torch = utils.flatnonzero(tensor).numpy()
            self.assertTrue((res_np == res_torch).all())


if __name__ == '__main__':
    unittest.main()
