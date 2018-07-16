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
            index_array = np.random.randint(0, 2, size=dims)
            array = np.random.rand(*dims)
            array[index_array] = 0
            tensor = torch.tensor(array)
            res_np = np.flatnonzero(array)
            res_torch = utils.flatnonzero(tensor).numpy()
            self.assertTrue((res_np == res_torch).all())

    def test_dot(self):
        array_1_a = np.random.rand(5)
        array_1_b = np.random.rand(5)

        array_2_a = np.random.rand(5, 10)
        array_2_b = np.random.rand(5, 10)

        array_n = np.random.rand(5, 10, 20, 15)
        array_m = np.random.rand(5, 10, 20, 15, 5)

        # case 1: 2 1d arrays
        res_np = np.dot(array_1_a, array_1_b)
        res_torch = utils.dot(torch.tensor(array_1_a), torch.tensor(array_1_b)).numpy()
        self.assertTrue((res_np == res_torch).all())

        # case 2: 2 2d arrays
        res_np = np.dot(array_2_a, array_2_b.T)
        res_torch = utils.dot(torch.tensor(array_2_a), torch.tensor(array_2_b.T)).numpy()
        self.assertTrue(np.allclose(res_np, res_torch))

        ###########################################################################################
        #                  Scalar multiply is not available with pytorch matmul                   #
        ###########################################################################################
        # # case 3: one scalar
        # res_np = np.dot(scalar, array_2_b)
        # res_torch = utils.dot(torch.tensor(scalar), torch.tensor(array_2_b)).numpy()
        # self.assertTrue(np.allclose(res_np, res_torch))

        # res_np = np.dot(array_n, scalar)
        # res_torch = utils.dot(torch.tensor(array_n), scalar).numpy()
        # self.assertTrue(np.allclose(res_np, res_torch))

        # case 4: 1 nd array and 1 1d array
        res_np = np.dot(array_m, array_1_a)     # last dimensions must be same
        res_torch = utils.dot(torch.tensor(array_m), torch.tensor(array_1_a)).numpy()
        self.assertTrue(np.allclose(res_np, res_torch))

        ###########################################################################################
        #               Pytorch matmul with multidim arrays is different from numpy               #
        ###########################################################################################
        # # case 5: 1 nd array and 1 md array
        # res_np = np.dot(array_n, array_m)
        # res_torch = utils.dot(torch.tensor(array_n), torch.tensor(array_m)).numpy()
        # self.assertTrue(np.allclose(res_np, res_torch))


if __name__ == '__main__':
    unittest.main()
