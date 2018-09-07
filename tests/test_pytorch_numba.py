import unittest
from os.path import join, dirname
import numpy as np
np.random.seed(0)
import torch
import numba.cuda as cuda


def create_cai_dict(tensor):
    import sys
    endianness = '<' if sys.byteorder == 'little' else '>'
    types = {
        torch.float32: 'f4',
        torch.float: 'f4',
        torch.float64: 'f8',
        torch.double: 'f8',
        torch.float16: 'f2',
        torch.half: 'f2',
        torch.uint8: 'u1',
        torch.int8: 'i1',
        torch.int16: 'i2',
        torch.short: 'i2',
        torch.int32: 'i4',
        torch.int: 'i4',
        torch.int64: 'i8',
        torch.long: 'i8'
    }
    typestr = endianness + types[tensor.dtype]
    cai_dict = {
        'shape': tuple(tensor.shape),
        'data': (tensor.data_ptr(), True),
        'typestr': typestr,
        'version': 0,
        'strides': tuple(s * tensor.storage().element_size() for s in tensor.stride()),
        'descr': [('', typestr)]
    }
    return cai_dict

class TestPTNumba(unittest.TestCase):

    def test_conversion(self):
        array = np.random.rand(5, 5)
        tensor = torch.tensor(array).cuda()
        setattr(tensor, '__cuda_array_interface__', create_cai_dict(tensor))
        device_array = cuda.as_cuda_array(tensor)
        self.assertTrue(np.allclose(device_array.copy_to_host(None), array))
        import ipdb; ipdb.set_trace()  # XXX BREAKPOINT


if __name__ == '__main__':
    unittest.main()
