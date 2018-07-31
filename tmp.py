import pycuda.gpuarray
import pycuda.autoinit
import numpy as np
from os.path import join, dirname


def fft_resize_torch():
    images = pycuda.gpuarray.to_gpu(np.load(join(dirname(__file__), 'array1.npy')))
    im_fft = fft2(images, axes=[1, 2])

    # downsample
    im_fft_downsampled = pycuda.gpuarray.empty((1024, 3, 3, 64), dtype=im_fft.dtype)

    for n, idx in enumerate([1, 2, 3]):
        im_fft_downsampled[:, n, n, :] = im_fft[:, int(idx), int(idx), :]

    return im_fft_downsampled

####################################################################################################
#  Taken from https://github.com/fjarri/reikna/blob/develop/examples/demo_real_to_complex_fft.py   #
####################################################################################################


def get_complex_trf(arr):
    from reikna.cluda import dtypes
    from reikna.core import Annotation, Type, Transformation, Parameter
    complex_dtype = dtypes.complex_for(arr.dtype)
    return Transformation(
        [Parameter('output', Annotation(Type(complex_dtype, arr.shape), 'o')),
         Parameter('input', Annotation(arr, 'i'))],
         """
         ${output.store_same}(
             COMPLEX_CTR(${output.ctype})(
                 ${input.load_same},
                 0));
         """)


def fft2(array_gpu, axes=None):
    import reikna
    import reikna.cluda as cluda
    from reikna.fft import FFT

    thread = cluda.cuda_api().Thread(pycuda.autoinit.context)
    complex_transf = get_complex_trf(array_gpu)

    fft = FFT(complex_transf.output, axes=axes)
    fft.parameter.input.connect(complex_transf, complex_transf.output,
                                new_input=complex_transf.input)
    cfft = fft.compile(thread)

    result_gpu = thread.array(array_gpu.shape, np.complex64)
    cfft(result_gpu, array_gpu)
    return result_gpu

if __name__ == '__main__':
    resized_torch = fft_resize_torch()
    print(resized_torch)
