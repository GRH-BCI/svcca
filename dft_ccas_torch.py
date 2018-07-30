from skcuda import fft
import numpy as np
import pycuda
import torch
import utils

import pandas as pd
from cca_core_torch import get_cca_similarity_torch

complex_factor = pycuda.gpuarray.to_gpu(np.array(1j, dtype=np.complex64))


def fft_resize_torch(images, resize=False, new_size=None):
    assert len(images.shape) == 4, ('expecting images to be'
                                    '[batch_size, height, width, num_channels]')
    images = utils.tensor_to_gpuarray(images)
    im_fft = utils.fft2(images, axes=[1, 2]).copy()

    # resizing images
    if resize:
        # get fourier frequencies to threshold
        assert (im_fft.shape[1] == im_fft.shape[2]), ('Need images to have same' 'height and width')
        # downsample by threshold
        width = im_fft.shape[2]
        new_width = new_size[0]
        freqs = np.fft.fftfreq(width, d=1.0 / width)
        idxs = np.flatnonzero((freqs >= -new_width / 2.0) & (freqs < new_width / 2.0))
        im_fft_downsampled = pycuda.gpuarray.empty((im_fft.shape[0], len(idxs), len(idxs),
                                                    im_fft.shape[3]), dtype=im_fft.dtype)
        for n, idx in enumerate(idxs):
            for n2, idx2 in enumerate(idxs):
                im_fft_downsampled[:, n, n2, :] = im_fft[:, int(idx), int(idx2), :]

    else:
        im_fft_downsampled = im_fft

    return im_fft_downsampled


def fourier_ccas_torch(conv_acts1, conv_acts2, return_coefs=False,
                       compute_dirns=False, verbose=False):
    height1, width1 = conv_acts1.shape[1], conv_acts1.shape[2]
    height2, width2 = conv_acts2.shape[1], conv_acts2.shape[2]
    if height1 != height2 or width1 != width2:
        height   = min(height1, height2)
        width    = min(width1, width2)
        new_size = [height, width]
        resize   = True
    else:
        height   = height1
        width    = width1
        new_size = None
        resize   = False

    # resize and preprocess with fft
    fft_acts1 = fft_resize_torch(conv_acts1, resize=resize, new_size=new_size)
    fft_acts2 = fft_resize_torch(conv_acts2, resize=resize, new_size=new_size)

    # loop over spatial dimensions and get cca coefficients
    all_results = pd.DataFrame()
    for i in range(height):
        for j in range(width):
            results_dict = get_cca_similarity_torch(
                fft_acts1[:, i, j, :].transpose(0, 1),
                fft_acts2[:, i, j, :].transpose(0, 1),
                compute_dirns=compute_dirns,
                verbose=verbose
            )

            # apply inverse FFT to get coefficients and directions if specified
            if return_coefs:
                results_dict['neuron_coeffs1'] = torch.irfft(results_dict['neuron_coeffs1'],
                                                             signal_ndim=2)
                results_dict['neuron_coeffs2'] = torch.irfft(results_dict['neuron_coeffs2'],
                                                             signal_ndim=2)
            else:
                del results_dict['neuron_coeffs1']
                del results_dict['neuron_coeffs2']

            if compute_dirns:
                results_dict['cca_dirns1'] = torch.irfft(results_dict['cca_dirns1'], signal_ndim=2)
                results_dict['cca_dirns2'] = torch.irfft(results_dict['cca_dirns2'], signal_ndim=2)

            # accumulate results
            results_dict['location'] = (i, j)
            all_results = all_results.append(results_dict, ignore_index=True)

    return all_results
