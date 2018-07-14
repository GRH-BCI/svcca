import torch
import utils


def fft_resize_torch(images, resize=False, new_size=None):
    assert len(images.shape) == 4, ('expecting images to be'
                                    '[batch_size, height, width, num_channels]')

    im_fft = torch.rfft(images, signal_ndim=2)

    # resizing images
    if resize:
        # get fourier frequencies to threshold
        assert (im_fft.shape[1] == im_fft.shape[2]), ('Need images to have same' 'height and width')
        # downsample by threshold
        width = im_fft.shape[2]
        new_width = new_size[0]
        freqs = utils.fftfreq(width, d=1.0 / width)
        idxs = np.flatnonzero((freqs >= -new_width / 2.0) & (freqs < new_width / 2.0))
        im_fft_downsampled = im_fft[:, :, idxs, :][:, idxs, :, :]

    else:
        im_fft_downsampled = im_fft

    return im_fft_downsampled
