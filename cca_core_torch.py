import torch
import utils
import pycuda
from skcuda.misc import divide, multiply
from skcuda.linalg import eig, diag

num_cca_trials = 5
epsilon = 1e-6


def positivedef_matrix_sqrt_pycuda(array):
    w, v = eig(array)
    #  A - np.dot(v, np.dot(np.diag(w), v.T))
    wsqrt = pycuda.cumath.sqrt(w)
    sqrtarray = multiply(v, multiply(diag(wsqrt), v.T))
    return sqrtarray


def remove_small_pycuda(sigma_xx, sigma_xy, sigma_yx, sigma_yy, threshold=1e-6):
    x_diag = abs(diag(sigma_xx))
    y_diag = abs(diag(sigma_yy))
    x_idxs = (x_diag >= threshold)
    y_idxs = (y_diag >= threshold)

    sigma_xx_crop = sigma_xx[x_idxs][:, x_idxs]
    sigma_xy_crop = sigma_xy[x_idxs][:, y_idxs]
    sigma_yx_crop = sigma_yx[y_idxs][:, x_idxs]
    sigma_yy_crop = sigma_yy[y_idxs][:, y_idxs]

    return (sigma_xx_crop, sigma_xy_crop, sigma_yx_crop, sigma_yy_crop, x_idxs,
            y_idxs)


def compute_ccas_pycuda(sigma_xx, sigma_xy, sigma_yx, sigma_yy, verbose=True):
    (sigma_xx, sigma_xy, sigma_yx, sigma_yy, x_idxs, y_idxs) = remove_small_pycuda(
        sigma_xx, sigma_xy, sigma_yx, sigma_yy)

    numx = sigma_xx.shape[0]
    numy = sigma_yy.shape[0]

    if numx == 0 or numy == 0:
        return ([0, 0, 0], [0, 0, 0], torch.zeros_like(sigma_xx),
                torch.zeros_like(sigma_yy), x_idxs, y_idxs)

    if verbose:
        print('adding eps to diagonal and taking inverse')
    sigma_xx += epsilon * torch.eye(numx)
    sigma_yy += epsilon * torch.eye(numy)
    inv_xx = torch.pinverse(sigma_xx)
    inv_yy = torch.pinverse(sigma_yy)

    if verbose:
        print('taking square root')
    invsqrt_xx = positivedef_matrix_sqrt_pycuda(inv_xx)
    invsqrt_yy = positivedef_matrix_sqrt_pycuda(inv_yy)

    if verbose:
        print('dot products...')
    arr_x = torch.ot(sigma_yx, invsqrt_xx)
    arr_x = torch.ot(inv_yy, arr_x)
    arr_x = torch.ot(invsqrt_xx, utils.dot(sigma_xy, arr_x))
    arr_y = torch.ot(sigma_xy, invsqrt_yy)
    arr_y = torch.ot(inv_xx, arr_y)
    arr_y = torch.ot(invsqrt_yy, utils.dot(sigma_yx, arr_y))

    if verbose:
        print('trying to take final svd')
    arr_x_stable = arr_x + epsilon * torch.eye(arr_x.shape[0])
    arr_y_stable = arr_y + epsilon * torch.eye(arr_y.shape[0])
    ux, sx, vx = torch.svd(arr_x_stable)
    uy, sy, vy = torch.svd(arr_y_stable)

    sx = torch.sqrt(torch.abs(sx))
    sy = torch.sqrt(torch.abs(sy))
    if verbose:
        print('computed everything!')

    return [ux, sx, vx], [uy, sy, vy], invsqrt_xx, invsqrt_yy, x_idxs, y_idxs


def sum_threshold_pycuda(array, threshold):
    assert (threshold >= 0) and (threshold <= 1), 'incorrect threshold'

    for i in range(len(array)):
        if torch.sum(array[:i]) / torch.sum(array) >= threshold:
            return i


def create_zero_dict_pycuda(compute_dirns, dimension):
    return_dict = {}
    return_dict['mean'] = (torch.tensor(0), torch.tensor(0))
    return_dict['sum'] = (torch.tensor(0), torch.tensor(0))
    return_dict['cca_coef1'] = torch.tensor(0)
    return_dict['cca_coef2'] = torch.tensor(0)
    return_dict['idx1'] = 0
    return_dict['idx2'] = 0

    if compute_dirns:
        return_dict['cca_dirns1'] = torch.zeros((1, dimension))
        return_dict['cca_dirns2'] = torch.zeros((1, dimension))

    return return_dict


def get_cca_similarity_pycuda(acts1, acts2, threshold=0.98, compute_dirns=True,
                             verbose=True, context=pycuda.autoinit.context):
    # assert dimensionality equal
    assert acts1.shape[1] == acts2.shape[1], 'dimensions don\'t match'
    # check that acts1, acts2 are transposition
    assert acts1.shape[0] < acts1.shape[1], ('input must be number of neurons by datapoints')
    return_dict = {}

    # compute covariance with numpy function for extra stability
    numx = acts1.shape[0]

    # TODO
    covariance = utils.cov_pycuda(acts1, acts2, context=context)
    sigmaxx = covariance[:numx, :numx].copy()
    sigmaxy = covariance[:numx, numx:].copy()
    sigmayx = covariance[numx:, :numx].copy()
    sigmayy = covariance[numx:, numx:].copy()

    # rescale covariance to make cca computation more stable
    xmax = pycuda.gpuarray.max(abs(sigmaxx))
    ymax = pycuda.gpuarray.max(abs(sigmayy))
    sigmaxx = divide(sigmaxx, xmax)
    sigmayy = divide(sigmayy, ymax)
    sigmaxy = divide(sigmaxy, pycuda.cumath.sqrt(xmax * ymax))
    sigmayx = divide(sigmayx, pycuda.cumath.sqrt(xmax * ymax))

    import ipdb; ipdb.set_trace()  # XXX BREAKPOINT
    ([_, sx, vx], [_, sy, vy], invsqrt_xx, invsqrt_yy, x_idxs,
     y_idxs) = compute_ccas_pycuda(sigmaxx, sigmaxy, sigmayx, sigmayy, verbose)

    # if x_idxs or y_idxs is all false, return_dict has zero entries
    if (not torch.any(x_idxs)) or (not torch.any(y_idxs)):
        return create_zero_dict_pycuda(compute_dirns, acts1.shape[1])

    if compute_dirns:
        # orthonormal directions that are CCA directions
        cca_dirns1 = utils.dot(vx, utils.dot(invsqrt_xx, acts1[x_idxs]))
        cca_dirns2 = utils.dot(vy, utils.dot(invsqrt_yy, acts2[y_idxs]))

    # get rid of trailing zeros in the cca coefficients
    idx1 = sum_threshold_pycuda(sx, threshold)
    idx2 = sum_threshold_pycuda(sy, threshold)

    return_dict['neuron_coeffs1'] = utils.dot(vx, invsqrt_xx)
    return_dict['neuron_coeffs2'] = utils.dot(vy, invsqrt_yy)
    return_dict['cca_coef1'] = sx
    return_dict['cca_coef2'] = sy
    return_dict['x_idxs'] = x_idxs
    return_dict['y_idxs'] = y_idxs
    # summary statistics
    return_dict['mean'] = (torch.mean(sx[:idx1]), torch.mean(sy[:idx2]))
    return_dict['sum'] = (torch.sum(sx), torch.sum(sy))

    if compute_dirns:
        return_dict['cca_dirns1'] = cca_dirns1
        return_dict['cca_dirns2'] = cca_dirns2

    return return_dict
