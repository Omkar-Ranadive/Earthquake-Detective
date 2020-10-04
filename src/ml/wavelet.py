from kymatio.torch import Scattering1D
import torch


def scatter_transform(data, excerpt_len, J=8, Q=32, cuda=False):
    """

    Args:
        data (torch tensor): Seismic data in form (batch, features, excerpt_len)
        excerpt_len (int): Number of samples in each seismic signal component
        J (int): The maximum scale of filters. That is, largest filter will be of size 2^J in
        time interval.
        Q (int): The number of first-order wavelets per octave (second-order wavelets are fixed to
         one wavelet per octave). Defaults to 1. Larger the Q, better frequency resolution.
        cuda (bool): Scattering transform will use GPU if true and GPU is available

    Returns (torch tensor): Scattering coefficients of the form (batch, features, x, y)

    """

    # Initialize the scattering transform
    scattering = Scattering1D(J=J, Q=Q, shape=excerpt_len)
    log_eps = 1e-6

    # Use GPU is cuda = True and GPU is available
    if cuda and torch.cuda.is_available():
        data = data.cuda()
        scattering.cuda()

    s_coeffs = scattering.forward(data)

    # Get rid of the 0th order scattering co-efficients
    s_coeffs = s_coeffs[:, :, 1:, :]

    # Calculate the log scattering transform
    s_coeffs = torch.log(torch.abs(s_coeffs) + log_eps)

    # Average over the time dimension
    s_coeffs = torch.mean(s_coeffs, dim=-1)

    # Clear the GPU memory
    del data
    del scattering

    return s_coeffs



def scatter_transform_v2(data, excerpt_len, J=8, Q=32, cuda=False):
    """
    The v2 returns two wavelet coefficients matrices - one averaged across time, the second
    averaged across frequencies for each time bin
    Args:
        data (torch tensor): Seismic data in form (batch, features, excerpt_len)
        excerpt_len (int): Number of samples in each seismic signal component
        J (int): The maximum scale of filters. That is, largest filter will be of size 2^J in
        time interval.
        Q (int): The number of first-order wavelets per octave (second-order wavelets are fixed to
         one wavelet per octave). Defaults to 1. Larger the Q, better frequency resolution.
        cuda (bool): Scattering transform will use GPU if true and GPU is available

    Returns (torch tensor): Scattering coefficients of the form (batch, features, x, y)

    """

    # Initialize the scattering transform
    scattering = Scattering1D(J=J, Q=Q, shape=excerpt_len)
    log_eps = 1e-6

    # Use GPU is cuda = True and GPU is available
    if cuda and torch.cuda.is_available():
        data = data.cuda()
        scattering.cuda()

    s_coeffs = scattering.forward(data)

    # Get rid of the 0th order scattering co-efficients
    s_coeffs = s_coeffs[:, :, 1:, :]

    # Calculate the log scattering transform
    s_coeffs = torch.log(torch.abs(s_coeffs) + log_eps)
    # Average over the time dimension
    s_coeffs_f = torch.mean(s_coeffs, dim=-1)
    # Average over freq dimension
    s_coeffs_t = torch.mean(s_coeffs, dim=-2)
    # Clear the GPU memory
    del data
    del scattering

    return s_coeffs_f, s_coeffs_t


if __name__ == '__main__':
    x = torch.rand(10, 3, 20000)
    scatter_transform_v2(data=x, excerpt_len=20000, cuda=True)
