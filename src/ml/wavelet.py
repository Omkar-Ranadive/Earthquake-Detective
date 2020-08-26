from kymatio.torch import Scattering1D


def scatter_transform(data, excerpt_len, J=6, Q=32):
    """

    Args:
        data (torch tensor): Seismic data in form (batch, features, excerpt_len)
        excerpt_len (int): Number of samples in each seismic signal component
        J (int): The maximum log-scale of the scattering transform. In other words, the maximum
        scale is given by 2^J.
        Q (int): The number of first-order wavelets per octave (second-order wavelets are fixed to
         one wavelet per octave). Defaults to 1.

    Returns (torch tensor): Scattering coefficients of the form (batch, features, x, y)

    """
    # Initialize the scattering transform
    scattering = Scattering1D(J=J, Q=Q, shape=excerpt_len)

    return scattering(data)

