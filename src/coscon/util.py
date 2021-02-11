import numpy as np
from numba import jit


@jit('float64[:, ::1](float64[:, ::1], int64[::1])', nopython=True, nogil=True, cache=True)
def from_Cl_to_Dl(spectra, l):
    """Convert from Cl scale to Dl scale.

    D_l = [l(l+1)/2pi] C_l
    """
    return ((0.5 / np.pi) * (l * (l + 1))).reshape(1, -1) * spectra


@jit('float64[:, ::1](float64[:, ::1], int64[::1])', nopython=True, nogil=True, cache=True)
def from_Dl_to_Cl(spectra, l):
    """Convert from Dl scale to Cl scale.

    D_l = [l(l+1)/2pi] C_l
    """
    return ((2. * np.pi) / (l * (l + 1))).reshape(1, -1) * spectra
