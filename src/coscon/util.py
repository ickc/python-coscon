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


# generate some square matrices ################################################


@jit('float64[:, ::1](int64, float64)', nopython=True, nogil=True, cache=True)
def geometric_matrix(n, r):
    """Generate a matrix with coefficients in geometric series.
    """
    idx = np.arange(n)
    return np.power(r, np.abs(idx.reshape(-1, 1) - idx.reshape(1, -1)))


@jit('float64[:, ::1](int64)', nopython=True, nogil=True, cache=True)
def unique_matrix(n):
    """generate a matrix with each entry to be unique.
    """
    return np.reciprocal(np.arange(n, n * n + n, dtype=np.float64).reshape((n, n))) + np.identity(n)


@jit('float64[:, ::1](int64, float64, float64, float64)', nopython=True, nogil=True, cache=True)
def joshian_matrix(n, nearest, next_nearest, floor):
    """generate a matrix based on Josh's simple assumption
    """
    res = np.empty((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(n):
            d = np.abs(i - j)
            if d > 2:
                strength = floor
            elif d == 1:
                strength = nearest
            elif d == 2:
                strength = next_nearest
            else:
                strength = 1.
            res[i, j] = strength
    return res
