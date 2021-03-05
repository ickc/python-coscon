from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from numba import jit

if TYPE_CHECKING:
    from typing import Union


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


# Montgomery et al., “Performance and Characterization of the SPT-3G Digital Frequency Multiplexed Readout System Using an Improved Noise and Crosstalk Model.”
# these are all relative magntudes, see comments after eq. 8, 12
# equivalent to set V = 1, δR = 1


@jit(
    [
        '''complex128[:, ::1](
            float64[::1],
            float64[::1],
            float64,
            float64[::1],
            float64[::1],
        )''',
        '''complex128[:, ::1](
            float64[::1],
            float64[::1],
            float64[::1],
            float64[::1],
            float64[::1],
        )''',
    ],
    nopython=True,
    nogil=True,
    cache=True,
)
def Z_n_omega_i(
    R_TES_n: np.ndarray[np.float64],
    r_s_n: np.ndarray[np.float64],
    L_n: Union[float, np.ndarray[np.float64]],
    C_n: np.ndarray[np.float64],
    omega_i: np.ndarray[np.float64],
) -> np.ndarray[np.complex128]:
    """Impedance of a single cryogenic filter leg n at angular freq. omega_i

    Eq. 1

    Return 2d-array, where the 1st dim as i & 2nd dim as n
    """
    # put i in the 1st axis
    omega_i_2d = omega_i.reshape(-1, 1)
    return (R_TES_n + r_s_n) + 1.j * (omega_i_2d * L_n - np.reciprocal(omega_i_2d * C_n))


@jit('complex128[::1](complex128[:, ::1])', nopython=True, nogil=True, cache=True)
def Z_net_omega_i(
    Z_n_omega_i: np.ndarray[np.complex128],
) -> np.ndarray[np.complex128]:
    """Impedance of full network at angular freq. omega_i

    Eq. 2
    """
    return np.reciprocal(np.reciprocal(Z_n_omega_i).sum(axis=1))


@jit('complex128[::1](float64, float64[::1])', nopython=True, nogil=True, cache=True)
def Z_com_omega_i(
    L_com: float,
    omega_i: np.ndarray[np.float64],
) -> np.ndarray[np.complex128]:
    return (1.j * L_com) * omega_i


@jit(
    [
        '''complex128[::1](
            float64[::1],
            float64[::1],
            float64,
            float64[::1],
            float64,
            float64[::1],
        )''',
        '''complex128[::1](
            float64[::1],
            float64[::1],
            float64[::1],
            float64[::1],
            float64,
            float64[::1],
        )''',
    ],
    nopython=True,
    nogil=True,
    cache=True,
)
def Z_tot_i(
    R_TES_n: np.ndarray[np.float64],
    r_s_n: np.ndarray[np.float64],
    L_n: Union[float, np.ndarray[np.float64]],
    C_n: np.ndarray[np.float64],
    L_com: float,
    omega_i: np.ndarray[np.float64],
) -> np.ndarray[np.complex128]:
    Z_in = Z_n_omega_i(R_TES_n, r_s_n, L_n, C_n, omega_i)
    Z_net_i = Z_net_omega_i(Z_in)
    Z_com_i = Z_com_omega_i(L_com, omega_i)
    return Z_net_i + Z_com_i


@jit(
    [
        '''float64[::1](
            float64[::1],
            float64[::1],
            float64,
            float64[::1],
            float64,
            float64[::1],
        )''',
        '''float64[::1](
            float64[::1],
            float64[::1],
            float64[::1],
            float64[::1],
            float64,
            float64[::1],
        )''',
    ],
    nopython=True,
    nogil=True,
    cache=True,
)
def Z_tot_norm_sq_i(
    R_TES_n: np.ndarray[np.float64],
    r_s_n: np.ndarray[np.float64],
    L_n: Union[float, np.ndarray[np.float64]],
    C_n: np.ndarray[np.float64],
    L_com: float,
    omega_i: np.ndarray[np.float64],
) -> np.ndarray[np.float64]:
    Z_i = Z_tot_i(R_TES_n, r_s_n, L_n, C_n, L_com, omega_i)
    Z_i_real = np.real(Z_i)
    Z_i_imag = np.imag(Z_i)
    return np.square(Z_i_real) + np.square(Z_i_imag)


@jit(
    [
        '''float64[::1](
            float64[::1],
            float64[::1],
            float64,
            float64[::1],
            float64,
            float64[::1],
        )''',
        '''float64[::1](
            float64[::1],
            float64[::1],
            float64[::1],
            float64[::1],
            float64,
            float64[::1],
        )''',
    ],
    nopython=True,
    nogil=True,
    cache=True,
)
def d_Z_tot_norm_sq_d_omega_i_over_2(
    R_TES_n: np.ndarray[np.float64],
    r_s_n: np.ndarray[np.float64],
    L_n: Union[float, np.ndarray[np.float64]],
    C_n: np.ndarray[np.float64],
    L_com: float,
    omega_i: np.ndarray[np.float64],
) -> np.ndarray[np.float64]:
    Z_in = Z_n_omega_i(R_TES_n, r_s_n, L_n, C_n, omega_i)
    Z_net_i = Z_net_omega_i(Z_in)
    Z_net_i_real = np.real(Z_net_i)
    Z_net_i_imag = np.imag(Z_net_i)
    Z_net_i_abs_sq = np.square(Z_net_i_real) + np.square(Z_net_i_imag)

    return L_com * (omega_i * L_com + Z_net_i_imag) + np.real(
        ((omega_i * Z_net_i) * L_com + 1.j * Z_net_i_abs_sq) * Z_net_i * (
            (L_n + np.reciprocal(np.square(omega_i).reshape(-1, 1) * C_n)) * np.power(Z_in, -2)
        ).sum(axis=1)
    )


@jit(
    [
        '''float64(
            float64[::1],
            float64[::1],
            float64[::1],
            float64,
            float64[::1],
            float64,
        )''',
        '''float64(
            float64[::1],
            float64[::1],
            float64[::1],
            float64[::1],
            float64[::1],
            float64,
        )''',
    ],
    nopython=True,
    nogil=True,
    cache=True,
)
def d_Z_tot_norm_sq_d_omega_over_2_for_optimize(
    omega_i: np.ndarray[np.float64],
    R_TES_n: np.ndarray[np.float64],
    r_s_n: np.ndarray[np.float64],
    L_n: Union[float, np.ndarray[np.float64]],
    C_n: np.ndarray[np.float64],
    L_com: float,
) -> float:
    res = d_Z_tot_norm_sq_d_omega_i_over_2(R_TES_n, r_s_n, L_n, C_n, L_com, omega_i)
    return res[0]


@jit(
    [
        '''float64[::1](
            float64,
            float64[::1],
        )''',
        '''float64[::1](
            float64[::1],
            float64[::1],
        )''',
    ],
    nopython=True,
    nogil=True,
    cache=True,
)
def omega_i_resonance_naive(
    L_n: Union[float, np.ndarray[np.float64]],
    C_n: np.ndarray[np.float64],
) -> np.ndarray[np.float64]:
    return np.reciprocal(np.sqrt(L_n * C_n))


def omega_i_resonance_exact(
    R_TES_n: np.ndarray[np.float64],
    r_s_n: np.ndarray[np.float64],
    L_n: Union[float, np.ndarray[np.float64]],
    C_n: np.ndarray[np.float64],
    L_com: float,
) -> np.ndarray[np.float64]:
    from scipy.optimize import fsolve

    omega_i_guess = omega_i_resonance_naive(L_n, C_n)
    N = C_n.size
    omega_i = np.empty(N)
    for i, omega in enumerate(omega_i_guess):
        temp = fsolve(d_Z_tot_norm_sq_d_omega_over_2_for_optimize, omega, args=(R_TES_n, r_s_n, L_n, C_n, L_com))
        assert temp.size == 1
        omega_i[i] = temp[0]
    return omega_i


@jit('complex128[:, ::1](complex128[:, ::1], complex128[::1])', nopython=True, nogil=True, cache=True)
def primary_signal_and_leakage_current_crosstalk_n_omega_i(
    Z_n_omega_i: np.ndarray[np.complex128],
    Z_com_omega_i: np.ndarray[np.complex128],
) -> np.ndarray[np.complex128]:
    """Primary signal & leakage current crosstalk at leg n & angular freq. omega_i

    Eq. 7

    Return 2d-array, where the 1st dim as i & 2nd dim as n
    """
    # put i in the 1st axis
    return -np.power(Z_n_omega_i + Z_com_omega_i.reshape(-1, 1), -2)


@jit('complex128[:, ::1](float64[::1], complex128[:, ::1], complex128[::1])', nopython=True, nogil=True, cache=True)
def leakage_power_crosstalk(
    R_TES_i: np.ndarray[np.float64],
    Z_n_omega_i: np.ndarray[np.complex128],
    Z_com_omega_n: np.ndarray[np.complex128],
) -> np.ndarray[np.complex128]:
    """Leakage power crosstalk onto the i-th detector from leakage current induced by the n-th voltage bias

    Eq. 11

    This made an approximation. See `leakage_power_crosstalk_exact` for an exact result.
    """
    Z_nn = np.diag(Z_n_omega_i)
    temp_n = (Z_nn * Z_com_omega_n) * np.power(Z_nn + Z_com_omega_n, -3)
    # put i in the 1st axis
    res = ((2. * temp_n) * R_TES_i.reshape(-1, 1)) * np.power(np.ascontiguousarray(Z_n_omega_i.T), -2)
    # diagonal term is non-sense in this equation
    for i in range(res.shape[0]):
        res[i, i] = 0.
    return res


@jit(
    [
        '''float64[:, ::1](
            float64[::1],
            float64[::1],
            float64,
            float64[::1],
            float64,
            float64[::1],
        )''',
        '''float64[:, ::1](
            float64[::1],
            float64[::1],
            float64[::1],
            float64[::1],
            float64,
            float64[::1],
        )''',
    ],
    nopython=True,
    nogil=True,
    cache=True,
)
def total_crosstalk_matrix(
    R_TES_n: np.ndarray[np.float64],
    r_s_n: np.ndarray[np.float64],
    L_n: Union[float, np.ndarray[np.float64]],
    C_n: np.ndarray[np.float64],
    L_com: float,
    omega_i: np.ndarray[np.float64],
) -> np.ndarray[np.complex128]:
    """Total crosstalk with 2 approximations in Montgomery's paper

    See `total_crosstalk_matrix_exact` for an exact result.
    """
    Z_in = Z_n_omega_i(R_TES_n, r_s_n, L_n, C_n, omega_i)
    Z_com_i = Z_com_omega_i(L_com, omega_i)
    total = (
        primary_signal_and_leakage_current_crosstalk_n_omega_i(Z_in, Z_com_i) + \
        leakage_power_crosstalk(R_TES_n, Z_in, Z_com_i)
    )
    # put i in the 1st axis
    primary_signal_i = np.diag(total).reshape(-1, 1)
    total /= primary_signal_i
    # approximation used in sec 3.2
    return np.ascontiguousarray(np.real(total))
