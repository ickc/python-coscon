import numpy as np
import pandas as pd

from coscon.cmb import Maps, PowerSpectra

s1 = PowerSpectra.from_planck_2018()


def test_planck_2018_extended():
    s2 = PowerSpectra.from_planck_2018_extended()
    df1, df2 = s1.intersect(s2)
    np.testing.assert_allclose(df2.values, df1.values, rtol=0.15, atol=0.1)


def test_pysm():
    m = Maps.from_pysm(140, 128)
    s2 = m.to_spectra()
    df1, df2 = s1.intersect(s2)
    np.testing.assert_allclose(df2.values, df1.values, rtol=0.1, atol=1000.)

def test_synfast():
    nside = 64
    # make this test func deterministic
    np.random.seed(nside)
    m = Maps.from_planck_2018(nside)
    s2 = m.to_spectra()
    df1, df2 = s1.intersect(s2)
    np.testing.assert_allclose(df2.values, df1.values, rtol=0.1, atol=1000.)


def test_power_spectra_matrix():
    s2 = s1.rotate(0.)
    pd.testing.assert_frame_equal(s1.dataframe[['TT', 'EE', 'BB', 'TE']], s2.dataframe[['TT', 'EE', 'BB', 'TE']])
    np.testing.assert_array_equal(s2.get_spectrum('TB'), 0.)
    np.testing.assert_array_equal(s2.get_spectrum('EB'), 0.)


def test_power_spectra_matrix_2():
    s2 = s1.rotate(0.25 * np.pi)
    np.testing.assert_array_equal(s1.get_spectrum('TT'), s2.get_spectrum('TT'))
    np.testing.assert_array_equal(s1.get_spectrum('EE'), s2.get_spectrum('BB'))
    np.testing.assert_array_equal(s1.get_spectrum('BB'), s2.get_spectrum('EE'))
    np.testing.assert_array_equal(s1.get_spectrum('TE'), -s2.get_spectrum('TB'))
    np.testing.assert_allclose(s2.get_spectrum('TE'), 0., atol=1e16)
    np.testing.assert_allclose(s2.get_spectrum('EB'), 0., atol=1e16)
