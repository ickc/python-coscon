import numpy as np

from coscon.cmb import Maps, PowerSpectra

s1 = PowerSpectra.from_planck_2018()


def test_planck_2018_extended():
    s2 = PowerSpectra.from_planck_2018_extended()
    df1, df2 = s1.intersect(s2)
    np.testing.assert_allclose(df2.values, df1.values, rtol=0.15, atol=0.1)


def test_pysm():
    m = Maps.from_pysm(140, 128)
    s2 = m.to_spectra
    df1, df2 = s1.intersect(s2)
    np.testing.assert_allclose(df2.values, df1.values, rtol=0.1, atol=1000.)
