import numpy as np

from coscon.cmb import PowerSpectra


def test_sanity():
    s1 = PowerSpectra.from_planck_2018()
    s2 = PowerSpectra.from_planck_2018_extended()

    cols = ['TT', 'EE', 'BB', 'TE']
    l = s1.l_array
    np.testing.assert_allclose(s1.dataframe.loc[l, cols].values, s2.dataframe.loc[l, cols].values, rtol=0.17, atol=0.06)
