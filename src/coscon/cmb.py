from __future__ import annotations

import io
from pathlib import Path
from logging import getLogger
from dataclasses import dataclass
from functools import cached_property
from typing import List, TYPE_CHECKING

import healpy as hp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

if TYPE_CHECKING:
    from typing import Optional, Dict, Tuple

import coscon.fits_helper

logger = getLogger('coscon')


@dataclass
class Maps:
    """A class for multiple healpix maps
    that wraps some healpy functions

    In uK_RJ ($\\mathrm{\\mu K_{{RJ}}}$) unit.
    """
    names: List[str]
    maps: np.ndarray

    def __post_init__(self):
        maps = self.maps
        assert maps.ndim == 2
        assert maps.shape[0] == len(self.names)

    @property
    def n_maps(self) -> int:
        return self.maps.shape[0]

    @classmethod
    def from_fits(
        cls,
        path: Path,
        memmap: bool = True,
        nest: bool = True,
    ) -> Maps:
        fits_helper = coscon.fits_helper.CMBFitsHelper(
            path,
            memmap=memmap,
            nest=nest,
        )
        return fits_helper.to_maps

    @classmethod
    def from_pysm(
        cls,
        freq: float,
        nside: int,
        preset_strings: List[str] = ["c1"],
    ) -> Maps:
        import pysm3
        import pysm3.units as u

        sky = pysm3.Sky(nside=nside, preset_strings=preset_strings)
        freq_u = freq * u.GHz
        m = sky.get_emission(freq_u)
        return cls(
            ['T', 'Q', 'U'],
            m.to(u.uK_CMB, equivalencies=u.cmb_equivalencies(freq_u)),
        )

    @cached_property
    def maps_dict(self) -> Dict[str, np.ndarray]:
        return dict(zip(map(lambda x: x[0], self.names), self.maps))

    @cached_property
    def maps_dict_fullname(self) -> Dict[str, np.ndarray]:
        return dict(zip(self.names, self.maps))

    @cached_property
    def spectra(self) -> np.ndarray:
        """Use anafast to calculate the spectra
        results are always 2D-array
        """
        res = hp.sphtfunc.anafast(self.maps)
        # force 2D array
        # healpy tried to be smart and return 1D array only if there's only 1 map
        return res.reshape(1, -1) if self.n_maps == 1 else res

    @property
    def to_spectra(self) -> PowerSpectra:
        n_maps = self.n_maps
        if n_maps == 1:
            names = ['TT']
        elif n_maps == 3:
            names = ['TT', 'EE', 'BB', 'TE', 'EB', 'TB']
        else:
            raise ValueError(f'There are {n_maps} maps where I can only understand 1 map or 3 maps.')
        return PowerSpectra(names, self.spectra, scale='Cl')

    @staticmethod
    def _mollview(
        name: str,
        m: np.ndarray,
        n_std: Optional[int] = None,
        fwhm: Optional[float] = None,
    ):
        if fwhm is not None:
            m = hp.sphtfunc.smoothing(m, fwhm=fwhm * np.pi / 10800.)
        if n_std is None:
            min_ = None
            max_ = None
        else:
            mean = m.mean()
            std = m.std()
            min_ = mean - n_std * std
            max_ = mean + n_std * std
        hp.mollview(m, title=name, min=min_, max=max_)
        hp.graticule()
        plt.show()

    def mollview(
        self,
        n_std: Optional[int] = None,
        fwhm: Optional[float] = None,
    ):
        """object wrap of healpy.mollview

        :param n_std: if specified, set the range to be mean ± n_std * std
        :param fwhm: if specified, smooth map to this number of arcmin
        """
        maps_dict = self.maps_dict
        for name, m in maps_dict.items():
            self._mollview(name, m, n_std=n_std, fwhm=fwhm)
        if 'Q' in maps_dict and 'U' in maps_dict:
            name = 'P'
            m = np.sqrt(np.square(maps_dict['Q']) + np.square(maps_dict['U']))
            self._mollview(name, m, n_std=n_std, fwhm=fwhm)


@dataclass
class PowerSpectra:
    """A class for power-spectra.

    In [microK]^2 unit.
    """
    names: List[str]
    spectra: np.ndarray
    l_min: int = 0
    l: Optional[np.ndarray] = None
    scale: str = 'Dl'

    def __post_init__(self):
        # make sure l_min and l are self-consistent
        l = self.l
        if l is not None:
            self.l_min = l[0]

        spectra = self.spectra
        assert spectra.ndim == 2
        assert spectra.shape[0] == len(self.names)

        if self.scale == 'Cl':
            from .util import from_Cl_to_Dl

            logger.info('Converting from Cl scale to Dl scale automatically.')
            self.spectra = from_Cl_to_Dl(self.spectra, self.l_array)
            self.scale = 'Dl'
        assert self.scale == 'Dl'

    def _repr_html_(self) -> str:
        return self.dataframe._repr_html_()

    @property
    def n_spectra(self) -> int:
        return self.spectra.shape[0]

    @cached_property
    def l_max(self) -> int:
        """exclusive l_max"""
        l = self.l
        return self.l_min + self.spectra.shape[1] if l is None else l[-1] + 1

    @cached_property
    def l_array(self) -> np.ndarray:
        l = self.l
        return np.arange(self.l_min, self.l_max) if l is None else l

    @cached_property
    def dataframe(self) -> pd.DataFrame:
        l = self.l
        index = pd.RangeIndex(self.l_min, self.l_max, name='l') if l is None else pd.Index(l, name='l')
        return pd.DataFrame(self.spectra.T, index=index, columns=self.names)

    @classmethod
    def from_dataframe(
        cls,
        df: pd.DataFrame,
        scale: str = 'Dl',
    ) -> PowerSpectra:
        names = df.columns.tolist()
        spectra = df.values.T
        l = df.index.values
        return cls(names, spectra, l=l, scale=scale)

    @classmethod
    def from_class(
        cls,
        path: Optional[Path] = None,
        url: Optional[str] = None,
        camb: bool = True,
    ) -> PowerSpectra:
        '''read from CLASS' .dat output

        :param bool camb: if True, assumed ``format = camb`` is used when
        generating the .dat from CLASS

        To self: migrated from abx's convert_theory.py
        '''
        if path is None and url is None:
            raise ValueError(f'either path or url has to be specified.')
        if url is not None:
            from astropy.utils.data import download_file

            if path is not None:
                logger.warn(f'Ignoring path as url is specified.')
            logger.info(f'Downloading and caching using astropy: {url}')
            path = download_file(url, cache=True)
        # read once
        with open(path, 'r') as f:
            text = f.read()

        # get last comment line
        comment = None
        for line in text.split('\n'):
            if line.startswith('#'):
                comment = line
        # remove beginning '#'
        comment = comment[1:]
        # parse comment line: get name after ':'
        names = [name.split(':')[1] for name in comment.strip().split()]

        with io.StringIO(text) as f:
            df = pd.read_csv(f, delim_whitespace=True, index_col=0, comment='#', header=None, names=names)
        # to [microK]^2
        if not camb:
            df *= 1.e12
        return cls.from_dataframe(df)

    @classmethod
    def from_planck_2018(cls) -> PowerSpectra:
        """Use the Planck 2018 best-fit ΛCDM model

        Officially released by Planck up to l equals 2508.
        see http://pla.esac.esa.int/pla/#cosmology and its description
        """
        from astropy.utils.data import download_file

        # see http://pla.esac.esa.int/pla/#cosmology and its description
        file = download_file(
            'http://pla.esac.esa.int/pla/aio/product-action?COSMOLOGY.FILE_ID=COM_PowerSpect_CMB-base-plikHM-TTTEEE-lowl-lowE-lensing-minimum-theory_R3.01.txt',
            cache=True,
        )
        calPlanck = 0.1000442E+01

        df = pd.read_csv(file, sep='\s+', names=['l', 'TT', 'TE', 'EE', 'BB', 'PP'], comment='#', index_col=0)
        # planck's 2018 released spectra above 2500 has identically zero BB so it shouldn't be trusted
        df = df.loc[pd.RangeIndex(2, 2501)]
        df *= 1. / (calPlanck * calPlanck)
        return cls.from_dataframe(df)

    @classmethod
    def from_planck_2018_extended(cls) -> PowerSpectra:
        """Use the Planck 2018 best fit ΛCDM model with extended l-range up to 4902

        This is reproduced using CLASS but with an extended l-range
        """
        return cls.from_class(url='https://gist.github.com/ickc/cd6bb1753a44ee09f2d09c49b190d65f/raw/398497217109000c0d670bb57cfcc2cb9a617d63/base_2018_plikHM_TTTEEE_lowl_lowE_lensing_cl_lensed.dat')

    def intersect(self, other: PowerSpectra) -> Tuple[pd.DataFrame, pd.DataFrame]:
        cols = np.intersect1d(self.names, other.names)
        l = np.intersect1d(self.l_array, other.l_array)
        return self.dataframe.loc[l, cols], other.dataframe.loc[l, cols]

    def compare(self, other: PowerSpectra, self_name: str, other_name: str) -> pd.DataFrame:
        """Return a tidy DataFrame comparing self and other.
        """
        df_all = pd.concat(
            self.intersect(other),
            keys=[self_name, other_name],
            names=['case', 'l']
        )
        df_all.columns.name = 'spectra'
        return df_all.stack().to_frame(self.scale).reset_index(level=(0, 1, 2))

    def compare_plot(self, other: PowerSpectra, self_name: str, other_name: str):
        import plotly.express as px

        df_tidy = self.compare(other, self_name, other_name)
        for name, group in df_tidy.groupby('spectra'):
            px.line(group, x='l', y='Dl', color='case', title=name).show()
