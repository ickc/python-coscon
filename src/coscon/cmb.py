from __future__ import annotations

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
    from typing import Optional, Dict

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
