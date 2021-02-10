from __future__ import annotations

from pathlib import Path
from logging import getLogger
from dataclasses import dataclass
from functools import cached_property
from typing import List, TYPE_CHECKING

import defopt
from astropy.io import fits
from astropy.io.fits.hdu.table import BinTableHDU
import healpy as hp
import numpy as np
import pandas as pd
from tabulate import tabulate
import matplotlib.pyplot as plt

if TYPE_CHECKING:
    from typing import Optional, Dict

logger = getLogger('coscon')


@dataclass
class FitsHelper:
    path: Path = Path()
    memmap: bool = True

    def __post_init__(self):
        self.path = Path(self.path)

    def __str__(self) -> str:
        return '\n\n'.join((self.__info_str__, self.__infos_str__))

    def _repr_html_(self) -> str:
        return ''.join((self.__info_html__, self.__infos_html__))

    @cached_property
    def info(self) -> pd.DataFrame:
        with fits.open(self.path, memmap=self.memmap) as file:
            return pd.DataFrame(
                (tup[:7] for tup in file.info(output=False)),
                columns=['No.', 'Name', 'Ver', 'Type', 'Cards', 'Dimensions', 'Format'],
            )

    @cached_property
    def __info_str__(self) -> str:
        return tabulate(self.info, tablefmt='fancy_grid', headers="keys", showindex=False)

    @cached_property
    def __info_html__(self) -> str:
        return self.info.to_html(index=False)

    @cached_property
    def infos(self) -> List[pd.DataFrame]:
        with fits.open(self.path, memmap=self.memmap) as file:
            return [pd.Series(datum.header).to_frame(datum.name) for datum in file]

    @cached_property
    def __infos_str__(self) -> str:
        return '\n\n'.join(tabulate(df, tablefmt='fancy_grid', headers="keys") for df in self.infos)

    @cached_property
    def __infos_html__(self) -> str:
        return ''.join(df.to_html() for df in self.infos)

    @cached_property
    def n_maps(self) -> int:
        """Return total number of maps assuming it is a CMB fits
        """
        with fits.open(self.path, memmap=self.memmap) as file:
            for f in file:
                if type(f) is BinTableHDU:
                    return f.header['TFIELDS']

    @cached_property
    def names(self) -> List[str]:
        """Get names of maps
        """
        with fits.open(self.path, memmap=self.memmap) as file:
            for f in file:
                if type(f) is BinTableHDU:
                    # return [key for key in f.header if key.startswith('TTYPE')]
                    return [f.header[f'TTYPE{i}'] for i in range(1, self.n_maps + 1)]

    @cached_property
    def maps(self) -> np.ndarray:
        """Return a 2D healpix map array.
        where the first dimension is the i-th map
        """
        n_maps = self.n_maps
        # force 2D array
        # healpy tried to be smart and return 1D array only if there's only 1 map
        m = hp.read_map(self.path).reshape(1, -1) if n_maps == 1 else hp.read_map(self.path, field=range(n_maps))
        return m

    @cached_property
    def maps_dict(self) -> Dict[str, np.ndarray]:
        return dict(zip(map(lambda x: x[0], self.names), self.maps))

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


def _fits_info(*filenames: Path):
    """Print a summary of the HDUs in a FITS file(s).

    :param Path filenames: Path to one or more FITS files. Wildcards are supported.
    """
    for filename in filenames:
        print(FitsHelper(filename))


def fits_info_cli():
    defopt.run(_fits_info)
