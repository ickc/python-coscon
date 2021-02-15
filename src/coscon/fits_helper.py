from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from logging import getLogger
from pathlib import Path
from typing import List

import defopt
import healpy as hp
import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.io.fits.hdu.table import BinTableHDU
from tabulate import tabulate

import coscon.cmb

logger = getLogger('coscon')


@dataclass
class BaseFitsHelper:
    """A class for general fits files"""
    path: Path
    memmap: bool = True
    name: str = ''

    def __post_init__(self):
        self.path = Path(self.path)
        if not self.name:
            self.name = self.path.stem

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
        return tabulate(self.info, tablefmt='psql', headers="keys", showindex=False)

    @cached_property
    def __info_html__(self) -> str:
        return self.info.to_html(index=False)

    @cached_property
    def infos(self) -> List[pd.DataFrame]:
        with fits.open(self.path, memmap=self.memmap) as file:
            return [pd.Series(datum.header).to_frame(datum.name) for datum in file]

    @cached_property
    def __infos_str__(self) -> str:
        return '\n\n'.join(tabulate(df, tablefmt='psql', headers="keys") for df in self.infos)

    @cached_property
    def __infos_html__(self) -> str:
        return ''.join(df.to_html() for df in self.infos)


@dataclass
class CMBFitsHelper(BaseFitsHelper):
    """Specialized in fits container typically used in CMB analysis"""

    @cached_property
    def n_maps(self) -> int:
        """Return total number of maps assuming it is a CMB fits
        """
        with fits.open(self.path, memmap=self.memmap) as file:
            for f in file:
                if type(f) is BinTableHDU:
                    return f.header['TFIELDS']
        raise TypeError(f"Is {self.path} really contains CMB maps?")

    @cached_property
    def names(self) -> List[str]:
        """Get names of maps
        """
        with fits.open(self.path, memmap=self.memmap) as file:
            for f in file:
                if type(f) is BinTableHDU:
                    # return [key for key in f.header if key.startswith('TTYPE')]
                    return [f.header[f'TTYPE{i}'] for i in range(1, self.n_maps + 1)]
        raise TypeError(f"Is {self.path} really contains CMB maps?")

    @cached_property
    def maps(self) -> np.ndarray:
        """Return a 2D healpix map array.
        where the first dimension is the i-th map
        """
        n_maps = self.n_maps
        # force 2D array
        # healpy tried to be smart and return 1D array only if there's only 1 map
        return (
            hp.read_map(self.path).reshape(1, -1)
        ) if n_maps == 1 else (
            hp.read_map(self.path, field=range(n_maps))
        )

    @property
    def to_maps(self) -> coscon.cmb.Maps:
        """package data in a Maps object"""
        return coscon.cmb.Maps(self.names, self.maps, name=self.name)


def _fits_info(*filenames: Path):
    """Print a summary of the HDUs in a FITS file(s).

    :param Path filenames: Path to one or more FITS files. Wildcards are supported.
    """
    for filename in filenames:
        print(BaseFitsHelper(filename))


def fits_info_cli():
    defopt.run(_fits_info)
