from __future__ import annotations

from pathlib import Path
from logging import getLogger
from dataclasses import dataclass
from functools import cached_property
from typing import List

import defopt
from astropy.io import fits
import pandas as pd
from tabulate import tabulate


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
    def infos(self) -> List(pd.DataFrame):
        with fits.open(self.path, memmap=self.memmap) as file:
            return [pd.Series(datum.header).to_frame(datum.name) for datum in file]

    @cached_property
    def __infos_str__(self) -> str:
        return '\n\n'.join(tabulate(df, tablefmt='fancy_grid', headers="keys") for df in self.infos)

    @cached_property
    def __infos_html__(self) -> str:
        return ''.join(df.to_html() for df in self.infos)


def _fits_info(*filenames: Path):
    """Print a summary of the HDUs in a FITS file(s).

    :param Path filenames: Path to one or more FITS files. Wildcards are supported.
    """
    for filename in filenames:
        print(FitsHelper(filename))


def fits_info_cli():
    defopt.run(_fits_info)
