from __future__ import annotations

from pathlib import Path
from logging import getLogger
from dataclasses import dataclass
from functools import cached_property
from typing import TYPE_CHECKING, ClassVar, Optional, List

import seaborn as sns
import numpy as np
import pandas as pd
import schema
from schema import Schema, SchemaError
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from toast.tod import plot_focalplane, hex_pol_angles_qu, hex_layout
import h5py

from .io_helper import dumper, loader
from .util import geometric_matrix, unique_matrix
from .io_helper import H5_CREATE_KW

if TYPE_CHECKING:
    from typing import Dict, Union, Tuple, Any

logger = getLogger('coscon')

# schema types
Int = schema.Or(int, np.int64)
Float = schema.Or(float, np.float64)
FloatArray = schema.And(np.ndarray, lambda quat: quat.dtype == np.float64)
Char = schema.And(str, lambda s: len(s) == 1)
OptionalChar = schema.Or(None, Char)
Number = schema.Or(float, int, np.float64, np.int64)
OptionalNumber = schema.Or(None, float, int, np.float64, np.int64)
ListOfString = schema.And(list, lambda list_: all(type(i) is str for i in list_))


def na_to_none(value):
    """Convert pandas null value to None"""
    return None if pd.isna(value) else value


class DictValueEqualsKey:
    """A fake dictionary where its values always equal their keys"""

    def __getitem__(self, key):
        return key


# from https://github.com/ickc/toast-workshop ##################################


def fake_focalplane(
    samplerate: float = 20.,
    epsilon: float = 0.,
    net: float = 1.,
    fmin: float = 0.,
    alpha: float = 1.,
    fknee: float = 0.05,
    fwhm: float = 30.,
    npix: int = 7,
    fov: float = 3.,
) -> Dict[str, Union[float, np.ndarray]]:
    """Create a set of fake detectors.

    This generates 7 pixels (14 dets) in a hexagon layout at the boresight
    and with a made up polarization orientation.

    This function is copied from TOAST workshop with the following license:

    BSD 2-Clause License

    Copyright (c) 2019, hpc4cmb
    All rights reserved.

    Redistribution and use in source and binary forms, with or without
    modification, are permitted provided that the following conditions are met:

    1. Redistributions of source code must retain the above copyright notice, this
    list of conditions and the following disclaimer.

    2. Redistributions in binary form must reproduce the above copyright notice,
    this list of conditions and the following disclaimer in the documentation
    and/or other materials provided with the distribution.

    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
    AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
    IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
    DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
    FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
    DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
    SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
    CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
    OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
    OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
    """
    pol_A = hex_pol_angles_qu(npix)
    pol_B = hex_pol_angles_qu(npix, offset=90.0)

    dets_A = hex_layout(npix, fov, "", "", pol_A)
    dets_B = hex_layout(npix, fov, "", "", pol_B)

    dets = {}
    for p in range(npix):
        pstr = "{:01d}".format(p)
        for d, layout in zip(["A", "B"], [dets_A, dets_B]):
            props = dict()
            props["quat"] = layout[pstr]["quat"]
            props["epsilon"] = epsilon
            props["rate"] = samplerate
            props["alpha"] = alpha
            props["NET"] = net
            props["fmin"] = fmin
            props["fknee"] = fknee
            props["fwhm_arcmin"] = fwhm
            dname = "{}{}".format(pstr, d)
            dets[dname] = props
    return dets


# FocalPlane ###################################################################


@dataclass
class GenericDictStructure:
    schema: ClassVar[dict] = {}
    data: Dict[str, Any]
    validate_schema: bool = True

    def __post_init__(self):
        if self.validate_schema:
            self.validate()

    def validate(self):
        try:
            Schema(self.schema).validate(self.data)
        except SchemaError as e:
            logger.warn(f'{type(self).__name__} scheme error detected:')
            raise e

    @cached_property
    def dataframe(self):
        """DataFrame representation of the focal plane

        This has opposite key-ordering from the dict
        i.e. self.data[some_key]['quat'] == self.dataframe['quat'][some_key] == self.dataframe.loc[some_key, 'quat']
        """
        return pd.DataFrame.from_dict(self.data, orient='index')

    def dump(
        self,
        path: Union[Path, str],
        overwrite: bool = False,
        compress: bool = None
    ):
        """Write hardware config to a TOML/YAML/JSON file.
        """
        dumper(self.data, path, overwrite=overwrite, compress=compress)

    @classmethod
    def load(cls, path: Union[Path, str]):
        """Read data from a TOML/YAML/JSON file.
        """
        data = loader(path)
        return cls(data)


@dataclass
class GenericFocalPlane(GenericDictStructure):

    def __post_init__(self):
        # quat is nparray[float64]
        # but writing to and reading from toml
        # resulted in list of str
        # so we have to cast it back to the right types
        for value in self.data.values():
            quat = value['quat']
            if type(quat) is not np.ndarray:
                value['quat'] = np.array(quat)
        super().__post_init__()


@dataclass
class FakeFocalPlane(GenericFocalPlane):
    """a class describing the output of fake_focalplane"""
    schema: ClassVar[dict] = {
        str: {
            'quat': FloatArray,
            'epsilon': Float,
            'rate': Float,
            'alpha': Float,
            'NET': Float,
            'fmin': Float,
            'fknee': Float,
            'fwhm_arcmin': Float,
        }
    }
    data: Dict[str, Union[float, np.ndarray]]

    def plot(
        self,
        width: float = 4.,
        height: float = 4.,
        outfile: Optional[Path] = None,
        facecolor: Optional[Dict[str, Union[str, Tuple[float, float, float]]]] = None,
        polcolor: Optional[Dict[str, Union[str, Tuple[float, float, float]]]] = None,
    ):
        df = self.dataframe

        if polcolor is None:
            data = self.data
            kinds = set(name[-1] for name in data)
            n_kinds = len(kinds)
            if n_kinds == 2:
                logger.info(f'Found detector names ending in 2 characters, treat them as unqiue polarizations: {kinds}')
                colors = dict(zip(kinds, sns.color_palette("husl", 2)))
                polcolor = {name: colors[name[-1]] for name in data}
            else:
                logger.warn('Cannot guess the polarization of detectors, drawing with no polarization color. Define polcolor explicitly to fix this.')

        return plot_focalplane(
            df.quat,
            width,
            height,
            outfile,
            fwhm=df.fwhm_arcmin,
            facecolor=facecolor,
            polcolor=polcolor,
            labels=DictValueEqualsKey(),
        )


@dataclass
class AvesDetectors(GenericFocalPlane):
    """a class describing the detectors of Aves.

    similar to FakeFocalPlane
    """
    schema: ClassVar[dict] = {
        str: {
            'wafer': str,
            'pixel': str,
            'pixtype': str,
            'band': str,
            'fwhm': Float,
            'pol': Char,
            schema.Optional('handed'): OptionalChar,
            'orient': Char,
            'quat': FloatArray,
            'UID': Int,
        }
    }
    data: Dict[str, Optional[Union[str, float, int, np.ndarray]]]

    def plot(
        self,
        width: float = 20.,
        height: float = 20.,
        outfile: Optional[Path] = None,
    ):
        df = self.dataframe
        color = {}
        for case in ('pixtype', 'pol'):
            col = df[case]
            values = col.unique()
            n = values.size
            color[case] = col.map(dict(zip(values, sns.color_palette("husl", n))))

        return plot_focalplane(
            df.quat,
            width,
            height,
            outfile,
            fwhm=df.fwhm,
            facecolor=color['pixtype'],
            polcolor=color['pol'],
            labels=DictValueEqualsKey(),
        )


@dataclass
class AvesBands(GenericDictStructure):
    """a class describing the bands of Aves.
    """
    schema: ClassVar[dict] = {
        str: {
            'center': Float,
            'bandwidth': Number,
            'bandpass': str,
            'NET': Float,
            'fwhm': Float,
            'fknee': Float,
            'fmin': Float,
            'alpha': Number,
        }
    }
    data: Dict[str, Union[float, int, str]]


@dataclass
class AvesPixels(GenericDictStructure):
    """a class describing the pixels of Aves.
    """
    schema: ClassVar[dict] = {
        str: {
            'sizemm': float,
            'bands': ListOfString,
        }
    }
    data: Dict[str, Union[float, List[str]]]


@dataclass
class AvesWafers(GenericDictStructure):
    """a class describing the wafers of Aves.
    """
    schema: ClassVar[dict] = {
        str: {
            'type': str,
            'npixel': Int,
            'pixels': ListOfString,
            schema.Optional('handed'): OptionalNumber,
            'position': Int,
            'telescope': str,
        }
    }
    data: Dict[str, Optional[Union[float, int, str, List[str]]]]


@dataclass
class AvesTelescopes(GenericDictStructure):
    """a class describing the telescopes of Aves.
    """
    schema: ClassVar[dict] = {
        str: {
            'wafers': ListOfString,
            'platescale': Float,
            'waferspace': Float,
        }
    }
    data: Dict[str, Union[float, List[str]]]


@dataclass
class AvesHardware(GenericDictStructure):
    schema: ClassVar[dict] = {
        str: {
            'telescopes': dict,
            'wafers': dict,
            'pixels': dict,
            'bands': dict,
            'detectors': dict,
        }
    }
    data: Dict[str, dict]

    def __post_init__(self):
        data =self.data
        self.telescopes = AvesTelescopes(data['telescopes'])
        self.wafers = AvesWafers(data['wafers'])
        self.pixels = AvesPixels(data['pixels'])
        self.bands = AvesBands(data['bands'])
        self.detectors = AvesDetectors(data['detectors'])
        self.plot = self.detectors.plot

    @cached_property
    def dataframe(self):
        """DataFrame representation of the hardware
        """
        return (
            self.detectors.dataframe
            .merge(
                self.telescopes.dataframe.merge(
                    self.wafers.dataframe,
                    left_index=True,
                    right_on='telescope',
                    how='outer',
                ).drop('wafers', axis=1),
                left_on='wafer',
                right_index=True,
                how='outer',
                suffixes=('', '_wafer')
            ).drop('pixels', axis=1)
            .merge(
                self.pixels.dataframe,
                left_on='pixtype',
                right_index=True,
                how='outer',
            )
            .merge(
                self.bands.dataframe,
                left_on='band',
                right_index=True,
                how='outer',
                suffixes=('', '_band')
            ).drop(['bands', 'fwhm_band'], axis=1)
        )


    @classmethod
    def from_dataframe(cls, df: pd.DataFrame, validate_schema=True):
        """Create Hardware from a dataframe representation
        """
        final: Dict[str, Dict[str, Dict[str, Any]]] = {}
        case: List[str]
        unique_cols: List[List[str]]
        non_unique_cols: List[List[str]]
        for case, unique_cols, non_unique_cols in [
            [
                ['telescope', 'telescopes'],
                [
                    ['platescale', 'platescale'],
                    ['waferspace', 'waferspace'],
                ],
                [
                    ['wafer', 'wafers'],
                ],
            ], [
                ['wafer', 'wafers'],
                [
                    ['type', 'type'],
                    ['npixel', 'npixel'],
                    ['handed_wafer', 'handed'],
                    ['position', 'position'],
                    ['telescope', 'telescope'],
                ],
                [
                    ['pixtype', 'pixels'],
                ],
            ], [
                ['pixtype', 'pixels'],
                [
                    ['sizemm', 'sizemm'],
                ],
                [
                    ['band', 'bands'],
                ],
            ], [
                ['band', 'bands'],
                [
                    ['center', 'center'],
                    ['bandwidth', 'bandwidth'],
                    ['bandpass', 'bandpass'],
                    ['NET', 'NET'],
                    ['fwhm', 'fwhm'],
                    ['fknee', 'fknee'],
                    ['fmin', 'fmin'],
                    ['alpha', 'alpha'],
                ],
                [],
            ],
        ]:
            final[case[1]] = temps = {}
            for name, group in df.groupby(case[0]):
                temps[name] = temp = {}
                for col, col_new in unique_cols:
                    values = group[col].unique()
                    assert values.size == 1
                    temp[col_new] = na_to_none(values[0])
                for col, col_new in non_unique_cols:
                    values = group[col].unique()
                    temp[col_new] = values.tolist()

        # detectors
        df_detectors = df[['wafer', 'pixel', 'pixtype', 'band', 'fwhm', 'pol', 'handed', 'orient', 'quat', 'UID']]
        final['detectors'] = df_detectors.to_dict(orient='index')
        return cls(final, validate_schema=validate_schema)


# Crosstalk ####################################################################


@dataclass
class GenericMatrix:
    """Generic matrix that has row names in ASCII.
    """
    names: np.ndarray['S']
    data: np.ndarray[np.float64]

    def __post_init__(self):
        self.names = np.asarray(self.names, dtype='S')
        assert self.names.size == self.data.shape[0]

    def __eq__(self, other: GenericMatrix) -> bool:
        if type(self) is not type(other):
            return False
        return np.array_equal(self.names, other.names) and np.array_equal(self.data, other.data)

    @property
    def size(self) -> int:
        return self.data.shape[0]

    @property
    def shape(self) -> int:
        return self.data.shape

    @cached_property
    def names_str(self) -> List[str]:
        """names in list of str"""
        return [name.decode() for name in self.names]

    @classmethod
    def load(cls, path: Path):
        with h5py.File(path, 'r') as f:
            names = f["names"][:]
            data = f["data"][:]
        return cls(names, data)

    def dump(self, path: Path, compress_level: int = 9):
        with h5py.File(path, 'w', libver='latest') as f:
            f.create_dataset(
                'names',
                data=self.names,
                compression_opts=compress_level,
                **H5_CREATE_KW
            )
            f.create_dataset(
                'data',
                data=self.data,
                compression_opts=compress_level,
                **H5_CREATE_KW
            )


@dataclass(eq=False)
class CrosstalkMatrix(GenericMatrix):
    """Crosstalk square matrix where column and row share the same names
    """

    @cached_property
    def dataframe(self):
        return pd.DataFrame(self.data, index=self.names_str, columns=self.names_str)

    def plot(
        self,
        annot: bool = True,
        figsize: Optional[Tuple[float, float]] = None,
        log: bool = True,
    ):
        if figsize is None:
            n = self.size
            figsize = (n, n)
        fig, ax = plt.subplots(figsize=figsize)
        norm = LogNorm() if log else None
        sns.heatmap(self.dataframe, annot=annot, ax=ax, norm=norm)
        return fig

    @classmethod
    def from_geometric(
        cls,
        names: Union[List[str], np.ndarray],
        r: float = 0.01,
    ) -> CrosstalkMatrix:
        """Generate a crosstalk matrix with coefficients in geometric series.
        """
        n = len(names)
        m = geometric_matrix(n, r)
        return cls(names, m)

    @classmethod
    def from_unique_matrix(
        cls,
        names: Union[List[str], np.ndarray],
    ) -> CrosstalkMatrix:
        """generate a crosstalk matrix with each entry to be unique.

        Mainly for debug use to confirm the matrix multiplication is correct.
        """
        n = len(names)
        m = unique_matrix(n)
        return cls(names, m)


@dataclass(eq=False)
class NaiveTod(GenericMatrix):
    """Naive TOD matrix that has all data in contiguous array
    """

    def apply_crosstalk(self, crosstalk_matrix: CrosstalkMatrix) -> NaiveTod:
        names = self.names
        np.testing.assert_array_equal(names, crosstalk_matrix.names)
        data = crosstalk_matrix.data @ self.data
        return NaiveTod(names, data)

    @cached_property
    def dataframe(self):
        return pd.DataFrame(self.data.T, columns=self.names_str)
