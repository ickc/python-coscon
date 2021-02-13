from __future__ import annotations

from pathlib import Path
from logging import getLogger
from dataclasses import dataclass
from functools import cached_property
from typing import TYPE_CHECKING, ClassVar, Optional, List

import numpy as np
import pandas as pd
import schema
from schema import Schema
import seaborn as sns
from toast.tod import plot_focalplane, hex_pol_angles_qu, hex_layout

from .io_helper import dumper, loader

if TYPE_CHECKING:
    from typing import Optional, Dict, Union, Tuple, Any

logger = getLogger('coscon')

FloatArray = schema.And(np.ndarray, lambda quat: quat.dtype == np.float64)
Char = schema.And(str, lambda s: len(s) == 1)
OptionalChar = schema.Or(None, Char)
Number = schema.Or(float, int)
ListOfString = schema.And(list, lambda list_: all(type(i) is str for i in list_))


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

    def __post_init__(self):
        self.validate()

    def validate(self):
        Schema(self.schema).validate(self.data)

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
class FakeFocalPlane(GenericDictStructure):
    """a class describing the output of fake_focalplane"""
    schema: ClassVar[dict] = {
        str: {
            'quat': FloatArray,
            'epsilon': float,
            'rate': float,
            'alpha': float,
            'NET': float,
            'fmin': float,
            'fknee': float,
            'fwhm_arcmin': float,
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
class AvesDetectors(GenericDictStructure):
    """a class describing the detectors of Aves.

    similar to FakeFocalPlane
    """
    schema: ClassVar[dict] = {
        str: {
            'wafer': str,
            'pixel': str,
            'pixtype': str,
            'band': str,
            'fwhm': float,
            'pol': Char,
            'handed': OptionalChar,
            'orient': Char,
            'quat': FloatArray,
            'UID': int,
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
            'center': float,
            'bandwidth': Number,
            'bandpass': str,
            'NET': float,
            'fwhm': float,
            'fknee': float,
            'fmin': float,
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
            'npixel': int,
            'pixels': ListOfString,
            schema.Optional('handed'): Number,
            'position': int,
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
            'platescale': float,
            'waferspace': float,
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
