from __future__ import annotations

from pathlib import Path
from logging import getLogger
from dataclasses import dataclass
from functools import cached_property
from typing import TYPE_CHECKING, ClassVar

import numpy as np
import pandas as pd
from schema import Schema, And
import seaborn as sns
from toast.tod import plot_focalplane, hex_pol_angles_qu, hex_layout

if TYPE_CHECKING:
    from typing import Optional, Dict, Union, Tuple

logger = getLogger('coscon')

FloatArray = And(np.ndarray, lambda quat: quat.dtype == np.float64)


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
class GenericFocalPlane:
    schema: ClassVar[dict] = {}
    content: Dict[str, Union[float, np.ndarray]]

    def __post_init__(self):
        self.validate()

    def validate(self):
        Schema(self.schema).validate(self.content)

    @cached_property
    def dataframe(self):
        """DataFrame representation of the focal plane

        This has opposite key-ordering from the dict
        i.e. self.content[some_key]['quat'] == self.dataframe['quat'][some_key] == self.dataframe.loc[some_key, 'quat']
        """
        return pd.DataFrame.from_dict(self.content, orient='index')


@dataclass
class FakeFocalPlane(GenericFocalPlane):
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
    content: Dict[str, Union[float, np.ndarray]]

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
            dict_ = self.content
            kinds = set(name[-1] for name in dict_)
            n_kinds = len(kinds)
            if n_kinds == 2:
                logger.info(f'Found detector names ending in 2 characters, treat them as unqiue polarizations: {kinds}')
                colors = dict(zip(kinds, sns.color_palette("husl", 2)))
                polcolor = {name: colors[name[-1]] for name in dict_}
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
