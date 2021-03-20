from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from logging import getLogger
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar, List, Optional
from itertools import product

import h5py
import matplotlib.pyplot as plt
import plotly
from plotly.subplots import make_subplots
import plotly.figure_factory as ff
import numpy as np
import pandas as pd
import schema
import seaborn as sns
from matplotlib.colors import LogNorm, SymLogNorm
from schema import Schema, SchemaError
from toast.tod import hex_layout, hex_pol_angles_qu, plot_focalplane

import numba_quaternion

from .io_helper import H5_CREATE_KW, dumper, loader
from .util import geometric_matrix, unique_matrix, joshian_matrix, total_crosstalk_matrix, total_crosstalk_matrix_exact, omega_i_resonance_exact, C_n_resonance_exact

if TYPE_CHECKING:
    from typing import Any, Dict, Tuple, Union

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
    def dataframe(self) -> pd.DataFrame:
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

    @cached_property
    def azimuthal_equidistant_projection_with_orientation(self) -> np.ndarray[np.float_]:
        qs = numba_quaternion.Quaternion(
            numba_quaternion.lastcol_quat_to_canonical(
                np.array(self.dataframe.quat.values.tolist())
            )
        )
        return qs.azimuthal_equidistant_projection_with_orientation

    @cached_property
    def dataframe_with_azimuthal_equidistant_projection_with_orientation(self) -> pd.DataFrame:
        azimuthal_equidistant_projection_with_orientation = self.azimuthal_equidistant_projection_with_orientation
        df = self.dataframe.copy()
        temp = azimuthal_equidistant_projection_with_orientation[:, :2] * (180. / np.pi)
        df['x'] = temp[:, 0]
        df['y'] = temp[:, 1]
        df['orient_angle'] = azimuthal_equidistant_projection_with_orientation[:, 2]
        return df

    def iplot(
        self,
        scale: float = 1.,
        height: int = 1000,
        width: int = 1000,
    ) -> plotly.graph_objs._figure.Figure:
        pointing = self.azimuthal_equidistant_projection_with_orientation
        temp = pointing[:, :2] * (180. / np.pi)
        x = temp[:, 0]
        y = temp[:, 1]
        u = np.cos(pointing[:, 2])
        v = np.sin(pointing[:, 2])
        fig = ff.create_quiver(x, y, u, v, scale=scale, scaleratio=1.)
        fig.update_layout(height=height, width=width)
        return fig


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
        fontname: str = 'TeX Gyre Schola',
        wire: bool = False,
        wire_TB: bool = False,
        wire_connectionstyle: Optional[str] = None,
    ):
        """Plot the detectors in Azimuthal equidistant projection with orientation.

        For `wire_connectionstyle`, use 'arc3' to draw straight-line.
        See more in https://matplotlib.org/stable/gallery/userdemo/connectionstyle_demo.html#sphx-glr-gallery-userdemo-connectionstyle-demo-py
        """
        df = self.dataframe_with_azimuthal_equidistant_projection_with_orientation
        x_min = df.x.min()
        x_max = df.x.max()
        y_min = df.y.min()
        y_max = df.y.max()
        df['orient_angle_x'] = np.cos(df.orient_angle.values)
        df['orient_angle_y'] = np.sin(df.orient_angle.values)

        colors = sns.color_palette("Paired", 8)
        color_dict = dict(zip(product(
            ('A', 'B'),
            ('Q', 'U'),
            ('T', 'B'),
        ), colors))

        fig, ax = plt.subplots(figsize=(width, height))
        ax.set_aspect(1.)
        ax.set_xlabel("Degrees", fontsize="large", fontname=fontname)
        ax.set_ylabel("Degrees", fontsize="large", fontname=fontname)
        bleed = df.fwhm.max() / 40.
        ax.set_xlim([x_min - bleed, x_max + bleed])
        ax.set_ylim([y_min - bleed, y_max + bleed])
        if wire:
            if wire_TB:
                prev_head_x = None
                prev_head_y = None
                prev_orient_angle = None
                r2d = 180. / np.pi
            else:
                prev_x = None
                prev_y = None
                prev_pixel = None
        for _, row in df.iterrows():
            x = row.x
            y = row.y
            radius = row.fwhm / 120.
            pol = row.pol
            orient = row.orient
            handed = row.handed
            pixel = row.pixel
            orient_angle_x = row.orient_angle_x
            orient_angle_y = row.orient_angle_y

            color = color_dict[(
                handed,
                orient,
                pol,
            )]

            # put circle once per TB
            if pol == 'T':
                circle = plt.Circle((x, y), radius=radius, color=color)
                ax.add_artist(circle)
            else:
                ax.text(x, y, pixel, horizontalalignment='center', verticalalignment='center', color=color, fontname=fontname)

            dx = 2. * radius * orient_angle_x
            dy = 2. * radius * orient_angle_y
            current_tail_x = x - dx
            current_tail_y = y - dy
            ax.arrow(
                current_tail_x,
                current_tail_y,
                2. * dx,
                2. * dy,
                width=0.1 * radius,
                head_width=0.3 * radius,
                head_length=0.3 * radius,
                fc=color,
                ec=color,
                length_includes_head=True,
            )
            if wire:
                if wire_TB:
                    orient_angle = row.orient_angle * r2d
                    if prev_orient_angle is not None:
                        connectionstyle = f"angle3,angleA={prev_orient_angle + 90.},angleB={orient_angle + 90.}" if wire_connectionstyle is None else wire_connectionstyle
                        ax.annotate(
                            "",
                            xy=(current_tail_x, current_tail_y),
                            xytext=(prev_head_x, prev_head_y),
                            xycoords='data',
                            textcoords='data',
                            arrowprops=dict(
                                arrowstyle="->",
                                color="0.5",
                                connectionstyle=connectionstyle,
                            ),
                        )
                    # for next loop
                    prev_head_x = x + dx
                    prev_head_y = y + dy
                    prev_orient_angle = orient_angle
                else:
                    pixel = row.pixel
                    if prev_pixel is not None and pixel != prev_pixel:
                        connectionstyle = 'arc3' if wire_connectionstyle is None else wire_connectionstyle

                        # shorten arrow
                        delta_x = x - prev_x
                        delta_y = y - prev_y
                        angle = np.arctan2(delta_y, delta_x)
                        dx = 2. * radius * np.cos(angle)
                        dy = 2. * radius * np.sin(angle)
                        x_start = prev_x + dx
                        x_end = x - dx
                        y_start = prev_y + dy
                        y_end = y - dy
                        ax.annotate(
                            "",
                            xy=(x_end, y_end),
                            xytext=(x_start, y_start),
                            xycoords='data',
                            textcoords='data',
                            arrowprops=dict(
                                arrowstyle="->",
                                color="0.5",
                                connectionstyle=connectionstyle,
                            ),
                        )
                    # for next loop
                    prev_x = x
                    prev_y = y
                    prev_pixel = pixel
        plt.close()
        return fig


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
        self.iplot = self.detectors.iplot

    @cached_property
    def dataframe(self) -> pd.DataFrame:
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

    @cached_property
    def dataframe_with_azimuthal_equidistant_projection_with_orientation(self) -> pd.DataFrame:
        """DataFrame representation of the hardware
        """
        return (
            self.detectors.dataframe_with_azimuthal_equidistant_projection_with_orientation
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

    def reorder_to(
        self,
        pixels: List[int],
        TB_anti_alignment: bool = False,
        TB_alternate_anti_alignment: bool = False,
        validate_schema: bool = False,
    ) -> AvesHardware:
        """Reorder pixels according its integer assignment

        :param bool TB_anti_alignment: if True, assume TB pixels ordering BTTBBTTBBT..
        else BTBTBTBT...
        """
        df = self.dataframe
        num_to_name = {}
        # assume always end in T, B pair
        for name, num in df.pixel.items():
            if num not in num_to_name:
                num_to_name[num] = name[:-1]
            else:
                assert num_to_name[num] == name[:-1]
        names = [num_to_name[str(pixel).zfill(3)] for pixel in pixels]
        names_full = []
        if TB_anti_alignment:
            forward = True
            for name in names:
                if forward:
                    names_full.append(f'{name}B')
                    names_full.append(f'{name}T')
                    forward = False
                else:
                    names_full.append(f'{name}T')
                    names_full.append(f'{name}B')
                    forward = True
        elif TB_alternate_anti_alignment:
            for i, name in enumerate(names):
                mod = i % 4
                if mod == 0 or mod == 3:
                    names_full.append(f'{name}B')
                    names_full.append(f'{name}T')
                else:
                    names_full.append(f'{name}T')
                    names_full.append(f'{name}B')
        else:
            for name in names:
                names_full.append(f'{name}B')
                names_full.append(f'{name}T')
        return AvesHardware.from_dataframe(df.loc[names_full], validate_schema=validate_schema)

    def reorder_to_from_csv(
        self,
        path: Path,
        TB_anti_alignment: bool = False,
        TB_alternate_anti_alignment: bool = False,
        validate_schema: bool = False,
    ) -> AvesHardware:
        """Reorder pixels according its integer assignment

        :param Path path: a csv or txt file that has the pixel numbers per line
        :param bool TB_anti_alignment: if True, assume TB pixels ordering BTTBBTTBBT..
        else BTBTBTBT...
        """
        df_name = pd.read_csv(path, header=None)
        pixels = df_name.T.values[0]
        return self.reorder_to(
            pixels,
            TB_anti_alignment=TB_anti_alignment,
            TB_alternate_anti_alignment=TB_alternate_anti_alignment,
            validate_schema=validate_schema,
        )

# Crosstalk ####################################################################


@dataclass
class GenericMatrix:
    """Generic matrix that has row names in ASCII.
    """
    names: np.ndarray['S']
    data: np.ndarray[np.float64]
    name: Optional[str] = None

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
    freq: Optional[np.ndarray[np.float64]] = None

    @cached_property
    def dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(self.data, index=self.names_str, columns=self.names_str)

    @cached_property
    def dataframe_nearest_neighbor_only(self) -> pd.DataFrame:
        freq = self.freq

        data = self.data
        n = data.shape[0]

        left = np.empty(n, np.float64)
        left[0] = np.nan
        left[1:] = np.diag(data, k=-1)

        right = np.empty(n, np.float64)
        right[:-1] = np.diag(data, k=1)
        right[-1] = np.nan

        index = None if freq is None else pd.Index(freq, name='freq')
        df = pd.DataFrame({'left': left, 'right': right}, index=index)
        if self.name is not None:
            df.columns = [f'{self.name}-{col}' for col in df.columns]
        return df

    @cached_property
    def dataframe_diag_only(self) -> pd.DataFrame:
        return pd.Series(np.diag(self.data), index=self.freq).to_frame(self.name)

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
        if log:
            data = self.data
            if data.min() < 0.:
                linthresh = np.abs(data[~(data == 0.)]).min()
                norm = SymLogNorm(linthresh)
            else:
                norm = LogNorm()
        else:
            norm = None
        sns.heatmap(self.dataframe, annot=annot, ax=ax, norm=norm)
        plt.close()
        return fig

    def compare_plot_nearest_neighbor(
        self,
        others: List[CrosstalkMatrix],
    ):
        figs = [self.dataframe_nearest_neighbor_only.plot(backend='plotly')]
        for other in others:
            figs.append(other.dataframe_nearest_neighbor_only.plot(backend='plotly'))

        fig_all = make_subplots()
        colors = sns.color_palette("husl", len(others) * 2 + 2)
        i = 0
        for fig in figs:
            for trace in fig.data:
                trace['line']['color'] = 'rgb({}%)'.format('%,'.join(map(lambda x: str(x * 100.), colors[i])))
                fig_all.add_trace(trace)
                i += 1
        fig_all.layout.xaxis.title = 'f (Hz)'
        fig_all.layout.title = 'Nearest neighbor crosstalk'
        return fig_all

    def compare_plot_diag(
        self,
        others: List[CrosstalkMatrix],
    ):
        figs = [self.dataframe_diag_only.plot(backend='plotly')]
        for other in others:
            figs.append(other.dataframe_diag_only.plot(backend='plotly'))

        fig_all = make_subplots()
        colors = sns.color_palette("husl", len(others) + 1)
        i = 0
        for fig in figs:
            for trace in fig.data:
                trace['line']['color'] = 'rgb({}%)'.format('%,'.join(map(lambda x: str(x * 100.), colors[i])))
                fig_all.add_trace(trace)
                i += 1
        fig_all.layout.xaxis.title = 'f (Hz)'
        fig_all.layout.title = 'Diagonal "crosstalk"'
        return fig_all

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

    @classmethod
    def from_joshian_matrix(
        cls,
        names: Union[List[str], np.ndarray],
        nearest: float,
        next_nearest: float,
        floor: float,
    ) -> CrosstalkMatrix:
        """Generate a crosstalk matrix with structure based on Josh's simple assumption
        """
        n = len(names)
        m = joshian_matrix(n, nearest, next_nearest, floor)
        return cls(names, m)

    def to_random_normal(
        self,
    ) -> CrosstalkMatrix:
        """Generate a random normal matrix based on original matrix except for diagonal

        with mean and std given by the original matrix.
        """
        data = self.data
        res = np.random.normal(data, np.abs(data))
        n = self.size
        idxs = np.diag_indices(n)
        res[idxs] = data[idxs]
        return CrosstalkMatrix(self.names, res)


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
    def dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(self.data.T, columns=self.names_str)


@dataclass
class SQUID:
    """A model of the SQUID readout.

    See fig. 4 in Montgomery2020.
    
    Montgomery2020: Montgomery et al., “Performance and Characterization of the SPT-3G Digital Frequency Multiplexed Readout System Using an Improved Noise and Crosstalk Model.”
    """
    R_TES: Union[float, np.ndarray[np.float64]]
    r_s: Union[float, np.ndarray[np.float64]]
    L: Union[float, np.ndarray[np.float64]]
    L_com: float
    C: Optional[np.ndarray[np.float64]] = None
    omega: Optional[np.ndarray[np.float64]] = None

    def __post_init__(self):
        if (self.C is None) is (self.omega is None):
            raise ValueError(f'You need to specify either C or omega.')
        elif self.omega is None:
            self.omega = omega_i_resonance_exact(self.R_TES, self.r_s, self.L, self.C, self.L_com)
        elif self.C is None:
            self.C = C_n_resonance_exact(self.R_TES, self.r_s, self.L, self.L_com, self.omega)

    @cached_property
    def freq(self):
        return self.omega / (2. * np.pi)

    def to_dict(
        self,
        include_omega: bool = False,
        include_freq: bool = False,
    ) -> dict:
        res = {
            'R_TES': self.R_TES,
            'r_s': self.r_s,
            'L': self.L,
            'C': self.C,
            'L_com': self.L_com,
        }
        if include_omega:
            res['omega'] = self.mega
        if include_freq:
            res['freq'] = self.freq
        return res

    def dataframe(
        self,
        include_omega: bool = False,
        include_freq: bool = False,
    ) -> pd.DataFrame:
        return pd.DataFrame(self.to_dict(include_omega=include_omega, include_freq=include_freq))

    def to_crosstalk_matrix(
        self,
        names: Optional[Union[List[str], np.ndarray]] = None,
    ) -> CrosstalkMatrix:
        """Compute the crosstalk matrix using Montgomery2020 approximations"""
        data = total_crosstalk_matrix(self.R_TES, self.r_s, self.L, self.C, self.L_com, self.omega)
        if names is None:
            # use the numerical value of freq in MHz as names
            names = (self.freq * 1.e-6).astype(np.int).astype('S')
        return CrosstalkMatrix(names, data)

    def to_crosstalk_matrix_exact(
        self,
        names: Optional[Union[List[str], np.ndarray]] = None,
    ) -> CrosstalkMatrix:
        """Compute the crosstalk matrix using Montgomery2020 formulism without the approximations"""
        data = total_crosstalk_matrix_exact(self.R_TES, self.r_s, self.L, self.C, self.L_com, self.omega)
        if names is None:
            # use the numerical value of freq in MHz as names
            names = (self.freq * 1.e-6).astype(np.int).astype('S')
        return CrosstalkMatrix(names, data)
