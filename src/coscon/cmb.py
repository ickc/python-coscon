from __future__ import annotations

import io
from collections import defaultdict
from dataclasses import dataclass
from functools import cached_property
from logging import getLogger
from pathlib import Path
from typing import TYPE_CHECKING, List, Optional, Union

import defopt
import healpy as hp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly
import plotly.express as px
import pysm3
import pysm3.units as u
from astropy.utils.data import download_file

if TYPE_CHECKING:
    from typing import Dict, Tuple

import coscon.fits_helper

from .util import from_Cl_to_Dl, from_Dl_to_Cl

logger = getLogger('coscon')

CANONICAL_NAME = ('TEMPERATURE', 'Q_POLARISATION', 'U_POLARISATION')


@dataclass
class Maps:
    """A class for multiple healpix maps
    that wraps some healpy functions

    In ring ordering, uK ($\\mathrm{\\mu K}$) unit.
    """
    names: Union[List[str], Tuple[str, ...]]
    maps: np.ndarray
    name: str = ''

    def __post_init__(self):
        maps = self.maps
        assert maps.ndim == 2
        assert maps.shape[0] == len(self.names)

    @property
    def n_maps(self) -> int:
        return self.maps.shape[0]

    @property
    def n_pix(self) -> int:
        return self.maps.shape[1]

    @cached_property
    def n_side(self) -> int:
        return hp.npix2nside(self.n_pix)

    @cached_property
    def f_sky(self) -> float:
        try:
            mask = self.maps.mask
            return 1. - mask.sum() / mask.size
        except (NameError, AttributeError):
            return 1.

    @classmethod
    def from_fits(
        cls,
        path: Path,
        memmap: bool = True,
    ) -> Maps:
        fits_helper = coscon.fits_helper.CMBFitsHelper(
            path,
            memmap=memmap,
        )
        return fits_helper.to_maps

    @classmethod
    def from_pysm(
        cls,
        freq: float,
        nside: int,
        preset_strings: List[str] = ["c1"],
    ) -> Maps:
        sky = pysm3.Sky(nside=nside, preset_strings=preset_strings)
        freq_u = freq * u.GHz
        m = sky.get_emission(freq_u)
        return cls(
            CANONICAL_NAME,
            m.to(u.uK_CMB, equivalencies=u.cmb_equivalencies(freq_u)),
            name=f"PySM 3 {freq} GHz map with preset {', '.join(preset_strings)}"
        )

    @classmethod
    def from_planck_2018(cls, nside: int) -> Maps:
        """Use the Planck 2018 best-fit ΛCDM model to simulate a map

        Use official release if nside is small enough.
        """
        if nside > 833:
            logger.info('Using extended Planck 2018 spectra computed using CLASS.')
            spectra = PowerSpectra.from_planck_2018_extended()
        else:
            spectra = PowerSpectra.from_planck_2018()
        return spectra.to_maps(nside)

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
        res = (1. / self.f_sky) * hp.sphtfunc.anafast(self.maps)
        # force 2D array
        # healpy tried to be smart and return 1D array only if there's only 1 map
        return res.reshape(1, -1) if self.n_maps == 1 else res

    @cached_property
    def pwf(self) -> np.ndarray:
        """Calculate pixel window function using pixwin."""
        return hp.sphtfunc.pixwin(
            self.n_side,
            pol=self.n_maps == 3,
        )

    def to_spectra(self, apply_pwf=False) -> PowerSpectra:
        """Convert to PowerSpectra.

        :param bool apply_pwf: if True, correct the spectra for pwf.
        """
        n_maps = self.n_maps
        spectra = self.spectra
        if n_maps == 1:
            names = ['TT']
            if apply_pwf:
                pwf = self.pwf
                # avoid changing self.spectra in-place
                spectra = spectra / np.square(pwf)
        elif n_maps == 3:
            names = ['TT', 'EE', 'BB', 'TE', 'EB', 'TB']
            if apply_pwf:
                t_pwf, p_pwf = self.pwf
                temp = np.empty_like(spectra)
                temp[0] = spectra[0] / np.square(t_pwf)
                temp[[1, 2, 4]] = spectra[[1, 2, 4]] / np.square(p_pwf)
                temp[[3, 5]] = spectra[[3, 5]] / (t_pwf * p_pwf)
                spectra = temp
        else:
            raise ValueError(f'There are {n_maps} maps where I can only understand 1 map or 3 maps.')
        return PowerSpectra(names, spectra, scale='Cl', name=self.name)

    def write(
        self,
        path: Path,
        nest: bool = True,
        dtype: Optional[Union[np.dtype, List[np.dtype]]] = None,
        fits_IDL: bool = True,
        coord: str = 'G',
        partial: bool = False,
        column_names: Optional[Union[str, List[str]]] = None,
        column_units: Union[str, List[str]] = 'uK',
        extra_header: list = [],
        overwrite: bool = False,
    ):
        maps = self.maps
        if nest:
            maps = hp.pixelfunc.reorder(maps, r2n=True)
        hp.write_map(
            path,
            maps,
            nest=nest,
            dtype=dtype,
            fits_IDL=fits_IDL,
            coord=coord,
            partial=partial,
            column_names=column_names,
            column_units=column_units,
            extra_header=extra_header,
            overwrite=overwrite,
        )

    @staticmethod
    def _mollview(
        name: str,
        m: np.ndarray,
        n_std: Optional[int] = None,
        fwhm: Optional[float] = None,
        graticule: bool = True,
        **kwargs
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
        hp.mollview(m, title=name, min=min_, max=max_, **kwargs)
        if graticule:
            hp.graticule()

    def mollview(
        self,
        n_std: Optional[int] = None,
        fwhm: Optional[float] = None,
        graticule: bool = True,
        **kwargs
    ):
        """object wrap of healpy.mollview

        :param n_std: if specified, set the range to be mean ± n_std * std
        :param fwhm: if specified, smooth map to this number of arcmin
        """
        maps_dict = self.maps_dict
        for name, m in maps_dict.items():
            full_name = f'{self.name} {name}' if self.name else name
            self._mollview(full_name, m, n_std=n_std, fwhm=fwhm, graticule=graticule, **kwargs)
        if 'Q' in maps_dict and 'U' in maps_dict:
            name = 'P'
            full_name = f'{self.name} {name}' if self.name else name
            m = np.sqrt(np.square(maps_dict['Q']) + np.square(maps_dict['U']))
            self._mollview(full_name, m, n_std=n_std, fwhm=fwhm, graticule=graticule, **kwargs)


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
    name: str = ''

    def __post_init__(self):
        # make sure l_min and l are self-consistent
        l = self.l
        if l is not None:
            self.l_min = l[0]

        spectra = self.spectra
        assert spectra.ndim == 2
        assert spectra.shape[0] == len(self.names)

        if self.scale == 'Cl':
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
        return pd.DataFrame(self.spectra.T, index=index, columns=pd.Index(self.names, name=self.name))

    @classmethod
    def from_dataframe(
        cls,
        df: pd.DataFrame,
        scale: str = 'Dl',
    ) -> PowerSpectra:
        names = df.columns.tolist()
        spectra = df.values.T
        l = df.index.values
        name = df.columns.name
        return cls(names, spectra, l=l, scale=scale, name=name)

    @classmethod
    def from_class(
        cls,
        path: Optional[Path] = None,
        url: Optional[str] = None,
        camb: bool = True,
        name: str = ''
    ) -> PowerSpectra:
        '''read from CLASS' .dat output

        :param bool camb: if True, assumed ``format = camb`` is used when
        generating the .dat from CLASS

        To self: migrated from abx's convert_theory.py
        '''
        if path is None and url is None:
            raise ValueError('Either path or url has to be specified.')
        if url is not None:
            if path is not None:
                logger.warn('Ignoring path as url is specified.')
            logger.info(f'Downloading and caching using astropy: {url}')
            path = download_file(url, cache=True)
            if not name:
                name = url.split('/')[-1].split('.')[0]
        else:
            path = Path(path)
            if not name:
                name = path.stem
        # read once
        with open(path, 'r') as f:
            text = f.read()

        # get last comment line
        comment = None
        for line in text.split('\n'):
            if line.startswith('#'):
                comment = line
        if comment is None:
            raise ValueError('Cannot find header row from CLASS output, make sure you run CLASS with setting headers = yes.')
        # remove beginning '#'
        comment = comment[1:]
        # parse comment line: get name after ':'
        names = [name.split(':')[1] for name in comment.strip().split()]

        with io.StringIO(text) as f:
            df = pd.read_csv(f, delim_whitespace=True, index_col=0, comment='#', header=None, names=names)
        # to [microK]^2
        if not camb:
            df *= 1.e12
        df.columns.name = name
        return cls.from_dataframe(df)

    @classmethod
    def from_planck_2018(cls) -> PowerSpectra:
        """Use the Planck 2018 best-fit ΛCDM model

        Officially released by Planck up to l equals 2508.
        see http://pla.esac.esa.int/pla/#cosmology and its description
        """
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
        df.columns.name = 'Planck 2018 best-fit ΛCDM model'
        return cls.from_dataframe(df)

    @classmethod
    def from_planck_2018_extended(cls) -> PowerSpectra:
        """Use the Planck 2018 best fit ΛCDM model with extended l-range up to 4902

        This is reproduced using CLASS but with an extended l-range
        """
        return cls.from_class(
            url='https://gist.github.com/ickc/cd6bb1753a44ee09f2d09c49b190d65f/raw/398497217109000c0d670bb57cfcc2cb9a617d63/base_2018_plikHM_TTTEEE_lowl_lowE_lensing_cl_lensed.dat',
            name='Planck 2018 best-fit ΛCDM model extended using CLASS'
        )

    def to_maps(self, nside: int, pixwin=False) -> Maps:
        """Use synfast to generate random maps
        """
        # new order
        cols = ['TT', 'EE', 'BB', 'TE']
        spectra = self.dataframe[cols].values.T
        l = self.l_array
        cl = from_Dl_to_Cl(spectra, l)
        ms = hp.sphtfunc.synfast(cl, nside, pixwin=pixwin, new=True)
        return Maps(CANONICAL_NAME, ms, name=self.name)

    def intersect(self, *others: PowerSpectra) -> List[pd.DataFrame]:
        cols = self.names
        l = self.l_array
        for other in others:
            cols = np.intersect1d(cols, other.names)
            l = np.intersect1d(l, other.l_array)
        res = [self.dataframe.loc[l, cols]]
        for other in others:
            res.append(other.dataframe.loc[l, cols])
        return res

    def compare(
        self,
        *others: PowerSpectra,
        relative: bool = False,
    ) -> pd.DataFrame:
        """Return a tidy DataFrame comparing self and others.
        """
        dfs = self.intersect(*others)
        try:
            if relative:
                dfs = [df / dfs[0] - 1. for df in dfs[1:]]
            df_all = pd.concat(
                dfs,
                keys=(df.columns.name for df in dfs),
                names=['name', 'l']
            )
        except pd.errors.InvalidIndexError:
            counter = defaultdict(int)
            for df in dfs:
                counter[df.columns.name] += 1
            raise ValueError(f"These names are repeated: {', '.join(k for k, v in counter.items() if v > 1)}")
        df_all.columns.name = 'spectra'
        return df_all.stack().to_frame(self.scale).reset_index(level=(0, 1, 2))

    def compare_plot(
        self,
        *others: PowerSpectra,
        show: bool = True,
        relative: bool = False,
    ) -> List[plotly.graph_objs._figure.Figure]:

        df_tidy = self.compare(*others, relative=relative)
        figs = {}
        for name, group in df_tidy.groupby('spectra'):
            title = f'{name}, relative change comparing to {self.name}' if relative else name
            figs[name] = px.line(group, x='l', y='Dl', color='name', title=title)
        if show:
            for fig in figs.values():
                fig.show()
        return figs


def _simmap_planck2018(
    path: Path,
    nside: int,
    *,
    seed: Optional[int] = None,
    nest: bool = True,
    fits_IDL: bool = True,
    coord: str = 'G',
    partial: bool = False,
    column_names: Optional[List[str]] = None,
    column_units: str = 'uK',
    extra_header: List[str] = [],
    overwrite: bool = False,
):
    if seed is not None:
        np.random.seed(seed)
    m = Maps.from_planck_2018(nside)
    m.write(
        path=path,
        nest=nest,
        fits_IDL=fits_IDL,
        coord=coord,
        partial=partial,
        column_names=column_names,
        column_units=column_units,
        extra_header=[tuple(header.split(':')) for header in extra_header],
        overwrite=overwrite,
    )


def simmap_planck2018_cli():
    defopt.run(_simmap_planck2018)
