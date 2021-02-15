

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING
from dataclasses import dataclass
import argparse

import h5py
import numpy as np

import toast
from toast.op import Operator
from toast.utils import Logger
from toast.mpi import get_world

if TYPE_CHECKING:
    from typing import Tuple, Optional


def add_crosstalk_args(parser: argparse.ArgumentParser):
    parser.add_argument(
        "--crosstalk-matrix",
        type=Path,
        required=False,
        help="input path to crosstalk matrix in HDF5 container.",
    )
    parser.add_argument(
        "--crosstalk-write-tod-input",
        type=Path,
        required=False,
        help="output path to write TOD input. For debug only.",
    )
    parser.add_argument(
        "--crosstalk-write-tod-output",
        type=Path,
        required=False,
        help="output path to write TOD input. For debug only.",
    )


@dataclass
class OpCrosstalk(Operator):
    """Operator that apply crosstalk matrix to detector ToDs.
    """
    # TODO
    crosstalk_names: np.ndarray['S']
    crosstalk_data: np.ndarray[np.float64]
    crosstalk_write_tod_input_path: Optional[Path] = None
    crosstalk_write_tod_output_path: Optional[Path] = None
    name: str = "crosstalk"
    # comm: Optional[toast.mpi.Comm] = None

    def __post_init__(self):
        self._name = self.name
        self.comm, self.procs, self.rank = get_world()
        self.log = Logger.get()

    @property
    def is_serial(self):
        return self.comm is None

    @staticmethod
    def read_serial(
        path: Path,
    ) -> Tuple[np.ndarray['S'], np.ndarray[np.float64]]:
        with h5py.File(path, 'r') as f:
            names = f["names"][:]
            data = f["data"][:]
        return names, data

    @staticmethod
    def read_mpi(
        path: Path,
        comm: toast.mpi.Comm,
        procs: int,
        rank: int,
    ) -> Tuple[np.ndarray['S'], np.ndarray[np.float64]]:
        log = Logger.get()
        comm, procs, rank = get_world()

        if rank == 0:
            names, data = OpCrosstalk.read_serial(path)
            lengths = np.array([names.size, names.dtype.itemsize], dtype=np.int64)
            # cast to int for boardcasting
            names_int = names.view(np.uint8)
            # data = data.astype(np.float64)
            data = data.view(np.float64)
        else:
            lengths = np.empty(2, dtype=np.int64)
        comm.Bcast(lengths, root=0)
        log.debug(f'crosstalk: Rank {rank} receives lengths {lengths}')
        if rank != 0:
            n = lengths[0]
            name_len = lengths[1]
            names_int = np.empty(n * name_len, dtype=np.uint8)
            data = np.empty((n, n), dtype=np.float64)
        comm.Bcast(names_int, root=0)
        if rank != 0:
            names = names_int.view(f'S{name_len}')
        log.debug(f'crosstalk: Rank {rank} receives names {names}')
        comm.Bcast(data, root=0)
        log.debug(f'crosstalk: Rank {rank} receives data {data}')
        return names, data

    @classmethod
    def read(
        cls,
        args: argparse.Namespace,
        name: str = "crosstalk",
    ) -> OpCrosstalk:
        path = args.crosstalk_matrix
        comm, procs, rank = get_world()
        names, data = cls.read_serial(path) if procs == 1 else cls.read_mpi(path, comm, procs, rank)
        return cls(
            names,
            data,
            crosstalk_write_tod_input_path=args.crosstalk_write_tod_input,
            crosstalk_write_tod_output_path=args.crosstalk_write_tod_output,
            name=name,
        )

    def exec_serial(
        self,
        data: toast.dist.Data,
        signal_name: str,
        debug: bool = True,  # TODO
    ):
        raise NotImplementedError

    def exec_mpi(
        self,
        data: toast.dist.Data,
        signal_name: str,
        debug: bool = True,  # TODO
    ):
        log = self.log
        comm = self.comm
        procs = self.procs
        rank = self.rank
        name = self.name
        names = [name.decode() for name in self.crosstalk_names]
        crosstalk_data = self.crosstalk_data
        n = len(names)

        for obs in data.obs:
            tod = obs["tod"]
            n_samples = tod.total_samples

            # this is easier to understand and shorter
            # but uses allgather instead of the more efficient Allgather
            # construct detector LUT
            # local_dets = tod.local_dets
            # global_dets = comm.allgather(local_dets)
            # det_lut = {}
            # for i, dets in enumerate(global_dets):
            #     for det in dets:
            #         det_lut[det] = i
            # log.debug(f'dets LUT: {dets_lut}')

            local_dets = set(tod.local_dets)

            local_has_det = tod.cache.create(f"{name}_local_has_det_{rank}", np.uint8, (n,)).view(np.bool)
            for i, name in enumerate(names):
                if name in local_dets:
                    local_has_det[i] = True

            global_has_det = tod.cache.create(f"{name}_global_has_det_{rank}", np.uint8, (procs, n)).view(np.bool)
            comm.Allgather(local_has_det, global_has_det)

            if debug:
                np.testing.assert_array_equal(local_has_det, global_has_det[rank])
            del local_has_det
            tod.cache.destroy(f"{name}_local_has_det_{rank}")

            det_lut = {}
            for i in range(procs):
                for j in range(n):
                    if global_has_det[i, j]:
                        det_lut[names[j]] = i
            del global_has_det
            tod.cache.destroy(f"{name}_global_has_det_{rank}")

            log.debug(f'dets LUT: {det_lut}')

            if debug:
                for name in local_dets:
                    assert det_lut[name] == rank

            row_local_total = tod.cache.create(f"{name}_row_local_total_{rank}", np.float64, (n_samples,))
            # row-loop
            # potentially the tod can have more detectors than OpCrosstalk.crosstalk_names has
            # and they will be skipped
            for name, row in zip(names, crosstalk_data):
                crosstalk_row_dict = {name_col: m_ij for name_col, m_ij in zip(names, row)}
                rank_owner = det_lut[name]
                # assume each process must have at least one detector
                row_local_total[:] = 0.
                for local_name in local_dets:
                    row_local_total += crosstalk_row_dict[local_name] * tod.cache.reference(f"{signal_name}_{local_name}")
                if rank == rank_owner:
                    row_global_total = tod.cache.create(f"{name}_{name}", np.float64, (n_samples,))
                    comm.Reduce(row_local_total, row_global_total, root=rank_owner)
                else:
                    comm.Reduce(row_local_total, None, root=rank_owner)
            del row_local_total
            tod.cache.destroy(f"{name}_row_local_total_{rank}")
            # overwrite original tod from cache
            for name in local_dets:
                tod.cache.destroy(f"{signal_name}_{name}")
                tod.cache.add_alias(f"{signal_name}_{name}", f"{name}_{name}")
                # tod.cache.put(f"{signal_name}_{name}", data=tod.cache.reference(f"{name}_{name}"), replace=False)
                # tod.write(detector=name, data=tod.cache.reference(f"{name}_{name}"))
            # tod.cache.clear(pattern=f'crosstalk_.+')

    def exec(
        self,
        data: toast.dist.Data,
        signal_name: str,
        debug: bool = True,  # TODO
    ):
        self.exec_serial(data, signal_name, debug=debug) if self.is_serial else self.exec_mpi(data, signal_name, debug=debug)
