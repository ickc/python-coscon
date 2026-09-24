# coscon

[![CI](https://github.com/ickc/python-coscon/actions/workflows/ci.yml/badge.svg)](https://github.com/ickc/python-coscon/actions/workflows/ci.yml)
[![Documentation](https://github.com/ickc/python-coscon/actions/workflows/docs.yml/badge.svg)](https://ickc.github.io/python-coscon)
[![PyPI](https://img.shields.io/pypi/v/coscon.svg)](https://pypi.org/project/coscon)
[![Python versions](https://img.shields.io/pypi/pyversions/coscon.svg)](https://pypi.org/project/coscon)
[![License](https://img.shields.io/pypi/l/coscon.svg)](https://github.com/ickc/python-coscon/blob/master/LICENSE)

Some convenience functions for Cosmology-related analysis.

## Installation

```sh
pip install coscon
```

Optional extras: `coscon[extras]` for colored logs, `coscon[mpi]` for MPI support.

## Python versions and toast

`coscon.toast_helper` and `coscon.toast_extras` depend on [toast](https://github.com/hpc4cmb/toast) 2, which only has wheels for Python<=3.9 on x86_64, and is installed automatically there. toast 3 has an incompatible API and is not supported.

The other modules, such as `coscon.cmb` and `coscon.fits_helper`, support Python>=3.8 including the latest versions.
