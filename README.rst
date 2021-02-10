========
Overview
========

.. start-badges

.. list-table::
    :stub-columns: 1

    * - docs
      - |docs|
    * - tests
      - |
        | |coveralls| |codecov|
        | |landscape| |scrutinizer| |codacy| |codeclimate|
    * - package
      - | |version| |wheel| |supported-versions| |supported-implementations|
        | |commits-since|
.. |docs| image:: https://readthedocs.org/projects/python-coscon/badge/?style=flat
    :target: https://readthedocs.org/projects/python-coscon
    :alt: Documentation Status

.. |coveralls| image:: https://coveralls.io/repos/ickc/python-coscon/badge.svg?branch=master&service=github
    :alt: Coverage Status
    :target: https://coveralls.io/r/ickc/python-coscon

.. |codecov| image:: https://codecov.io/gh/ickc/python-coscon/branch/master/graphs/badge.svg?branch=master
    :alt: Coverage Status
    :target: https://codecov.io/github/ickc/python-coscon

.. |landscape| image:: https://landscape.io/github/ickc/python-coscon/master/landscape.svg?style=flat
    :target: https://landscape.io/github/ickc/python-coscon/master
    :alt: Code Quality Status

.. |codacy| image:: https://img.shields.io/codacy/grade/013d60298aae4c53b33916c44a6675ab.svg
    :target: https://www.codacy.com/app/ickc/python-coscon
    :alt: Codacy Code Quality Status

.. |codeclimate| image:: https://codeclimate.com/github/ickc/python-coscon/badges/gpa.svg
   :target: https://codeclimate.com/github/ickc/python-coscon
   :alt: CodeClimate Quality Status

.. |version| image:: https://img.shields.io/pypi/v/coscon.svg
    :alt: PyPI Package latest release
    :target: https://pypi.org/project/coscon

.. |wheel| image:: https://img.shields.io/pypi/wheel/coscon.svg
    :alt: PyPI Wheel
    :target: https://pypi.org/project/coscon

.. |supported-versions| image:: https://img.shields.io/pypi/pyversions/coscon.svg
    :alt: Supported versions
    :target: https://pypi.org/project/coscon

.. |supported-implementations| image:: https://img.shields.io/pypi/implementation/coscon.svg
    :alt: Supported implementations
    :target: https://pypi.org/project/coscon

.. |commits-since| image:: https://img.shields.io/github/commits-since/ickc/python-coscon/v0.1.0.svg
    :alt: Commits since latest release
    :target: https://github.com/ickc/python-coscon/compare/v0.1.0...master


.. |scrutinizer| image:: https://img.shields.io/scrutinizer/quality/g/ickc/python-coscon/master.svg
    :alt: Scrutinizer Status
    :target: https://scrutinizer-ci.com/g/ickc/python-coscon/


.. end-badges

Some convenience functions for Cosmology-related analysis.

* Free software: BSD 3-Clause License

Installation
============

::

    pip install coscon

You can also install the in-development version with::

    pip install https://github.com/ickc/python-coscon/archive/master.zip


Documentation
=============


https://python-coscon.readthedocs.io/


Development
===========

To run all the tests run::

    tox

Note, to combine the coverage data from all the tox environments run:

.. list-table::
    :widths: 10 90
    :stub-columns: 1

    - - Windows
      - ::

            set PYTEST_ADDOPTS=--cov-append
            tox

    - - Other
      - ::

            PYTEST_ADDOPTS=--cov-append tox
