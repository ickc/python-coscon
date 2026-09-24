# Revision history for `coscon`

- v0.2.0:
    - maintenance:
        - support numba>=0.59 (`numba.generated_jit` was removed), numpy 2, pandas 3, and Python up to 3.14, while keeping support for Python>=3.8.
        - toast 2 (`toast-cmb`) is only required on Python<3.10, where it is installable. `coscon.toast_helper` and `coscon.toast_extras` need it.
        - require numba<0.59 on Python<3.10, as numba-quaternion<0.3 needs it.
        - declare scipy as a dependency; remove upper bounds of dependencies.
        - move the `tests` and `docs` extras to dependency groups.
    - breaking changes:
        - `toast_helper`: rename `reorder_to` and `reorder_to_from_csv` to `reorder_TB` and `reorder_TB_from_csv`.
        - `toast_extras`: `OpCrosstalk` takes multiple crosstalk matrices, see `SimpleCrosstalkMatrix`; remove its methods that read and save TOD.
    - new features:
        - `cmb`: `PowerSpectraMatrix`, `PowerSpectra.rotate`, `get_spectrum`, `pwf`, `from_planck`, `from_planck_2015`.
        - `toast_helper`: quaternion support via numba-quaternion, `dist_spherical_pairwise`, `reorder_QUAB`.
- v0.1.1: upgrade numba_quaternion to v0.2.0.
- v0.1.0: first release and proof of concept.
