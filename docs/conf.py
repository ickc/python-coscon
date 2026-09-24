import sys
import types
from importlib.metadata import version as get_version

project = "coscon"
author = "Kolen Cheung"
copyright = f"2021, {author}"
version = release = get_version("coscon")

extensions = [
    "myst_parser",
    "sphinx.ext.apidoc",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
]
source_suffix = {".md": "markdown", ".rst": "restructuredtext"}
exclude_patterns = ["_build"]

html_theme = "furo"
html_title = f"{project} {version}"

apidoc_modules = [
    {
        "path": "../src/coscon",
        "destination": "api",
        "separate_modules": True,
        "module_first": True,
    },
]
# toast 2 is not installable on the Python versions sphinx supports
autodoc_mock_imports = ["toast"]
# coscon.toast_extras unpacks toast.mpi.get_world() on import, which a mock cannot do
_toast_mpi = types.ModuleType("toast.mpi")
_toast_mpi.get_world = lambda: (None, 1, 0)
sys.modules["toast.mpi"] = _toast_mpi
