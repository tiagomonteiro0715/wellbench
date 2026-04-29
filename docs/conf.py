"""Sphinx configuration for the wellbench documentation."""

from __future__ import annotations

import importlib
import importlib.util
import sys
import types
from datetime import date
from importlib import metadata
from pathlib import Path


def _bootstrap_wellbench() -> None:
    """Make ``import wellbench`` work for autodoc.

    Prefers the installed package (e.g. ``pip install -e .``). If it isn't
    installed, fall back to loading the modules directly from ``../src/`` and
    registering them under the ``wellbench`` namespace.
    """
    if importlib.util.find_spec("wellbench") is not None:
        return

    src = Path(__file__).resolve().parent.parent / "src"
    if not src.is_dir():
        return

    pkg = types.ModuleType("wellbench")
    pkg.__path__ = [str(src)]
    sys.modules["wellbench"] = pkg

    for module_path in src.glob("*.py"):
        name = module_path.stem
        if name.startswith("_"):
            continue
        spec = importlib.util.spec_from_file_location(
            f"wellbench.{name}", module_path
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules[f"wellbench.{name}"] = module
        spec.loader.exec_module(module)

    init = src / "__init__.py"
    if init.exists():
        spec = importlib.util.spec_from_file_location(
            "wellbench", init, submodule_search_locations=[str(src)]
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules["wellbench"] = module
        spec.loader.exec_module(module)


_bootstrap_wellbench()

# ---------------------------------------------------------------------------
# Project information
# ---------------------------------------------------------------------------
project = "wellbench"
author = "wellbench contributors"
copyright = f"{date.today().year}, {author}"

try:
    release = metadata.version("wellbench")
except metadata.PackageNotFoundError:
    release = "0.1.0"
version = ".".join(release.split(".")[:2])

# ---------------------------------------------------------------------------
# Extensions
# ---------------------------------------------------------------------------
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx_autodoc_typehints",
    "myst_parser",
]

autosummary_generate = True
autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "show-inheritance": True,
    "member-order": "bysource",
}
autodoc_typehints = "description"
napoleon_google_docstring = True
napoleon_numpy_docstring = True

# Mock heavy optional deps so autodoc doesn't need them installed.
autodoc_mock_imports = ["torch", "ctgan"]

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "pandas": ("https://pandas.pydata.org/docs/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
}

# ---------------------------------------------------------------------------
# Sources / output
# ---------------------------------------------------------------------------
source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}
master_doc = "index"
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

html_theme = "furo"
html_static_path = ["_static"]
html_title = f"wellbench {release}"
