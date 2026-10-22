"""Sphinx configuration: Furo, MyST pages, a sphinx-gallery of examples and autodoc."""

import re
import sys
from importlib import metadata
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))

from benchmarks import style

project = "pyvectorfit"
author = "Luke Lowery"
copyright = "2026, Luke Lowery"
release = metadata.version("pyvectorfit")

extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    "sphinx_copybutton",
    "sphinx_gallery.gen_gallery",
]
exclude_patterns = ["_build", "examples"]
myst_enable_extensions = ["colon_fence", "deflist", "dollarmath"]

autodoc_member_order = "bysource"
autodoc_typehints = "description"
python_display_short_literal_types = True
intersphinx_mapping = {
    "numpy": ("https://numpy.org/doc/stable", None),
    "python": ("https://docs.python.org/3", None),
}


sphinx_gallery_conf = {
    "examples_dirs": "examples",
    "gallery_dirs": "gallery",
    "within_subsection_order": "FileNameSortKey",
    "image_scrapers": ("benchmarks.style.scrape",),
    "reset_modules": ("benchmarks.style.reset",),
    "abort_on_example_error": True,
    "download_all_examples": False,
    "show_signature": False,
    "write_computation_times": False,
}
THUMBNAIL = re.compile(
    r"(?m)^( *)\.\. image:: /gallery/images/thumb/sphx_glr_(\w+)_thumb\.svg\n( +):alt:\n"
)


def dark_thumbnails(app, docname, source):
    """Pair each gallery thumbnail with the dark SVG of its example's first figure."""
    if docname == "gallery/index":
        dark = r"\1.. image:: /gallery/images/sphx_glr_\2_001-dark.svg\n\3:alt:\n"
        source[0] = THUMBNAIL.sub(
            rf"\g<0>\3:class: only-light\n\n{dark}\3:class: only-dark\n", source[0]
        )


def setup(app):
    app.connect("source-read", dark_thumbnails)


def palette(theme):
    """Furo colors from the figure tokens: theme 0 is light, 1 dark."""
    color = {role: pair[theme] for role, pair in style.COLORS.items()}
    return {
        "color-brand-primary": color["brand"],
        "color-brand-content": color["brand"],
        "color-brand-visited": color["brand"],
        "color-foreground-primary": color["ink"],
        "color-foreground-secondary": color["secondary"],
        "color-foreground-muted": color["muted"],
        "color-foreground-border": color["axis"],
        "color-background-primary": color["surface"],
        "color-background-secondary": color["panel"],
        "color-background-border": color["grid"],
        "color-api-name": color["ink"],
        "color-api-pre-name": color["secondary"],
        "color-code-background": color["panel"],
        "color-inline-code-background": color["panel"],
    }


GITHUB = (
    '<svg viewBox="0 0 16 16" fill="currentColor"><path fill-rule="evenodd" d="M8 0C3.58 0 0 '
    "3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53"
    "-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21"
    " 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08"
    "-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04"
    " 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29"
    ".25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.013 8.013 0 0 0 16 8c0-4.42"
    '-3.58-8-8-8z"/></svg>'
)

html_theme = "furo"
html_title = "pyvectorfit"
pygments_style = "github-light"
pygments_dark_style = "github-dark"
html_logo = "_static/logo.svg"
html_favicon = "_static/logo.svg"
html_static_path = ["_static", "../benchmarks/fonts"]
html_css_files = ["custom.css"]
html_theme_options = {
    "light_css_variables": palette(0),
    "dark_css_variables": palette(1),
    "top_of_page_buttons": [],
    "footer_icons": [
        {
            "name": "GitHub",
            "url": "https://github.com/BirchfieldResearchGroup/pyvectorfit",
            "html": GITHUB,
        }
    ],
}
