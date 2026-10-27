"""One visual system for figures and docs: Inter, steel and copper, light and dark SVG.

Series C0 to C2 are steel blues stepped by lightness; C3 is copper, the accent, for poles and the
one thing a figure points at. Figures are drawn in the light colors; the dark SVG swaps each for
its dark counterpart.
"""

import os
import re
from io import StringIO
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.figure import Figure

WIDTH = 6.67  # inches; the docs scale figures to their 46em column

# role: (light, dark)
COLORS = {
    "ink": ("#1b1f24", "#e6e9ed"),
    "secondary": ("#4a525c", "#b3bac3"),
    "muted": ("#6b737d", "#8a929c"),
    "axis": ("#c4cad1", "#3a4048"),
    "grid": ("#e9ecef", "#23272c"),
    "surface": ("#ffffff", "#121417"),
    "panel": ("#f6f7f9", "#181b1f"),
    "brand": ("#a85419", "#e09864"),
    "black": ("#000000", "#e6e9ed"),
}
SERIES = (  # most prominent first in both themes; validated as an ordinal ramp
    ("#1f4e8c", "#c2d9f8"),
    ("#3b78c9", "#78a5ea"),
    ("#86afe3", "#3561aa"),
    COLORS["brand"],
)
LIGHT = {role: light for role, (light, _) in COLORS.items()}

_SWAP = dict([*COLORS.values(), *SERIES])
_HEX = re.compile(r"#[0-9a-f]{6}\b")

for font in (Path(__file__).parent / "fonts").glob("*.ttf"):
    font_manager.fontManager.addfont(str(font))

RC = {
    "axes.axisbelow": True,
    "axes.edgecolor": LIGHT["axis"],
    "axes.facecolor": "none",
    "axes.grid": True,
    "axes.grid.which": "major",
    "axes.labelcolor": LIGHT["secondary"],
    "axes.labelpad": 6,
    "axes.labelsize": 9,
    "axes.linewidth": 0.8,
    "axes.prop_cycle": mpl.cycler(color=[light for light, _ in SERIES]),
    "axes.spines.left": False,
    "axes.spines.right": False,
    "axes.spines.top": False,
    "axes.titlecolor": LIGHT["ink"],
    "axes.titlelocation": "left",
    "axes.titlepad": 22,  # room for a legend between title and plot
    "axes.titlesize": 10,
    "axes.titleweight": "semibold",
    "axes.xmargin": 0,
    "figure.constrained_layout.h_pad": 0.06,
    "figure.constrained_layout.use": True,
    "figure.constrained_layout.w_pad": 0.06,
    "figure.facecolor": "none",
    "figure.figsize": (WIDTH, 4.4),
    "font.family": "Inter",
    "font.size": 9,
    "grid.color": LIGHT["grid"],
    "grid.linewidth": 0.8,
    "legend.borderaxespad": 0.2,
    "legend.columnspacing": 1.6,
    "legend.fontsize": 8.5,
    "legend.frameon": False,
    "legend.handlelength": 1.4,
    "legend.handletextpad": 0.6,
    "legend.labelcolor": LIGHT["secondary"],
    "lines.linewidth": 1.6,
    "lines.markeredgecolor": LIGHT["surface"],
    "lines.markeredgewidth": 1.0,
    "lines.markersize": 4.5,
    "lines.solid_capstyle": "round",
    "lines.solid_joinstyle": "round",
    "savefig.facecolor": "none",
    "svg.fonttype": "path",
    "svg.hashsalt": "pyvectorfit",
    "text.color": LIGHT["ink"],
    "xtick.color": LIGHT["axis"],
    "xtick.labelcolor": LIGHT["muted"],
    "xtick.labelsize": 8,
    "xtick.major.pad": 5,
    "xtick.major.size": 0,
    "xtick.minor.size": 0,
    "ytick.color": LIGHT["axis"],
    "ytick.labelcolor": LIGHT["muted"],
    "ytick.labelsize": 8,
    "ytick.major.pad": 5,
    "ytick.major.size": 0,
    "ytick.minor.size": 0,
}


def reset(gallery_conf=None, fname=None):
    """Apply the style from defaults; sphinx-gallery calls this before each example."""
    mpl.rcdefaults()
    mpl.rcParams.update(RC)


reset()


def figure(rows=1, columns=1, *, height=2.6, **options):
    """Figure and a (rows, columns) array of axes; height is per row, in inches."""
    fig = Figure(figsize=(WIDTH, height * rows))
    return fig, fig.subplots(rows, columns, squeeze=False, **options)


def legend(fig, ax, columns=4):
    """One legend above the axes, at the left, from the labelled marks of ax."""
    fig.legend(*ax.get_legend_handles_labels(), loc="outside upper left", ncols=columns)


def save(fig, stem):
    """Write stem.svg and stem-dark.svg; return both paths."""
    buffer = StringIO()
    fig.savefig(buffer, format="svg", metadata={"Creator": None, "Date": None})
    light = buffer.getvalue()
    paths = stem.with_name(f"{stem.name}.svg"), stem.with_name(f"{stem.name}-dark.svg")
    dark = _HEX.sub(lambda m: _SWAP.get(m[0], m[0]), light)
    for path, svg in zip(paths, (light, dark), strict=True):
        path.write_text(svg, encoding="utf-8", newline="\n")
    return paths


def scrape(block, block_vars, gallery_conf):
    """sphinx-gallery scraper: each new figure as SVG, light and dark, shown with its theme."""
    rst = ""
    for number, path in zip(plt.get_fignums(), block_vars["image_path_iterator"], strict=False):
        fig = plt.figure(number)
        alt = ", ".join(filter(None, (ax.get_title(loc="left") for ax in fig.axes)))
        light, dark = save(fig, Path(path).with_suffix(""))
        rst += "\n.. container:: plot\n"
        for svg, theme in ((light, "light"), (dark, "dark")):
            uri = Path(os.path.relpath(svg, gallery_conf["src_dir"])).as_posix()
            rst += f"\n   .. image:: /{uri}\n      :alt: {alt}\n      :class: only-{theme}\n"
    plt.close("all")
    return rst
