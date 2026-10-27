"""The docs' banner: a line's terminal admittance, its samples, rational model and poles.

Drawn as SVG that animates once, also inside an <img>: the samples sweep in, the model draws
through them and the poles rise beneath their resonances. Without motion it is the final frame.
"""

from pathlib import Path

import numpy as np

import pyvectorfit as vf

from .cases import CASES
from .style import COLORS, SERIES

DOCS = Path(__file__).parents[1] / "docs" / "_static"
BAND = (1.0, 1e4)  # Hz
SAMPLES = 321  # 80 per decade
POLES = 60
ELEMENTS = ((0, 0), (0, 1), (0, 3))  # self and mutual admittance at one end, and end to end
WIDTH, HEIGHT, PAD = 1200, 360, 10  # px
TOP, BOTTOM, BASE = 18, 268, 346  # the curves' extent and the poles' baseline, px
TALL, SHORT = 46, 16  # complex pole pairs and real poles, px

STYLE = """
.grid{{stroke:{grid};stroke-width:1}}
.base{{stroke:{axis};stroke-width:1}}
.curve{{fill:none;stroke-width:2.2;stroke-linecap:round;stroke-linejoin:round}}
.dot{{stroke:{surface};stroke-width:1.8}}
.pole{{fill:{brand}}}
.real{{fill-opacity:.5}}
@media (prefers-reduced-motion:no-preference){{
.dot{{transform-box:fill-box;transform-origin:center;
animation:dot .7s cubic-bezier(.16,1,.3,1) var(--d) both}}
.curve{{stroke-dasharray:1;animation:draw 1.8s cubic-bezier(.45,0,.2,1) var(--d) both}}
.area{{animation:fade 1s ease-out 2s both}}
.pole{{transform-box:fill-box;transform-origin:bottom;
animation:rise .8s cubic-bezier(.16,1,.3,1) var(--d) both}}
}}
@keyframes dot{{from{{opacity:0;transform:scale(.2)}}}}
@keyframes draw{{from{{stroke-dashoffset:1}}to{{stroke-dashoffset:0}}}}
@keyframes fade{{from{{opacity:0}}}}
@keyframes rise{{from{{transform:scaleY(0)}}}}
"""


def terminal(f):
    """The exact terminal admittance of the three-phase line at frequencies f."""
    case = CASES["horizontal"]
    yc, h, _ = vf.line_functions(*case.system.zy(2j * np.pi * f), case.length)
    return vf.terminal_admittance(yc, h)


def fitted():
    """Samples, the rational model fitted to them, and the frequencies its curves are drawn at."""
    f = np.geomspace(*BAND, SAMPLES)
    samples = terminal(f)
    model = vf.fit(f, samples, poles=POLES, symmetric=True).model
    pairs = model.poles[model.poles.imag > 0]
    peaks = [(p.imag + abs(p.real) * np.linspace(-6, 6, 49)) / (2 * np.pi) for p in pairs]
    drawn = np.concatenate([np.geomspace(*BAND, 2401), *peaks])  # each peak's top included
    drawn = np.unique(drawn[(drawn >= BAND[0]) & (drawn <= BAND[1])])
    return f, samples, model, drawn


def across(f):
    """Horizontal position of frequencies f, in px."""
    return PAD + (WIDTH - 2 * PAD) * np.log10(f / BAND[0]) / np.log10(BAND[1] / BAND[0])


def thinned(x, y, step=1.2):
    """The points of a polyline at least step px apart, and its last."""
    kept = [0]
    for k in range(1, len(x)):
        if np.hypot(x[k] - x[kept[-1]], y[k] - y[kept[-1]]) >= step or k == len(x) - 1:
            kept.append(k)
    return x[kept], y[kept]


def path(x, y):
    return "M" + " ".join(f"{a:.1f} {b:.1f}" for a, b in zip(x, y, strict=True))


def draw(f, samples, model, drawn, theme):
    """The banner's SVG; theme 0 is light, 1 dark."""
    color = {role: pair[theme] for role, pair in COLORS.items()}
    curves = model.response(drawn)
    shown = [abs(curves[:, i, j]) for i, j in ELEMENTS]
    low, high = min(map(np.min, shown)) / 1.4, max(map(np.max, shown)) * 1.25

    def down(values):
        return TOP + (BOTTOM - TOP) * np.log10(high / values) / np.log10(high / low)

    x = across(drawn)
    parts = [
        f'<line class="grid" x1="{a:.1f}" y1="0" x2="{a:.1f}" y2="{BASE}"/>'
        for a in across(10.0 ** np.arange(1, 4))
    ]
    edge, top = thinned(x, down(shown[0]))
    floor = BOTTOM + 20
    parts.append(
        f'<path class="area" fill="url(#fill)" d="{path(edge, top)} L{edge[-1]:.1f} {floor} '
        f'L{edge[0]:.1f} {floor}Z"/>'
    )
    order = range(len(ELEMENTS) - 1, -1, -1)  # the self admittance on top
    for k in order:
        line = path(*thinned(x, down(shown[k])))
        parts.append(
            f'<path class="curve" pathLength="1" stroke="{SERIES[k][theme]}" '
            f'style="--d:{0.3 + 0.12 * k:.2f}s" d="{line}"/>'
        )
    dots = across(f)
    for k in order:
        i, j = ELEMENTS[k]
        y = down(abs(samples[:, i, j]))
        circles = "".join(
            f'<circle class="dot" cx="{a:.1f}" cy="{b:.1f}" r="3.4" '
            f'style="--d:{0.1 + 0.9 * a / WIDTH:.2f}s"/>'
            for a, b in zip(dots[::6], y[::6], strict=True)
        )
        parts.append(f'<g fill="{color["secondary"]}">{circles}</g>')
    parts.append(f'<line class="base" x1="0" y1="{BASE + 0.5}" x2="{WIDTH}" y2="{BASE + 0.5}"/>')
    for p in model.poles:
        frequency = (p.imag if p.imag else abs(p.real)) / (2 * np.pi)
        if not BAND[0] <= frequency <= BAND[1]:
            continue
        a, height = across(frequency), TALL if p.imag else SHORT
        parts.append(
            f'<rect class="pole{"" if p.imag else " real"}" x="{a - 1.5:.1f}" '
            f'y="{BASE - height}" width="3" height="{height}" rx="1.5" '
            f'style="--d:{1.3 + 0.8 * a / WIDTH:.2f}s"/>'
        )
    style = STYLE.format(**color).strip()
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {WIDTH} {HEIGHT}" '
        f'width="{WIDTH}" height="{HEIGHT}" role="img" aria-labelledby="title">\n'
        '<title id="title">The terminal admittance of a 100 km line: samples, the rational '
        "model fitted through them, and its poles beneath the resonances</title>\n"
        f"<style>{style}</style>\n"
        '<defs><linearGradient id="fill" x1="0" y1="0" x2="0" y2="1">'
        f'<stop offset="0" stop-color="{SERIES[0][theme]}" stop-opacity=".18"/>'
        f'<stop offset="1" stop-color="{SERIES[0][theme]}" stop-opacity="0"/>'
        "</linearGradient></defs>\n" + "\n".join(parts) + "\n</svg>\n"
    )


def write():
    """Write banner.svg and banner-dark.svg into the docs' static files."""
    data = fitted()
    for theme, name in enumerate(("banner", "banner-dark")):
        (DOCS / f"{name}.svg").write_text(draw(*data, theme), encoding="utf-8", newline="\n")


if __name__ == "__main__":
    write()
