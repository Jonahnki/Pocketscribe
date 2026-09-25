"""Inline SVG figures for the report.

Everything here emits SVG markup as a string, embedded directly in the single-file
HTML report. No plotting library, no PNG rasterisation, no external asset: the report
has to be one file that opens offline on any machine and survives being emailed.

Colour
------
Confidence is an ordered four-band scale, and structural biologists read the
AlphaFold band convention (blue = reliable, orange = unreliable) instantly, so the
hue order is preserved. The exact steps were re-chosen for a light figure surface and
validated: lightness band, chroma floor, adjacent CVD separation (worst pair
ΔE 17.1 deutan), normal-vision separation (worst pair ΔE 19.7) and >= 3:1 contrast
against the ``#fcfcfb`` figure plane all pass.

The figure plane stays light in both page themes. That is a deliberate choice rather
than an omission: re-stepping this palette for a dark surface could not clear the
normal-vision floor between the amber and orange bands without abandoning the domain
colour convention, and a report figure that keeps printing colours in a dark-themed
viewer is the behaviour scientific readers expect. Colour is also fully redundant
here -- band membership is readable from the y position and the labelled threshold
lines alone.
"""

from __future__ import annotations

from html import escape

from ..models import ConfidenceProfile

#: Validated band palette for the light figure surface.
BAND_COLORS = {
    "very_high": "#0B4FBF",
    "confident": "#4C93C8",
    "low": "#B8860B",
    "very_low": "#B0431A",
}

BAND_LABELS = {
    "very_high": "Very high (pLDDT >= 90)",
    "confident": "Confident (70-90)",
    "low": "Low (50-70)",
    "very_low": "Very low (< 50)",
}

# Chart chrome.
SURFACE = "#fcfcfb"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
BASELINE = "#c3c2b7"

#: Above this residue count, per-residue hover titles are dropped to keep the single
#: HTML file a reasonable size.
MAX_HOVER_TITLES = 2500


def band_of(value: float) -> str:
    """Which confidence band a pLDDT value falls into."""
    if value >= 90:
        return "very_high"
    if value >= 70:
        return "confident"
    if value >= 50:
        return "low"
    return "very_low"


def confidence_plot(
    profile: ConfidenceProfile,
    width: int = 940,
    height: int = 280,
    title: str = "Per-residue prediction confidence",
) -> str:
    """Render the per-residue confidence plot as inline SVG.

    A magnitude-along-an-ordered-index plot, drawn as a contiguous filled area coloured
    by band. Threshold lines at 70 and 50 are labelled, so the two decision boundaries
    that matter for interpreting a pocket are readable without consulting the legend.
    """
    residues = sorted(profile.residues, key=lambda r: (r.chain, r.resseq, r.icode))
    if not residues:
        return '<p class="empty">No per-residue confidence available to plot.</p>'

    margin_left, margin_right = 52, 16
    margin_top, margin_bottom = 16, 46
    plot_width = width - margin_left - margin_right
    plot_height = height - margin_top - margin_bottom
    n = len(residues)
    bar_width = plot_width / n

    def x_of(index: int) -> float:
        return margin_left + index * bar_width

    def y_of(value: float) -> float:
        return margin_top + plot_height * (1.0 - min(max(value, 0.0), 100.0) / 100.0)

    parts: list[str] = [
        f'<svg class="viz" role="img" viewBox="0 0 {width} {height}" '
        f'width="100%" preserveAspectRatio="xMidYMid meet" '
        f'aria-label="{escape(title)}: pLDDT for {n} residues along the sequence.">',
        f'<rect x="0" y="0" width="{width}" height="{height}" fill="{SURFACE}"/>',
    ]

    # Horizontal gridlines. 50 and 70 are drawn as labelled thresholds below.
    for value in (25, 90, 100):
        y = y_of(value)
        parts.append(
            f'<line x1="{margin_left}" y1="{y:.1f}" x2="{margin_left + plot_width}" '
            f'y2="{y:.1f}" stroke="{GRIDLINE}" stroke-width="1"/>'
        )
        parts.append(
            f'<text x="{margin_left - 8}" y="{y + 4:.1f}" text-anchor="end" '
            f'font-size="11" fill="{INK_MUTED}">{value}</text>'
        )

    # The data. Contiguous bars form a banded area; at this density a gap between
    # marks would read as missing data rather than as separation.
    include_titles = n <= MAX_HOVER_TITLES
    for index, residue in enumerate(residues):
        band = band_of(residue.value)
        y = y_of(residue.value)
        bar_height = margin_top + plot_height - y
        rect = (
            f'<rect x="{x_of(index):.2f}" y="{y:.2f}" '
            f'width="{max(bar_width, 0.6):.2f}" height="{max(bar_height, 0.5):.2f}" '
            f'fill="{BAND_COLORS[band]}"'
        )
        if include_titles:
            tooltip = (
                f"{residue.label} - pLDDT {residue.value:.1f} "
                f"({BAND_LABELS[band].split(' (')[0]})"
            )
            parts.append(f"{rect}><title>{escape(tooltip)}</title></rect>")
        else:
            parts.append(f"{rect}/>")

    # Labelled decision thresholds, drawn over the data.
    for value, label in ((70.0, "70"), (50.0, "50")):
        y = y_of(value)
        parts.append(
            f'<line x1="{margin_left}" y1="{y:.1f}" x2="{margin_left + plot_width}" '
            f'y2="{y:.1f}" stroke="{INK_SECONDARY}" stroke-width="1.5" '
            f'stroke-dasharray="5 3"/>'
        )
        parts.append(
            f'<text x="{margin_left - 8}" y="{y + 4:.1f}" text-anchor="end" '
            f'font-size="11" font-weight="600" fill="{INK_SECONDARY}">{label}</text>'
        )

    # Chain boundaries, when there is more than one chain.
    boundaries = _chain_boundaries(residues)
    for index, chain_id in boundaries:
        x = x_of(index)
        parts.append(
            f'<line x1="{x:.1f}" y1="{margin_top}" x2="{x:.1f}" '
            f'y2="{margin_top + plot_height}" stroke="{INK_MUTED}" stroke-width="1"/>'
        )
        parts.append(
            f'<text x="{x + 4:.1f}" y="{margin_top + 12}" font-size="10" '
            f'fill="{INK_MUTED}">chain {escape(chain_id)}</text>'
        )

    # Axes.
    baseline_y = margin_top + plot_height
    parts.append(
        f'<line x1="{margin_left}" y1="{baseline_y}" x2="{margin_left + plot_width}" '
        f'y2="{baseline_y}" stroke="{BASELINE}" stroke-width="1"/>'
    )
    for index, label in _x_ticks(residues):
        x = x_of(index)
        parts.append(
            f'<line x1="{x:.1f}" y1="{baseline_y}" x2="{x:.1f}" '
            f'y2="{baseline_y + 4}" stroke="{BASELINE}" stroke-width="1"/>'
        )
        parts.append(
            f'<text x="{x:.1f}" y="{baseline_y + 17}" text-anchor="middle" '
            f'font-size="11" fill="{INK_MUTED}">{escape(label)}</text>'
        )

    parts.append(
        f'<text x="{margin_left + plot_width / 2:.0f}" y="{height - 6}" '
        f'text-anchor="middle" font-size="11" fill="{INK_SECONDARY}">'
        "Residue (sequence order)</text>"
    )
    parts.append(
        f'<text x="14" y="{margin_top + plot_height / 2:.0f}" font-size="11" '
        f'fill="{INK_SECONDARY}" transform="rotate(-90 14 '
        f'{margin_top + plot_height / 2:.0f})" text-anchor="middle">pLDDT</text>'
    )
    parts.append("</svg>")
    return "".join(parts)


def _chain_boundaries(residues) -> list[tuple[int, str]]:
    """Indices at which a new chain starts, excluding the first."""
    boundaries: list[tuple[int, str]] = []
    previous: str | None = None
    for index, residue in enumerate(residues):
        if previous is not None and residue.chain != previous:
            boundaries.append((index, residue.chain))
        previous = residue.chain
    return boundaries


def _x_ticks(residues, max_ticks: int = 8) -> list[tuple[int, str]]:
    """Evenly spaced ticks labelled with the residue number at that position."""
    n = len(residues)
    step = max(1, n // max_ticks)
    ticks: list[tuple[int, str]] = []
    for index in range(0, n, step):
        ticks.append((index, str(residues[index].resseq)))
    return ticks


def confidence_legend(counts: dict[str, int], total: int) -> str:
    """Legend swatches with residue counts, so identity is never colour-alone."""
    items: list[str] = []
    for band in ("very_high", "confident", "low", "very_low"):
        count = counts.get(band, 0)
        percentage = (count / total * 100.0) if total else 0.0
        items.append(
            '<li class="legend-item">'
            f'<span class="swatch" style="background:{BAND_COLORS[band]}"></span>'
            f'<span class="legend-label">{escape(BAND_LABELS[band])}</span>'
            f'<span class="legend-value">{count} ({percentage:.0f}%)</span>'
            "</li>"
        )
    return f'<ul class="legend">{"".join(items)}</ul>'


def caveat_meter(fraction_below_70: float, width: int = 180, height: int = 10) -> str:
    """A small proportion bar showing how much of a pocket is low-confidence."""
    fraction = min(max(fraction_below_70, 0.0), 1.0)
    filled = width * fraction
    return (
        f'<svg class="meter" viewBox="0 0 {width} {height}" width="{width}" '
        f'height="{height}" role="img" aria-label="'
        f'{fraction * 100:.0f}% of lining residues below pLDDT 70">'
        f'<rect x="0" y="0" width="{width}" height="{height}" rx="3" fill="#e1e0d9"/>'
        f'<rect x="0" y="0" width="{filled:.1f}" height="{height}" rx="3" '
        f'fill="{BAND_COLORS["low"] if fraction < 0.5 else BAND_COLORS["very_low"]}"/>'
        "</svg>"
    )
