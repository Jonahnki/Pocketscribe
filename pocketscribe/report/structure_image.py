"""Offline structure rendering for the report.

A deliberate engineering choice: this module projects the Cα trace to inline SVG in
pure Python rather than driving py3Dmol or NGL. Those are JavaScript viewers, and
rasterising them to PNG needs a headless browser at report time -- a heavy, fragile
dependency for an academic install, and one that would break the promise that the
report is a single self-contained file produced offline. A 2D projection is enough
for what this figure has to do: show *where* on the fold a pocket sits and how
confidently that part of the model was predicted.

Pocket-lining residues are coloured by their own confidence band, reusing the
validated palette from :mod:`pocketscribe.report.plots`. That way the pocket image
answers the question the report exists to raise -- is this cavity in a
well-predicted part of the structure? -- rather than needing a separate legend.
"""

from __future__ import annotations

from html import escape

import numpy as np

from ..models import ConfidenceProfile, Pocket
from ..parsing import ParsedStructure
from .plots import BAND_COLORS, INK_MUTED, INK_SECONDARY, SURFACE, band_of

BACKBONE_COLOR = "#c3c2b7"
BACKBONE_FAR_COLOR = "#dedcd3"


def render_pocket_image(
    parsed: ParsedStructure,
    pocket: Pocket,
    profile: ConfidenceProfile | None = None,
    width: int = 420,
    height: int = 320,
) -> str:
    """Render the fold with one pocket highlighted, as inline SVG.

    The viewing direction is chosen by principal component analysis of the Cα cloud,
    so the projection shows the structure's largest two dimensions and the fold is as
    unfolded on the page as an orthographic projection allows. The same orientation is
    used for every pocket in a run, which makes the images comparable.
    """
    keys, coords = parsed.ca_coords()
    if len(coords) < 3:
        return '<p class="empty">Too few C-alpha atoms to render a structure image.</p>'

    projected, depth = _project(coords)
    scaled, _ = _fit_to_viewport(projected, width, height, margin=24)

    index_of = {key: position for position, key in enumerate(keys)}

    parts: list[str] = [
        f'<svg class="viz structure" role="img" viewBox="0 0 {width} {height}" '
        f'width="100%" preserveAspectRatio="xMidYMid meet" '
        f'aria-label="C-alpha trace of the model with pocket {pocket.rank} highlighted; '
        f'{len(pocket.residues)} lining residues are drawn as filled circles coloured by '
        'prediction confidence.">',
        f'<rect x="0" y="0" width="{width}" height="{height}" fill="{SURFACE}"/>',
    ]

    # Backbone, drawn per chain so separate chains are not joined by a spurious line.
    # Segments behind the structure's midplane are drawn first and lighter, which gives
    # enough depth cue to read the fold without any shading machinery.
    midplane = float(np.median(depth))
    for chain_segments in _chain_segments(keys):
        far: list[str] = []
        near: list[str] = []
        for start, end in chain_segments:
            x1, y1 = scaled[start]
            x2, y2 = scaled[end]
            segment_depth = (depth[start] + depth[end]) / 2.0
            line = (
                f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
                f'stroke-linecap="round" stroke-width="2" stroke="%s"/>'
            )
            if segment_depth < midplane:
                far.append(line % BACKBONE_FAR_COLOR)
            else:
                near.append(line % BACKBONE_COLOR)
        parts.extend(far)
        parts.extend(near)

    # Pocket residues.
    for residue in pocket.residues:
        position = index_of.get(residue.key)
        if position is None:
            continue
        x, y = scaled[position]
        value = residue.confidence
        if value is None and profile is not None:
            value = profile.value_for(residue.chain, residue.resseq, residue.icode)
        color = BAND_COLORS[band_of(value)] if value is not None else INK_MUTED
        tooltip = (
            f"{residue.label}"
            + (f" - pLDDT {value:.1f}" if value is not None else " - confidence unknown")
        )
        parts.append(
            f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4.5" fill="{color}" '
            f'stroke="{SURFACE}" stroke-width="2">'
            f"<title>{escape(tooltip)}</title></circle>"
        )

    # Pocket centroid, projected through the same transform.
    if pocket.centroid is not None:
        centroid_projected, _ = _project(coords, point=np.array(pocket.centroid, dtype=float))
        centroid_scaled = _apply_viewport(centroid_projected, projected, width, height, margin=24)
        parts.append(
            f'<circle cx="{centroid_scaled[0]:.1f}" cy="{centroid_scaled[1]:.1f}" r="11" '
            f'fill="none" stroke="{INK_SECONDARY}" stroke-width="1.5" '
            'stroke-dasharray="3 2"/>'
        )

    parts.append(
        f'<text x="12" y="{height - 10}" font-size="11" fill="{INK_MUTED}">'
        f"Pocket {pocket.rank} - {len(pocket.residues)} lining residues, "
        "C-alpha trace, orthographic projection</text>"
    )
    parts.append("</svg>")
    return "".join(parts)


def _project(coords: np.ndarray, point: np.ndarray | None = None):
    """Project 3D coordinates onto their two principal axes.

    Returns the 2D projection and the depth along the third axis. When ``point`` is
    given, that single point is projected through the *same* transform, so a pocket
    centroid lands in the right place relative to the trace.
    """
    centre = coords.mean(axis=0)
    centred = coords - centre
    # Right singular vectors are the principal axes of the point cloud.
    _, _, axes = np.linalg.svd(centred, full_matrices=False)
    if point is not None:
        projected_point = (point - centre) @ axes.T
        return projected_point[:2], projected_point[2]
    projected = centred @ axes.T
    return projected[:, :2], projected[:, 2]


def _fit_to_viewport(projected: np.ndarray, width: int, height: int, margin: int):
    """Scale and centre a 2D projection inside the viewport, preserving aspect ratio."""
    lower = projected.min(axis=0)
    upper = projected.max(axis=0)
    span = np.maximum(upper - lower, 1e-6)
    scale = min((width - 2 * margin) / span[0], (height - 2 * margin) / span[1])
    offset = np.array([width, height]) / 2.0 - (lower + upper) / 2.0 * scale * np.array([1, -1])
    scaled = projected * scale * np.array([1, -1]) + offset
    return scaled, (scale, offset)


def _apply_viewport(
    point: np.ndarray, projected: np.ndarray, width: int, height: int, margin: int
) -> np.ndarray:
    """Apply the viewport transform derived from ``projected`` to a single point."""
    _, (scale, offset) = _fit_to_viewport(projected, width, height, margin)
    return point * scale * np.array([1, -1]) + offset


def _chain_segments(keys: list[tuple[str, int, str]]) -> list[list[tuple[int, int]]]:
    """Consecutive-residue index pairs, grouped per chain and broken at numbering gaps."""
    groups: list[list[tuple[int, int]]] = []
    current: list[tuple[int, int]] = []
    for position in range(len(keys) - 1):
        chain_a, resseq_a, _ = keys[position]
        chain_b, resseq_b, _ = keys[position + 1]
        if chain_a != chain_b:
            if current:
                groups.append(current)
                current = []
            continue
        if resseq_b - resseq_a > 4:
            continue  # a large numbering gap is not a bond
        current.append((position, position + 1))
    if current:
        groups.append(current)
    return groups
