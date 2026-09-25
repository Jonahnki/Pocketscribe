"""HTML report rendering.

Produces one self-contained file: CSS inlined, figures as inline SVG, no scripts, no
external requests. That constraint is not aesthetic. These reports get emailed to
collaborators, attached to grant applications and opened years later on machines that
no longer have the tool installed, and a report that needs a CDN to render is a report
that eventually stops rendering.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, select_autoescape

from ..citations import Citation, citations_for_report
from ..confidence import MIN_REGION_LENGTH, confidence_band_counts
from ..errors import ReportRenderError
from ..models import AnalysisResult, ArchitectureFamily, Pocket, SupportTier
from ..parsing import ParsedStructure
from .plots import BAND_LABELS, caveat_meter, confidence_legend, confidence_plot
from .structure_image import render_pocket_image

TEMPLATE_DIR = Path(__file__).parent / "templates"

_TIER_LABELS = {
    SupportTier.TIER_1: "Tier 1 - full support",
    SupportTier.TIER_2_REFERENCE: "Tier 2 - reference adapter",
    SupportTier.TIER_2_STUB: "Tier 2 - stub",
}

_FAMILY_LABELS = {
    ArchitectureFamily.MSA_COEVOLUTION: "MSA / co-evolution",
    ArchitectureFamily.SINGLE_SEQUENCE_LM: "Single-sequence language model",
    ArchitectureFamily.UNKNOWN: "Unknown",
}


@dataclass
class PocketImage:
    """One rendered pocket figure, paired with the pocket it shows."""

    pocket: Pocket
    svg: str


def _environment() -> Environment:
    return Environment(
        loader=FileSystemLoader(str(TEMPLATE_DIR)),
        autoescape=select_autoescape(["html"]),
        trim_blocks=True,
        lstrip_blocks=True,
    )


def render_report(
    result: AnalysisResult,
    parsed_structures: list[ParsedStructure],
    output_path: str | Path,
    max_images: int = 3,
) -> Path:
    """Render ``result`` to a single HTML file and return its path."""
    output_path = Path(output_path)
    environment = _environment()

    try:
        template = environment.get_template("report.html")
        stylesheet = (TEMPLATE_DIR / "style.css").read_text()
    except Exception as exc:
        raise ReportRenderError(f"Could not load report templates: {exc}") from exc

    primary = result.primary_structure
    primary_pockets = result.primary_pockets
    band_counts = confidence_band_counts(primary.confidence)

    images: list[PocketImage] = []
    if parsed_structures:
        for pocket in primary_pockets.top(max_images):
            images.append(
                PocketImage(
                    pocket=pocket,
                    svg=render_pocket_image(
                        parsed_structures[0], pocket, primary.confidence
                    ),
                )
            )

    citations: list[Citation] = citations_for_report(
        source_keys=[s.source.citation_key for s in result.structures if s.source.citation_key],
        used_fpocket=primary_pockets.backend_is_fpocket,
        used_consensus=result.consensus is not None,
    )

    try:
        html = template.render(
            result=result,
            primary=primary,
            primary_pockets=primary_pockets,
            top_pocket=primary_pockets.pockets[0] if primary_pockets.pockets else None,
            tool_version=result.tool_version,
            generated_at=result.generated_at,
            stylesheet=stylesheet,
            confidence_plot_svg=confidence_plot(primary.confidence),
            confidence_legend_svg=confidence_legend(
                band_counts, len(primary.confidence.residues)
            ),
            band_counts=band_counts,
            band_rows=[
                ("very_high", BAND_LABELS["very_high"]),
                ("confident", BAND_LABELS["confident"]),
                ("low", BAND_LABELS["low"]),
                ("very_low", BAND_LABELS["very_low"]),
            ],
            pocket_images=images,
            narrative_paragraphs=_split_paragraphs(result.narrative.text),
            citations=citations,
            min_region_length=MIN_REGION_LENGTH,
            tier_label=lambda tier: _TIER_LABELS.get(tier, tier.value),
            family_label=lambda family: _FAMILY_LABELS.get(family, family.value),
            caveat_meter=caveat_meter,
        )
    except Exception as exc:
        raise ReportRenderError(f"Report rendering failed: {exc}") from exc

    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(html, encoding="utf-8")
    except OSError as exc:
        raise ReportRenderError(f"Could not write report to '{output_path}': {exc}") from exc

    return output_path


def _split_paragraphs(text: str) -> list[str]:
    """Split narrative prose into paragraphs.

    Returned unescaped: the template has autoescaping on, so escaping here would show
    entities as literal text. The narrative is model-generated, so it is rendered as
    text and never as markup.
    """
    if not text:
        return []
    blocks = [block.strip() for block in text.replace("\r\n", "\n").split("\n\n")]
    return [block for block in blocks if block]
