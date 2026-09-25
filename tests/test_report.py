"""Report rendering: structure, required sections, and the claims that must not appear."""

from __future__ import annotations

from html.parser import HTMLParser

import pytest

from pocketscribe.config import RunConfig
from pocketscribe.pipeline import StructureInput, run_pipeline
from pocketscribe.report.plots import band_of, confidence_plot
from pocketscribe.report.render import render_report

REQUIRED_SECTIONS = [
    "summary",
    "input",
    "confidence",
    "pockets",
    "images",
    "narrative",
    "methods",
    "disclaimer",
    "cite",
]


class _WellFormednessCheck(HTMLParser):
    """Checks that every element opened is closed, in order."""

    VOID = {"br", "img", "meta", "link", "input", "hr", "area", "base", "col", "source"}
    SVG_SELF_CLOSING = {"rect", "line", "circle", "path", "polyline", "polygon", "use"}

    def __init__(self) -> None:
        super().__init__()
        self.stack: list[str] = []
        self.errors: list[str] = []

    def handle_starttag(self, tag, attrs):
        if tag in self.VOID or tag in self.SVG_SELF_CLOSING:
            return
        self.stack.append(tag)

    def handle_endtag(self, tag):
        if tag in self.VOID or tag in self.SVG_SELF_CLOSING:
            return
        if not self.stack:
            self.errors.append(f"closing </{tag}> with nothing open")
        elif self.stack[-1] == tag:
            self.stack.pop()
        else:
            self.errors.append(f"expected </{self.stack[-1]}>, found </{tag}>")


def _render(tmp_path, inputs, **config_overrides):
    config = RunConfig()
    for key, value in config_overrides.items():
        section, _, field = key.partition("__")
        setattr(getattr(config, section), field, value)
    config.narrative.enabled = False

    result, parsed = run_pipeline(
        inputs=inputs,
        config=config,
        md_output_dir=tmp_path / "md",
        allow_backend_fallback=True,
    )
    path = render_report(result, parsed, tmp_path / "report.html")
    return result, path.read_text()


@pytest.fixture(scope="module")
def single_report(tmp_path_factory, request):
    tmp_path = tmp_path_factory.mktemp("single")
    af2_path = request.getfixturevalue("af2_path")
    return _render(
        tmp_path, [StructureInput(path=af2_path, source_id="alphafold2", label=af2_path.name)]
    )


@pytest.fixture(scope="module")
def consensus_report(tmp_path_factory, request):
    tmp_path = tmp_path_factory.mktemp("consensus")
    paths = [
        (request.getfixturevalue("af2_path"), "alphafold2"),
        (request.getfixturevalue("openfold_path"), "openfold"),
        (request.getfixturevalue("esmfold_path"), "esmfold"),
    ]
    return _render(
        tmp_path,
        [StructureInput(path=path, source_id=source, label=path.name) for path, source in paths],
    )


# --------------------------------------------------------------------------------------
# Structure
# --------------------------------------------------------------------------------------


def test_report_is_well_formed_html(single_report):
    _, html = single_report
    checker = _WellFormednessCheck()
    checker.feed(html)
    assert checker.errors == []
    assert checker.stack == []


def test_report_has_every_required_section(single_report):
    _, html = single_report
    for section in REQUIRED_SECTIONS:
        if section == "consensus":
            continue
        assert f'id="{section}"' in html, f"missing section: {section}"


def test_report_is_self_contained(single_report):
    """It must open offline years from now: no CDN, no scripts, no external assets."""
    _, html = single_report
    assert "<script" not in html.lower()
    assert "http://" not in html.replace("http://www.w3.org", "")
    for marker in ("cdn.", "googleapis", "unpkg", "jsdelivr"):
        assert marker not in html
    assert "<link" not in html.lower()


def test_stylesheet_is_inlined(single_report):
    _, html = single_report
    assert "<style>" in html
    assert "--ink" in html


def test_figures_are_inline_svg(single_report):
    _, html = single_report
    assert "<svg" in html
    assert "<img" not in html


def test_report_declares_its_version_and_timestamp(single_report):
    result, html = single_report
    assert result.tool_version in html
    assert result.generated_at in html


# --------------------------------------------------------------------------------------
# Content
# --------------------------------------------------------------------------------------


def test_report_names_the_detected_source(single_report):
    _, html = single_report
    assert "AlphaFold2" in html
    assert "Tier 1" in html


def test_report_shows_the_pocket_table_with_caveats(single_report):
    result, html = single_report
    for pocket in result.primary_pockets.pockets:
        assert pocket.confidence.level.value in html
    assert "Ranked pockets" in html
    assert "Confidence caveats in full" in html


def test_report_includes_a_table_view_of_the_confidence_figure(single_report):
    """The chart palette needs a non-colour path to the same data."""
    _, html = single_report
    assert "Residue counts per confidence band" in html
    assert "Very high (pLDDT &gt;= 90)" in html or "Very high (pLDDT >= 90)" in html


def test_report_renders_pocket_images(single_report):
    result, html = single_report
    assert "Pocket locations" in html
    assert html.count("C-alpha trace") >= len(result.primary_pockets.pockets[:3])


def test_report_lists_the_citations_for_what_it_used(single_report):
    """Citing the wrapper without the underlying methods misattributes the science."""
    _, html = single_report
    assert "Built on" in html
    assert "GROMACS" in html
    assert "CHARMM36m" in html
    assert "Biopython" in html
    assert "Jumper" in html  # the AlphaFold2 reference


def test_report_states_the_backend_actually_used(single_report):
    """When the fallback backend runs, the report must say so, loudly."""
    result, html = single_report
    if not result.primary_pockets.backend_is_fpocket:
        assert "not detected with fpocket" in html
        assert "not</em> fpocket" in html or "NOT fpocket" in html


def test_narrative_section_explains_why_it_was_skipped(single_report):
    _, html = single_report
    assert "Interpretation" in html
    assert "optional" in html


# --------------------------------------------------------------------------------------
# The disclaimer is not optional
# --------------------------------------------------------------------------------------


def test_disclaimer_covers_research_use_and_prediction_limits(single_report):
    import re

    _, html = single_report
    # The template wraps prose across lines, so whitespace is collapsed before matching.
    text = re.sub(r"\s+", " ", html.lower())
    assert "research use only" in text
    assert "not a medical device" in text
    assert "not a diagnostic" in text
    assert "predicted structure, not an experimental one" in text
    assert "no simulation was run" in text
    assert "triage, not judgement" in text


def test_report_contains_no_commercial_language(single_report):
    """A non-negotiable boundary of this project."""
    _, html = single_report
    lowered = html.lower()
    for banned in (
        "pricing",
        "subscription",
        "upgrade to",
        "free tier",
        "paid plan",
        "enterprise plan",
        "buy now",
        "per month",
    ):
        assert banned not in lowered, f"commercial language in report: {banned}"


def test_report_makes_no_clinical_claim(single_report):
    _, html = single_report
    lowered = html.lower()
    for banned in ("treat patients", "therapeutic indication", "clinically proven", "cure for"):
        assert banned not in lowered


# --------------------------------------------------------------------------------------
# Consensus section appears only when it should
# --------------------------------------------------------------------------------------


def test_single_structure_report_has_no_consensus_section(single_report):
    _, html = single_report
    assert 'id="consensus"' not in html
    assert "Cross-model consensus" not in html


def test_consensus_report_has_the_consensus_section(consensus_report):
    _, html = consensus_report
    assert 'id="consensus"' in html
    assert "Cross-model consensus" in html


def test_consensus_table_separates_the_two_agreement_kinds(consensus_report):
    """They must never be pooled into one column."""
    _, html = consensus_report
    assert "CROSS-FAMILY AGREEMENT" in html.upper()
    assert "SAME-FAMILY AGREEMENT" in html.upper()
    assert "cross-family" in html
    assert "same-family-only" in html


def test_consensus_report_is_well_formed(consensus_report):
    _, html = consensus_report
    checker = _WellFormednessCheck()
    checker.feed(html)
    assert checker.errors == []
    assert checker.stack == []


def test_single_structure_report_is_unaffected_by_the_consensus_module(
    tmp_path, af2_path
):
    """Definition of done: single-structure runs are byte-for-byte unaffected.

    Pinned to the built-in backend deliberately. What this test asserts is that the
    consensus module's *existence* changes nothing about a single-structure run -- a
    property of Pocketscribe's own wiring. Running it through fpocket would instead be
    testing fpocket's reproducibility, which is a different question with a different
    answer: see :func:`test_fpocket_volumes_are_not_expected_to_be_reproducible`.
    """
    first_result, first_html = _render(
        tmp_path / "a",
        [StructureInput(path=af2_path, source_id="alphafold2", label=af2_path.name)],
        pockets__backend="builtin",
    )
    second_result, second_html = _render(
        tmp_path / "b",
        [StructureInput(path=af2_path, source_id="alphafold2", label=af2_path.name)],
        pockets__backend="builtin",
    )
    assert first_result.consensus is None
    assert second_result.consensus is None

    normalise = lambda text, result: text.replace(result.generated_at, "TIMESTAMP")  # noqa: E731
    assert normalise(first_html, first_result) == normalise(second_html, second_result)


def test_fpocket_volumes_are_not_expected_to_be_reproducible():
    """Documents a property of fpocket that surprised us, so nobody re-asserts otherwise.

    fpocket estimates pocket volume by Monte Carlo integration. Two runs over the same
    structure therefore report volumes differing by roughly a percent, while the
    druggability scores and the resulting ranking stay stable. Observed on fpocket
    4.2.3: the same pocket came back as 1564 A^3 and then 1555 A^3.

    Consequences that the rest of the code has to respect:

    * a report generated with the fpocket backend is not byte-reproducible, and no test
      may assert that it is;
    * the consensus module must not match pockets on volume equality (it matches on
      lining-residue overlap and centroid distance, neither of which is stochastic);
    * the methods appendix says so, so a user comparing two runs is not left puzzled.

    This test asserts the *design response*, not fpocket's behaviour, so that it stays
    meaningful on a machine without fpocket installed.
    """
    from pathlib import Path

    from pocketscribe.models import ConsensusParameters

    # Pocket matching must not depend on volume at all.
    parameters = ConsensusParameters()
    assert not hasattr(parameters, "volume_tolerance")

    source = (Path(__file__).parent.parent / "pocketscribe" / "consensus.py").read_text()
    matcher = source.split("def _match_score")[1].split("\ndef ")[0]
    assert "volume" not in matcher, (
        "pocket correspondence must not use volume: fpocket volumes are stochastic"
    )


# --------------------------------------------------------------------------------------
# Plot internals
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "value,expected",
    [(98.0, "very_high"), (90.0, "very_high"), (75.0, "confident"), (70.0, "confident"),
     (60.0, "low"), (50.0, "low"), (30.0, "very_low")],
)
def test_confidence_bands_follow_the_alphafold_convention(value, expected):
    assert band_of(value) == expected


def test_confidence_plot_labels_both_decision_thresholds(af2_structure):
    from pocketscribe.confidence import build_confidence_profile
    from pocketscribe.sources import get_source

    profile = build_confidence_profile(af2_structure, get_source("alphafold2"))
    svg = confidence_plot(profile)
    assert svg.startswith("<svg")
    assert svg.endswith("</svg>")
    assert 'aria-label' in svg
    assert ">70<" in svg and ">50<" in svg


def test_confidence_plot_handles_an_empty_profile():
    from pocketscribe.models import ConfidenceMetric, ConfidenceProfile

    empty = ConfidenceProfile(
        metric=ConfidenceMetric.PLDDT,
        residues=[],
        mean=0.0,
        median=0.0,
        minimum=0.0,
        maximum=0.0,
        fraction_below_70=0.0,
        fraction_below_50=0.0,
    )
    assert "No per-residue confidence" in confidence_plot(empty)
