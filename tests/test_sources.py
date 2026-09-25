"""Structure-source adapters: detection, confidence extraction, and refusal to guess.

Each Tier 1 source gets its own test against its own fixture file, with that source's
actual header and filename conventions. Reusing one relabelled file across four tests
would prove nothing about detection, which is precisely where these sources differ.
"""

from __future__ import annotations

import json

import pytest

from pocketscribe.confidence import build_confidence_profile
from pocketscribe.errors import (
    ConfidenceExtractionError,
    SourceDetectionError,
    UnsupportedSourceError,
)
from pocketscribe.models import ArchitectureFamily, SupportTier
from pocketscribe.parsing import parse_structure
from pocketscribe.sources import (
    AlphaFold2Source,
    BoltzSource,
    ColabFoldSource,
    ESMFoldSource,
    OpenFoldSource,
    all_sources,
    detect_source,
    get_source,
)

# --------------------------------------------------------------------------------------
# Registry
# --------------------------------------------------------------------------------------


def test_registry_lists_every_declared_source():
    ids = {adapter.info.id for adapter in all_sources()}
    assert ids == {
        "alphafold2",
        "openfold",
        "colabfold",
        "esmfold",
        "boltz",
        "protenix",
        "alphafold3",
        "rosettafold",
    }


def test_unknown_source_id_is_rejected():
    with pytest.raises(SourceDetectionError, match="Unknown structure source"):
        get_source("alphafold9")


def test_every_source_declares_a_citation():
    """A source without a citation would be uncitable in a methods section."""
    from pocketscribe.citations import CITATIONS

    for adapter in all_sources():
        assert adapter.info.citation_key, f"{adapter.info.id} has no citation key"
        assert adapter.info.citation_key in CITATIONS, adapter.info.id


# --------------------------------------------------------------------------------------
# Tier 1: one test per source, each with its own fixture
# --------------------------------------------------------------------------------------


def test_alphafold2_confidence(af2_path):
    parsed = parse_structure(af2_path)
    metric, values = AlphaFold2Source().extract_confidence(af2_path, parsed.residues)
    assert metric.value.startswith("pLDDT")
    assert len(values) == parsed.qc.n_residues
    assert 0.0 <= min(values.values()) <= max(values.values()) <= 100.0
    # The fixture deliberately contains a poorly-predicted stretch.
    assert min(values.values()) < 50.0
    assert max(values.values()) > 90.0


def test_openfold_confidence(openfold_path):
    parsed = parse_structure(openfold_path)
    metric, values = OpenFoldSource().extract_confidence(openfold_path, parsed.residues)
    assert len(values) == parsed.qc.n_residues
    assert 0.0 <= min(values.values()) <= max(values.values()) <= 100.0


def test_openfold_shares_alphafold2_convention_explicitly(openfold_path):
    """Asserted rather than assumed.

    OpenFold reimplements AlphaFold2 and currently writes pLDDT the same way. If a
    future OpenFold release diverged, sharing the adapter would silently produce wrong
    confidence, so the equivalence is pinned here.
    """
    parsed = parse_structure(openfold_path)
    _, from_openfold = OpenFoldSource().extract_confidence(openfold_path, parsed.residues)
    _, from_alphafold = AlphaFold2Source().extract_confidence(openfold_path, parsed.residues)
    assert from_openfold == from_alphafold


def test_colabfold_confidence(colabfold_path):
    parsed = parse_structure(colabfold_path)
    metric, values = ColabFoldSource().extract_confidence(colabfold_path, parsed.residues)
    assert len(values) == parsed.qc.n_residues
    assert 0.0 <= min(values.values()) <= max(values.values()) <= 100.0


def test_esmfold_confidence(esmfold_path):
    parsed = parse_structure(esmfold_path)
    metric, values = ESMFoldSource().extract_confidence(esmfold_path, parsed.residues)
    assert len(values) == parsed.qc.n_residues


def test_esmfold_is_a_different_architecture_family():
    """The whole consensus module depends on this classification being right."""
    assert ESMFoldSource().info.family is ArchitectureFamily.SINGLE_SEQUENCE_LM
    assert ESMFoldSource().info.uses_msa is False
    for adapter in (AlphaFold2Source(), OpenFoldSource(), ColabFoldSource(), BoltzSource()):
        assert adapter.info.family is ArchitectureFamily.MSA_COEVOLUTION


# --------------------------------------------------------------------------------------
# Auto-detection
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "fixture_name,expected_id",
    [
        ("af2_path", "alphafold2"),
        ("openfold_path", "openfold"),
        ("colabfold_path", "colabfold"),
        ("esmfold_path", "esmfold"),
        ("boltz_path", "boltz"),
    ],
)
def test_auto_detection_identifies_each_source(request, fixture_name, expected_id):
    path = request.getfixturevalue(fixture_name)
    adapter, evidence = detect_source(path)
    assert adapter.info.id == expected_id
    assert evidence, "detection must explain itself so a user can correct a wrong guess"


def test_colabfold_is_not_misattributed_to_alphafold2(colabfold_path):
    """ColabFold output mentions AlphaFold2; it must still be labelled ColabFold."""
    adapter, _ = detect_source(colabfold_path)
    assert adapter.info.id == "colabfold"
    assert "ColabFold" in adapter.info.display_name


def test_unrecognised_file_is_not_guessed_as_alphafold2(tmp_path, af2_path):
    """The most dangerous silent failure: defaulting to AF2 and reading B-factors."""
    stripped = [
        line for line in af2_path.read_text().splitlines() if line.startswith(("ATOM", "END"))
    ]
    path = tmp_path / "anonymous.pdb"
    path.write_text("\n".join(stripped) + "\n")
    with pytest.raises(SourceDetectionError, match="Could not determine"):
        detect_source(path)


# --------------------------------------------------------------------------------------
# Experimental structures must never be read as predictions
# --------------------------------------------------------------------------------------


def test_experimental_bfactors_are_refused(experimental_path):
    """Per-atom temperature factors vary within a residue; pLDDT does not."""
    parsed = parse_structure(experimental_path)
    with pytest.raises(ConfidenceExtractionError, match="vary between atoms"):
        AlphaFold2Source().extract_confidence(experimental_path, parsed.residues)


def test_all_zero_bfactors_are_refused(tmp_path, af2_path):
    lines = []
    for line in af2_path.read_text().splitlines():
        lines.append(line[:60] + "  0.00" + line[66:] if line.startswith("ATOM") else line)
    path = tmp_path / "zeroed.pdb"
    path.write_text("\n".join(lines) + "\n")
    parsed = parse_structure(path)
    with pytest.raises(ConfidenceExtractionError, match="zero"):
        AlphaFold2Source().extract_confidence(path, parsed.residues)


# --------------------------------------------------------------------------------------
# Tier 2 reference adapter: Boltz
# --------------------------------------------------------------------------------------


def test_boltz_reads_its_sidecar_and_normalises_scale(boltz_path):
    parsed = parse_structure(boltz_path)
    metric, values = BoltzSource().extract_confidence(boltz_path, parsed.residues)
    assert len(values) == parsed.qc.n_residues
    # Boltz reports 0-1; Pocketscribe normalises so every threshold means the same thing.
    assert max(values.values()) > 1.0
    assert max(values.values()) <= 100.0


def test_boltz_does_not_fall_back_to_bfactors(tmp_path, boltz_path):
    """Boltz B-factors are not pLDDT, so a missing sidecar must raise, not degrade."""
    copied = tmp_path / boltz_path.name
    copied.write_text(boltz_path.read_text())
    parsed = parse_structure(copied)
    with pytest.raises(ConfidenceExtractionError, match="sidecar"):
        BoltzSource().extract_confidence(copied, parsed.residues)


def test_boltz_rejects_a_mismatched_sidecar(tmp_path, boltz_path):
    copied = tmp_path / boltz_path.name
    copied.write_text(boltz_path.read_text())
    (tmp_path / f"confidence_{copied.stem}.json").write_text(
        json.dumps({"plddt": [0.9, 0.8, 0.7]})
    )
    parsed = parse_structure(copied)
    with pytest.raises(ConfidenceExtractionError, match="per-residue values"):
        BoltzSource().extract_confidence(copied, parsed.residues)


def test_boltz_is_tier_2_reference():
    assert BoltzSource().info.tier is SupportTier.TIER_2_REFERENCE


# --------------------------------------------------------------------------------------
# Tier 2 stubs must fail loudly and specifically
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("source_id", ["protenix", "alphafold3", "rosettafold"])
def test_tier_2_stub_fails_with_a_source_specific_message(source_id, af2_path):
    adapter = get_source(source_id)
    assert adapter.info.tier is SupportTier.TIER_2_STUB

    parsed = parse_structure(af2_path)
    with pytest.raises(UnsupportedSourceError) as excinfo:
        adapter.extract_confidence(af2_path, parsed.residues)

    message = str(excinfo.value)
    assert source_id in message
    assert "not-yet-implemented" in message or "not yet implemented" in message
    # The error must explain why the AlphaFold2 parser cannot simply be reused, or a
    # user will assume the tool is being unhelpful and work around it.
    assert "Why it cannot fall back" in message
    assert "CONTRIBUTING.md" in message


@pytest.mark.parametrize("source_id", ["protenix", "alphafold3", "rosettafold"])
def test_tier_2_stub_never_returns_numbers(source_id, af2_path):
    """A stub that returned plausible values would be worse than one that fails."""
    parsed = parse_structure(af2_path)
    with pytest.raises(UnsupportedSourceError):
        build_confidence_profile(parsed, get_source(source_id))
