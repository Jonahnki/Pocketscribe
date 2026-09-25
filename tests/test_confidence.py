"""Confidence profiling and pocket cross-referencing.

The cross-referencing logic is what distinguishes Pocketscribe from a bare pocket
scorer, so it is tested against constructed cases where the right answer is known,
not only against the bundled structures.
"""

from __future__ import annotations

import pytest

from pocketscribe.confidence import (
    annotate_pocket_confidence,
    build_confidence_profile,
    confidence_band_counts,
    find_low_confidence_regions,
)
from pocketscribe.models import (
    CaveatLevel,
    ConfidenceMetric,
    ConfidenceProfile,
    Pocket,
    PocketResidue,
    ResidueConfidence,
)
from pocketscribe.sources import get_source


def _residue(resseq: int, value: float, chain: str = "A") -> ResidueConfidence:
    return ResidueConfidence(
        chain=chain, resseq=resseq, icode=" ", resname="ALA", value=value
    )


def _profile(values: list[float], chain: str = "A") -> ConfidenceProfile:
    residues = [_residue(index + 1, value, chain) for index, value in enumerate(values)]
    return ConfidenceProfile(
        metric=ConfidenceMetric.PLDDT,
        residues=residues,
        mean=sum(values) / len(values),
        median=sorted(values)[len(values) // 2],
        minimum=min(values),
        maximum=max(values),
        fraction_below_70=sum(v < 70 for v in values) / len(values),
        fraction_below_50=sum(v < 50 for v in values) / len(values),
        regions_below_70=find_low_confidence_regions(residues, 70.0),
        regions_below_50=find_low_confidence_regions(residues, 50.0),
    )


def _pocket(resseqs: list[int], chain: str = "A") -> Pocket:
    return Pocket(
        id=1,
        rank=1,
        volume=500.0,
        druggability_score=0.8,
        residues=[
            PocketResidue(chain=chain, resseq=resseq, icode=" ", resname="ALA")
            for resseq in resseqs
        ],
    )


# --------------------------------------------------------------------------------------
# Profiles
# --------------------------------------------------------------------------------------


def test_profile_from_a_real_model(af2_structure, af2_path):
    profile = build_confidence_profile(af2_structure, get_source("alphafold2"))
    assert len(profile.residues) == af2_structure.qc.n_residues
    assert 0.0 < profile.mean < 100.0
    assert profile.regions_below_50, "the fixture has a deliberately bad region"


def test_band_counts_sum_to_the_residue_count(af2_structure):
    profile = build_confidence_profile(af2_structure, get_source("alphafold2"))
    counts = confidence_band_counts(profile)
    assert sum(counts.values()) == len(profile.residues)


# --------------------------------------------------------------------------------------
# Low-confidence regions
# --------------------------------------------------------------------------------------


def test_finds_a_contiguous_low_region():
    values = [95.0] * 10 + [40.0] * 8 + [92.0] * 10
    regions = find_low_confidence_regions(
        [_residue(index + 1, value) for index, value in enumerate(values)], 50.0
    )
    assert len(regions) == 1
    assert (regions[0].start, regions[0].end) == (11, 18)
    assert regions[0].length == 8


def test_short_dips_are_not_reported_as_regions():
    values = [95.0] * 10 + [40.0] * 2 + [95.0] * 10
    regions = find_low_confidence_regions(
        [_residue(index + 1, value) for index, value in enumerate(values)], 50.0
    )
    assert regions == []


def test_regions_do_not_span_chains():
    """A run must break at a chain boundary, or it would describe a gap that is not real."""
    residues = [_residue(index + 1, 40.0, "A") for index in range(5)]
    residues += [_residue(index + 1, 40.0, "B") for index in range(5)]
    regions = find_low_confidence_regions(residues, 50.0)
    assert len(regions) == 2
    assert {region.chain for region in regions} == {"A", "B"}


def test_regions_break_at_numbering_gaps():
    residues = [_residue(resseq, 40.0) for resseq in [1, 2, 3, 4, 50, 51, 52, 53]]
    regions = find_low_confidence_regions(residues, 50.0)
    assert len(regions) == 2


# --------------------------------------------------------------------------------------
# Pocket cross-referencing -- the core behaviour
# --------------------------------------------------------------------------------------


def test_confident_pocket_gets_no_caveat():
    profile = _profile([95.0] * 40)
    annotation = annotate_pocket_confidence(_pocket([1, 2, 3, 4, 5]), profile)
    assert annotation.level is CaveatLevel.NONE
    assert annotation.mean_confidence == pytest.approx(95.0)
    assert annotation.fraction_below_70 == 0.0


def test_pocket_in_a_disordered_region_is_flagged_severe():
    """The case the whole tool exists for: a good score in structure nobody should trust."""
    profile = _profile([95.0] * 20 + [35.0] * 20)
    annotation = annotate_pocket_confidence(_pocket([21, 22, 23, 24, 25]), profile)
    assert annotation.level is CaveatLevel.SEVERE
    assert annotation.fraction_below_50 == 1.0
    assert "should not be relied upon" in annotation.text


def test_mixed_pocket_is_flagged_moderate():
    profile = _profile([95.0] * 20 + [35.0] * 20)
    annotation = annotate_pocket_confidence(_pocket([18, 19, 20, 21, 22]), profile)
    assert annotation.level in {CaveatLevel.MODERATE, CaveatLevel.SEVERE}
    assert 0.0 < annotation.fraction_below_50 < 1.0


def test_caveat_names_the_overlapping_region():
    profile = _profile([95.0] * 20 + [35.0] * 20)
    annotation = annotate_pocket_confidence(_pocket([21, 22, 23]), profile)
    assert annotation.overlapping_low_confidence_regions
    assert "A:21-40" in annotation.text


def test_annotation_writes_confidence_onto_each_lining_residue():
    """Downstream code (the pocket image, consensus weighting) reads these values."""
    profile = _profile([95.0] * 10)
    pocket = _pocket([1, 2, 3])
    annotate_pocket_confidence(pocket, profile)
    assert all(residue.confidence == pytest.approx(95.0) for residue in pocket.residues)


def test_pocket_with_no_matching_confidence_is_severe_not_silent():
    """Unknown confidence must never read as good confidence."""
    profile = _profile([95.0] * 10)
    annotation = annotate_pocket_confidence(_pocket([500, 501]), profile)
    assert annotation.level is CaveatLevel.SEVERE
    assert annotation.n_residues_scored == 0
    assert "cannot be assessed" in annotation.text


def test_every_caveat_text_mentions_that_this_is_a_model():
    """A confident pocket still carries the predicted-structure caveat."""
    profile = _profile([96.0] * 30)
    annotation = annotate_pocket_confidence(_pocket([1, 2, 3]), profile)
    assert "computational model" in annotation.text
