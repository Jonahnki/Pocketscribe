"""Cross-model consensus.

This module contains Pocketscribe's only original method, so it is tested against
constructed cases where the right answer is known in advance, not only against the
bundled fixtures. The four properties that must hold:

1. Different proteins are refused, never compared.
2. Pocket correspondence requires both residue overlap and geometric proximity.
3. Cross-family and same-family agreement are classified correctly and never pooled.
4. A single structure never activates the module at all.
"""

from __future__ import annotations

import numpy as np
import pytest

from pocketscribe.confidence import annotate_pocket_confidence
from pocketscribe.consensus import (
    _is_cross_family,
    _match_score,
    _possible_pair_counts,
    compute_consensus,
    kabsch,
    map_structures,
)
from pocketscribe.errors import ConsensusError, SequenceMismatchError
from pocketscribe.models import (
    ArchitectureFamily,
    ConsensusParameters,
    PocketResidue,
)
from pocketscribe.parsing import parse_structure
from pocketscribe.pipeline import StructureInput, load_structure
from pocketscribe.pockets import BuiltinGeometricBackend


def _analyse(paths_and_sources):
    """Parse, profile and detect pockets for several structures."""
    structures, parsed_list, pocket_sets = [], [], []
    for path, source_id in paths_and_sources:
        structure, parsed, _ = load_structure(
            StructureInput(path=path, source_id=source_id), label=path.name
        )
        pocket_set = BuiltinGeometricBackend().detect(parsed)
        for pocket in pocket_set.pockets:
            pocket.confidence = annotate_pocket_confidence(pocket, structure.confidence)
        structures.append(structure)
        parsed_list.append(parsed)
        pocket_sets.append(pocket_set)
    return structures, parsed_list, pocket_sets


# --------------------------------------------------------------------------------------
# 1. Mismatched proteins are refused
# --------------------------------------------------------------------------------------


def _mutated_copy(source_path, tmp_path, fraction: float):
    """Write a copy of a model with a fraction of its residues mutated.

    Used to exercise the identity *threshold* specifically. Two unrelated proteins fail
    earlier, at residue correspondence, so they cannot test that branch.
    """
    import random

    random.seed(7)
    alternatives = ["GLY", "SER", "THR", "ASN", "ASP", "GLU", "LYS", "ARG"]
    resseqs = {
        int(line[22:26])
        for line in source_path.read_text().splitlines()
        if line.startswith("ATOM")
    }
    mutate = {
        resseq for resseq in sorted(resseqs) if random.random() < fraction
    }
    replacement = {resseq: random.choice(alternatives) for resseq in mutate}

    lines = []
    for line in source_path.read_text().splitlines():
        if line.startswith("ATOM"):
            resseq = int(line[22:26])
            atom_name = line[12:16].strip()
            if resseq in mutate:
                if atom_name not in {"N", "CA", "C", "O", "CB"}:
                    continue
                line = f"{line[:17]}{replacement[resseq]:>3}{line[20:]}"
        lines.append(line)

    path = tmp_path / "mutated.pdb"
    path.write_text("\n".join(lines) + "\n")
    return path


def test_different_proteins_are_rejected(af2_path, different_protein_path):
    """The headline safety property: never produce a meaningless comparison."""
    structures, parsed, pocket_sets = _analyse(
        [(af2_path, "alphafold2"), (different_protein_path, "alphafold2")]
    )
    with pytest.raises(SequenceMismatchError) as excinfo:
        compute_consensus(structures, parsed, pocket_sets)

    message = str(excinfo.value)
    assert "not the same protein" in message
    assert "nothing meaningful to compare" in message
    # The error must name both models, so the user knows which input to look at.
    assert af2_path.name in message
    assert different_protein_path.name in message


def test_related_but_non_identical_models_hit_the_identity_gate(af2_path, tmp_path):
    """The threshold branch: similar enough to align, too different to be one protein."""
    mutated = _mutated_copy(af2_path, tmp_path, fraction=0.4)
    structures, parsed, pocket_sets = _analyse(
        [(af2_path, "alphafold2"), (mutated, "alphafold2")]
    )
    with pytest.raises(SequenceMismatchError) as excinfo:
        compute_consensus(structures, parsed, pocket_sets)

    message = str(excinfo.value)
    assert "sequence identity" in message
    # Here, lowering the threshold IS a sensible remedy, so the error must say how.
    assert "--min-identity" in message


def test_mismatch_error_reports_the_actual_identity(af2_path, tmp_path):
    mutated = _mutated_copy(af2_path, tmp_path, fraction=0.4)
    structures, parsed, pocket_sets = _analyse(
        [(af2_path, "alphafold2"), (mutated, "alphafold2")]
    )
    with pytest.raises(SequenceMismatchError, match=r"\d+\.\d%"):
        compute_consensus(structures, parsed, pocket_sets)


def test_identical_proteins_pass_the_gate(af2_path, openfold_path):
    structures, parsed, pocket_sets = _analyse(
        [(af2_path, "alphafold2"), (openfold_path, "openfold")]
    )
    report = compute_consensus(structures, parsed, pocket_sets)
    assert report.alignments[0].identity >= 0.95


def test_lowering_the_gate_still_refuses_unrelated_proteins(
    af2_path, different_protein_path
):
    """The identity threshold is configurable, but it is not a way to force nonsense.

    Two unrelated sequences align to nothing, so even with the gate fully open there is
    no residue correspondence to compare pockets through, and the tool says so instead
    of failing later with a message about missing backbone atoms.
    """
    structures, parsed, pocket_sets = _analyse(
        [(af2_path, "alphafold2"), (different_protein_path, "alphafold2")]
    )
    with pytest.raises(SequenceMismatchError, match="residue correspondence"):
        compute_consensus(
            structures, parsed, pocket_sets, ConsensusParameters(min_sequence_identity=0.0)
        )


def test_threshold_is_configurable_for_genuinely_related_models(
    af2_path, openfold_path
):
    """A user comparing two constructs of one protein can lower the gate and proceed."""
    structures, parsed, pocket_sets = _analyse(
        [(af2_path, "alphafold2"), (openfold_path, "openfold")]
    )
    report = compute_consensus(
        structures, parsed, pocket_sets, ConsensusParameters(min_sequence_identity=0.80)
    )
    assert report.parameters.min_sequence_identity == 0.80
    assert report.groups


def test_a_single_structure_cannot_reach_consensus(af2_path):
    structures, parsed, pocket_sets = _analyse([(af2_path, "alphafold2")])
    with pytest.raises(ConsensusError, match="at least 2"):
        compute_consensus(structures, parsed, pocket_sets)


# --------------------------------------------------------------------------------------
# 2. Superposition and residue mapping
# --------------------------------------------------------------------------------------


def test_kabsch_recovers_a_known_rotation():
    rng = np.random.default_rng(0)
    points = rng.normal(size=(30, 3))
    angle = 0.9
    rotation = np.array(
        [
            [np.cos(angle), -np.sin(angle), 0.0],
            [np.sin(angle), np.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    moved = (rotation @ points.T).T + np.array([5.0, -2.0, 1.0])

    recovered_rotation, translation, rmsd = kabsch(moved, points)
    assert rmsd < 1e-8
    assert np.allclose((recovered_rotation @ moved.T).T + translation, points, atol=1e-8)


def test_kabsch_never_produces_a_reflection():
    """A mirror image would superpose well and be structurally meaningless."""
    points = np.array([[0.0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 1]])
    rotation, _, _ = kabsch(points, points * np.array([1.0, 1.0, -1.0]))
    assert np.linalg.det(rotation) > 0


def test_kabsch_needs_enough_atoms():
    with pytest.raises(ConsensusError, match="at least 3"):
        kabsch(np.zeros((2, 3)), np.zeros((2, 3)))


def test_residue_mapping_survives_renumbering(af2_path, tmp_path):
    """Models differing only in numbering must still be put into correspondence."""
    renumbered_lines = []
    for line in af2_path.read_text().splitlines():
        if line.startswith("ATOM"):
            resseq = int(line[22:26]) + 500
            line = f"{line[:22]}{resseq:>4}{line[26:]}"
        renumbered_lines.append(line)
    path = tmp_path / "renumbered.pdb"
    path.write_text("\n".join(renumbered_lines) + "\n")

    reference = parse_structure(af2_path)
    other = parse_structure(path)
    mapping, identity, aligned = map_structures(reference, other)

    assert identity == pytest.approx(1.0)
    assert len(mapping) == reference.qc.n_residues
    assert mapping[("A", 501, " ")] == ("A", 1, " ")


def test_superposition_rmsd_is_reported(af2_path, openfold_path):
    """The OpenFold fixture is rotated and translated, so a real superposition happened."""
    structures, parsed, pocket_sets = _analyse(
        [(af2_path, "alphafold2"), (openfold_path, "openfold")]
    )
    report = compute_consensus(structures, parsed, pocket_sets)
    rmsd = report.alignments[0].rmsd
    assert rmsd is not None and 0.0 < rmsd < 3.0


# --------------------------------------------------------------------------------------
# 3. Pocket correspondence needs BOTH criteria
# --------------------------------------------------------------------------------------


class _FakePlaced:
    """Minimal stand-in for a placed pocket, so matching can be tested in isolation."""

    def __init__(self, structure_index, residues, centroid, family, confidence=90.0):
        from pocketscribe.models import Pocket

        self.structure_index = structure_index
        self.structure_label = f"model{structure_index}"
        self.source_id = "x"
        self.source_display_name = "X"
        self.family = family
        self.pocket = Pocket(
            id=1,
            rank=1,
            residues=[
                PocketResidue(chain="A", resseq=resseq, icode=" ", resname="ALA")
                for resseq in residues
            ],
        )
        self.mapped_residues = {("A", resseq, " ") for resseq in residues}
        self.centroid = np.array(centroid, dtype=float) if centroid is not None else None
        self.mean_confidence = confidence


def test_matching_requires_residue_overlap():
    """Two nearby cavities that share no residues are not the same pocket."""
    parameters = ConsensusParameters()
    a = _FakePlaced(0, [1, 2, 3, 4], [0, 0, 0], ArchitectureFamily.MSA_COEVOLUTION)
    b = _FakePlaced(1, [90, 91, 92, 93], [1, 0, 0], ArchitectureFamily.MSA_COEVOLUTION)
    matched, _, jaccard, distance = _match_score(a, b, parameters)
    assert distance < parameters.centroid_tolerance  # geometry alone would have matched
    assert jaccard == 0.0
    assert matched is False


def test_matching_requires_geometric_proximity():
    """Two distant cavities that share residues are not the same pocket either."""
    parameters = ConsensusParameters()
    a = _FakePlaced(0, [1, 2, 3, 4], [0, 0, 0], ArchitectureFamily.MSA_COEVOLUTION)
    b = _FakePlaced(1, [1, 2, 3, 4], [60, 0, 0], ArchitectureFamily.MSA_COEVOLUTION)
    matched, _, jaccard, distance = _match_score(a, b, parameters)
    assert jaccard == 1.0  # overlap alone would have matched
    assert distance > parameters.centroid_tolerance
    assert matched is False


def test_matching_succeeds_when_both_criteria_hold():
    parameters = ConsensusParameters()
    a = _FakePlaced(0, [1, 2, 3, 4, 5, 6], [0, 0, 0], ArchitectureFamily.MSA_COEVOLUTION)
    b = _FakePlaced(1, [2, 3, 4, 5, 6, 7], [2, 0, 0], ArchitectureFamily.MSA_COEVOLUTION)
    matched, score, jaccard, distance = _match_score(a, b, parameters)
    assert matched is True
    # 5 shared of 7 union.
    assert jaccard == pytest.approx(5 / 7)
    assert distance == pytest.approx(2.0)
    assert 0.0 < score <= 1.0


def test_a_pocket_without_a_centroid_never_matches():
    """Falling back to residue overlap alone would break the both-criteria guarantee."""
    parameters = ConsensusParameters()
    a = _FakePlaced(0, [1, 2, 3], None, ArchitectureFamily.MSA_COEVOLUTION)
    b = _FakePlaced(1, [1, 2, 3], [0, 0, 0], ArchitectureFamily.MSA_COEVOLUTION)
    matched, _, _, _ = _match_score(a, b, parameters)
    assert matched is False


# --------------------------------------------------------------------------------------
# 4. Family classification -- cross vs same must never be pooled
# --------------------------------------------------------------------------------------


def test_family_classification():
    msa = ArchitectureFamily.MSA_COEVOLUTION
    lm = ArchitectureFamily.SINGLE_SEQUENCE_LM
    unknown = ArchitectureFamily.UNKNOWN

    assert _is_cross_family(msa, lm) is True
    assert _is_cross_family(msa, msa) is False
    # An unknown family is never counted as independent evidence.
    assert _is_cross_family(msa, unknown) is False
    assert _is_cross_family(unknown, unknown) is False


def test_possible_pair_counts():
    msa = ArchitectureFamily.MSA_COEVOLUTION
    lm = ArchitectureFamily.SINGLE_SEQUENCE_LM
    # Three models: AF2, OpenFold (same family) and ESMFold (different).
    cross, same = _possible_pair_counts([msa, msa, lm])
    assert (cross, same) == (2, 1)


def test_two_same_family_models_report_no_cross_family_support(af2_path, openfold_path):
    """AlphaFold2 + OpenFold agreeing is NOT independent confirmation."""
    structures, parsed, pocket_sets = _analyse(
        [(af2_path, "alphafold2"), (openfold_path, "openfold")]
    )
    report = compute_consensus(structures, parsed, pocket_sets)

    assert report.cross_family_comparison_possible is False
    assert report.n_cross_family_pockets == 0
    for group in report.groups:
        assert group.cross_family_support is False
        assert group.cross_family_agreement == 0.0
    # And the report must say so, not leave the user to infer it.
    assert any("same architecture family" in note for note in report.notes)


def test_cross_family_agreement_is_detected(af2_path, esmfold_path):
    """AlphaFold2 + ESMFold agreeing IS independent evidence."""
    structures, parsed, pocket_sets = _analyse(
        [(af2_path, "alphafold2"), (esmfold_path, "esmfold")]
    )
    report = compute_consensus(structures, parsed, pocket_sets)

    assert report.cross_family_comparison_possible is True
    assert report.n_cross_family_pockets >= 1
    top = report.groups[0]
    assert top.cross_family_support is True
    assert top.evidence_label == "cross-family"
    assert top.cross_family_agreement > 0.0


def test_the_three_evidence_classes_are_distinguished(
    af2_path, openfold_path, esmfold_path
):
    """The constructed case with a known answer.

    The bundled fixtures are built so that the large cavity is found by all three
    models (cross-family), the small one only by AlphaFold2 and OpenFold
    (same-family-only), and one ESMFold-specific cavity by nothing else (single-model).
    """
    structures, parsed, pocket_sets = _analyse(
        [
            (af2_path, "alphafold2"),
            (openfold_path, "openfold"),
            (esmfold_path, "esmfold"),
        ]
    )
    report = compute_consensus(structures, parsed, pocket_sets)

    labels = {group.evidence_label for group in report.groups}
    assert {"cross-family", "same-family-only", "single-model"} <= labels

    cross = next(g for g in report.groups if g.evidence_label == "cross-family")
    same_only = next(g for g in report.groups if g.evidence_label == "same-family-only")

    # Cross-family group: both kinds of pair exist inside it and are reported apart.
    assert cross.n_family_pairs_cross > 0
    assert cross.cross_family_agreement > 0.0

    # Same-family-only group: same-family support, and zero cross-family support.
    assert same_only.same_family_support is True
    assert same_only.cross_family_support is False
    assert same_only.cross_family_agreement == 0.0
    assert same_only.same_family_agreement > 0.0


def test_cross_family_groups_rank_above_same_family(af2_path, openfold_path, esmfold_path):
    """Ordering encodes the evidence hierarchy, so a reader sees the strongest first."""
    structures, parsed, pocket_sets = _analyse(
        [
            (af2_path, "alphafold2"),
            (openfold_path, "openfold"),
            (esmfold_path, "esmfold"),
        ]
    )
    report = compute_consensus(structures, parsed, pocket_sets)
    assert report.groups[0].evidence_label == "cross-family"


def test_same_family_note_warns_against_reading_it_as_confirmation(
    af2_path, openfold_path, esmfold_path
):
    structures, parsed, pocket_sets = _analyse(
        [
            (af2_path, "alphafold2"),
            (openfold_path, "openfold"),
            (esmfold_path, "esmfold"),
        ]
    )
    report = compute_consensus(structures, parsed, pocket_sets)
    same_only = next(g for g in report.groups if g.evidence_label == "same-family-only")
    assert "not be read as independent confirmation" in same_only.consensus_note


def test_confidence_weighting_changes_agreement_values(af2_path, esmfold_path):
    """Weighting must actually do something, or the option is a lie."""
    structures, parsed, pocket_sets = _analyse(
        [(af2_path, "alphafold2"), (esmfold_path, "esmfold")]
    )
    weighted = compute_consensus(
        structures, parsed, pocket_sets, ConsensusParameters(confidence_weighting=True)
    )
    unweighted = compute_consensus(
        structures, parsed, pocket_sets, ConsensusParameters(confidence_weighting=False)
    )
    assert unweighted.groups[0].cross_family_agreement == 1.0
    assert weighted.groups[0].cross_family_agreement < 1.0


def test_one_model_cannot_agree_with_itself(af2_path, openfold_path):
    """Two pockets from the same model must never land in one group."""
    structures, parsed, pocket_sets = _analyse(
        [(af2_path, "alphafold2"), (openfold_path, "openfold")]
    )
    report = compute_consensus(structures, parsed, pocket_sets)
    for group in report.groups:
        labels = [member.structure_label for member in group.members]
        assert len(labels) == len(set(labels))


def test_consensus_feeds_back_into_pocket_caveats(af2_path, esmfold_path):
    """Consensus extends the existing caveat rather than creating a second score."""
    from pocketscribe.consensus import apply_consensus_to_pockets

    structures, parsed, pocket_sets = _analyse(
        [(af2_path, "alphafold2"), (esmfold_path, "esmfold")]
    )
    before = pocket_sets[0].pockets[0].confidence.text
    report = compute_consensus(structures, parsed, pocket_sets)
    apply_consensus_to_pockets(report, structures, pocket_sets)
    after = pocket_sets[0].pockets[0].confidence.text

    assert after.startswith(before)
    assert "Cross-model consensus:" in after
