"""Cross-model pocket consensus (pipeline stage 3, optional).

This is the one stage of the pipeline that is Pocketscribe's own algorithmic content
rather than orchestration of an existing tool. Everything else here wraps fpocket,
GROMACS or a structure-prediction model; this module implements a method.

The idea, stated plainly
------------------------
Structural biologists already treat agreement between independent crystal forms as
stronger evidence than agreement between copies in one asymmetric unit. The same
logic applies to computational predictions: if an MSA-based model (AlphaFold2) and a
single-sequence language model (ESMFold) independently produce the same cavity, that
is better evidence the cavity is real than if AlphaFold2 and OpenFold agree -- because
OpenFold is a reimplementation of AlphaFold2 and shares its failure modes.

The underlying principle (agreement between independent methods is stronger evidence)
is not new to science. What this module contributes is packaging it as an automatic
pipeline step for predicted-structure pocket triage: sequence-identity gating,
cross-model residue mapping, superposition, pocket correspondence matching, and an
architecture-family-aware agreement score that never pools independent and
non-independent agreement into one number.

Method
------
1. **Identity gate.** Align the input sequences and require near-identity (default
   >= 95%). Different proteins are rejected outright rather than compared, because a
   plausible-looking consensus table for two unrelated proteins is worse than an error.
2. **Residue mapping.** Build a residue-to-residue map from the alignment, so that
   models differing in numbering, missing residues or construct boundaries can still
   be compared.
3. **Superposition.** Kabsch-superpose corresponding Ca atoms onto the first model,
   which becomes the reference frame for all centroid comparisons.
4. **Pocket correspondence.** Two pockets from different models are "the same pocket"
   only if *both* criteria agree: their lining-residue sets (mapped through the
   alignment) overlap by at least a Jaccard threshold, *and* their post-superposition
   centroids are within a distance tolerance. Requiring both avoids matching two
   distant cavities that happen to share a few residues, and avoids matching two
   unrelated cavities that happen to be near each other.
5. **Family-aware agreement.** Each source is tagged with an architecture family.
   Cross-family and same-family agreement are computed and reported *separately*,
   optionally weighted by each model's own confidence in the pocket's lining residues.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from Bio import Align

from .errors import ConsensusError, SequenceMismatchError
from .models import (
    ArchitectureFamily,
    ConsensusMember,
    ConsensusParameters,
    ConsensusPocket,
    ConsensusReport,
    PairwiseAlignmentSummary,
    Pocket,
    PocketSet,
    StructureModel,
)
from .parsing import ParsedStructure

ResidueKey = tuple[str, int, str]


# --------------------------------------------------------------------------------------
# Sequence alignment and residue mapping
# --------------------------------------------------------------------------------------


def _aligner() -> Align.PairwiseAligner:
    """A global aligner with affine gaps, suitable for near-identical sequences."""
    aligner = Align.PairwiseAligner()
    aligner.mode = "global"
    aligner.match_score = 2.0
    aligner.mismatch_score = -1.0
    aligner.open_gap_score = -10.0
    aligner.extend_gap_score = -0.5
    # Terminal gaps are free: models often differ only in construct boundaries.
    aligner.target_end_gap_score = 0.0
    aligner.query_end_gap_score = 0.0
    return aligner


@dataclass
class ChainAlignment:
    """Residue-level correspondence between one chain of two structures."""

    chain_a: str
    chain_b: str
    identity: float
    aligned_length: int
    n_identical: int
    #: Maps a residue key in structure B to the corresponding key in structure A.
    b_to_a: dict[ResidueKey, ResidueKey]


def align_chain(
    residues_a: list[dict], residues_b: list[dict], chain_a: str, chain_b: str
) -> ChainAlignment:
    """Align two chains and build their residue-key correspondence.

    Identity is reported as identical aligned columns divided by the length of the
    shorter sequence, so that a model missing a disordered tail is not penalised for
    the missing residues twice.
    """
    seq_a = "".join(r["one_letter"] for r in residues_a)
    seq_b = "".join(r["one_letter"] for r in residues_b)
    if not seq_a or not seq_b:
        raise ConsensusError(
            f"Chain {chain_a}/{chain_b} produced an empty sequence; cannot align."
        )

    alignment = _aligner().align(seq_a, seq_b)[0]
    blocks_a, blocks_b = alignment.aligned

    mapping: dict[ResidueKey, ResidueKey] = {}
    n_identical = 0
    aligned_length = 0

    for (start_a, end_a), (start_b, _end_b) in zip(blocks_a, blocks_b, strict=False):
        for offset in range(end_a - start_a):
            index_a = start_a + offset
            index_b = start_b + offset
            aligned_length += 1
            residue_a = residues_a[index_a]
            residue_b = residues_b[index_b]
            key_a = (residue_a["chain"], residue_a["resseq"], residue_a["icode"])
            key_b = (residue_b["chain"], residue_b["resseq"], residue_b["icode"])
            mapping[key_b] = key_a
            if residue_a["one_letter"] == residue_b["one_letter"]:
                n_identical += 1

    denominator = max(min(len(seq_a), len(seq_b)), 1)
    return ChainAlignment(
        chain_a=chain_a,
        chain_b=chain_b,
        identity=n_identical / denominator,
        aligned_length=aligned_length,
        n_identical=n_identical,
        b_to_a=mapping,
    )


def map_structures(
    reference: ParsedStructure, other: ParsedStructure
) -> tuple[dict[ResidueKey, ResidueKey], float, int]:
    """Map every residue of ``other`` onto ``reference``.

    Chains are paired in sorted chain-ID order. When the two structures have different
    numbers of chains, only the chains that can be paired positionally are used, and
    the caller is expected to surface that in the report notes.
    """
    chains_ref = sorted({r["chain"] for r in reference.residues})
    chains_other = sorted({r["chain"] for r in other.residues})
    n_pairs = min(len(chains_ref), len(chains_other))

    combined: dict[ResidueKey, ResidueKey] = {}
    total_identical = 0
    total_denominator = 0
    total_aligned = 0

    for index in range(n_pairs):
        chain_ref = chains_ref[index]
        chain_other = chains_other[index]
        residues_ref = [r for r in reference.residues if r["chain"] == chain_ref]
        residues_other = [r for r in other.residues if r["chain"] == chain_other]
        chain_alignment = align_chain(residues_ref, residues_other, chain_ref, chain_other)
        combined.update(chain_alignment.b_to_a)
        total_identical += chain_alignment.n_identical
        total_denominator += min(len(residues_ref), len(residues_other))
        total_aligned += chain_alignment.aligned_length

    identity = total_identical / max(total_denominator, 1)
    return combined, identity, total_aligned


# --------------------------------------------------------------------------------------
# Superposition
# --------------------------------------------------------------------------------------


def kabsch(mobile: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """Optimal rigid superposition of ``mobile`` onto ``target``.

    Returns the rotation matrix, the translation vector, and the RMSD after applying
    them. Uses the standard SVD formulation with the reflection correction, so a
    near-degenerate point set cannot produce an improper rotation.
    """
    if mobile.shape != target.shape or mobile.shape[0] < 3:
        raise ConsensusError(
            "Superposition needs at least 3 corresponding atoms in both structures; "
            f"got {mobile.shape[0]}."
        )

    centroid_mobile = mobile.mean(axis=0)
    centroid_target = target.mean(axis=0)
    centred_mobile = mobile - centroid_mobile
    centred_target = target - centroid_target

    covariance = centred_mobile.T @ centred_target
    u_matrix, _, v_transpose = np.linalg.svd(covariance)
    reflection = np.sign(np.linalg.det(v_transpose.T @ u_matrix.T))
    correction = np.diag([1.0, 1.0, reflection])
    rotation = v_transpose.T @ correction @ u_matrix.T

    translation = centroid_target - rotation @ centroid_mobile
    transformed = (rotation @ mobile.T).T + translation
    rmsd = float(np.sqrt(np.mean(np.sum((transformed - target) ** 2, axis=1))))
    return rotation, translation, rmsd


def superpose_onto_reference(
    reference: ParsedStructure,
    other: ParsedStructure,
    mapping: dict[ResidueKey, ResidueKey],
) -> tuple[np.ndarray, np.ndarray, float, int]:
    """Kabsch-superpose ``other`` onto ``reference`` using mapped Cα atoms."""
    reference_ca = {
        (r["chain"], r["resseq"], r["icode"]): r["ca_coord"]
        for r in reference.residues
        if r["ca_coord"] is not None
    }
    other_ca = {
        (r["chain"], r["resseq"], r["icode"]): r["ca_coord"]
        for r in other.residues
        if r["ca_coord"] is not None
    }

    mobile_points: list[np.ndarray] = []
    target_points: list[np.ndarray] = []
    for key_other, key_reference in mapping.items():
        if key_other in other_ca and key_reference in reference_ca:
            mobile_points.append(other_ca[key_other])
            target_points.append(reference_ca[key_reference])

    if len(mobile_points) < 3:
        raise ConsensusError(
            "Fewer than 3 Ca atoms could be put into correspondence between the "
            "structures, so they cannot be superposed. Check that both files contain "
            "backbone atoms."
        )

    rotation, translation, rmsd = kabsch(
        np.array(mobile_points, dtype=float), np.array(target_points, dtype=float)
    )
    return rotation, translation, rmsd, len(mobile_points)


# --------------------------------------------------------------------------------------
# Pocket correspondence
# --------------------------------------------------------------------------------------


@dataclass
class PlacedPocket:
    """A pocket expressed in the reference frame with reference residue keys."""

    structure_index: int
    structure_label: str
    source_id: str
    source_display_name: str
    family: ArchitectureFamily
    pocket: Pocket
    mapped_residues: set[ResidueKey]
    centroid: np.ndarray | None
    mean_confidence: float | None


def _place_pockets(
    structures: list[StructureModel],
    parsed: list[ParsedStructure],
    pocket_sets: list[PocketSet],
    mappings: list[dict[ResidueKey, ResidueKey]],
    transforms: list[tuple[np.ndarray, np.ndarray] | None],
) -> list[PlacedPocket]:
    """Express every pocket of every model in the reference model's frame."""
    placed: list[PlacedPocket] = []
    for index, (structure, pocket_set) in enumerate(
        zip(structures, pocket_sets, strict=True)
    ):
        mapping = mappings[index]
        transform = transforms[index]
        for pocket in pocket_set.pockets:
            mapped = {
                mapping[residue.key] for residue in pocket.residues if residue.key in mapping
            }
            centroid = None
            if pocket.centroid is not None:
                centroid = np.array(pocket.centroid, dtype=float)
                if transform is not None:
                    rotation, translation = transform
                    centroid = rotation @ centroid + translation

            confidences = [r.confidence for r in pocket.residues if r.confidence is not None]
            placed.append(
                PlacedPocket(
                    structure_index=index,
                    structure_label=structure.label,
                    source_id=structure.source.id,
                    source_display_name=structure.source.display_name,
                    family=structure.source.family,
                    pocket=pocket,
                    mapped_residues=mapped,
                    centroid=centroid,
                    mean_confidence=(
                        float(np.mean(confidences)) if confidences else None
                    ),
                )
            )
    return placed


def _jaccard(a: set[ResidueKey], b: set[ResidueKey]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _match_score(
    first: PlacedPocket, second: PlacedPocket, parameters: ConsensusParameters
) -> tuple[bool, float, float, float]:
    """Decide whether two pockets are the same pocket.

    Both criteria must hold: lining-residue Jaccard overlap at or above the threshold,
    **and** post-superposition centroid distance within tolerance. Returns
    ``(matched, combined_score, jaccard, centroid_distance)``.
    """
    jaccard = _jaccard(first.mapped_residues, second.mapped_residues)

    if first.centroid is None or second.centroid is None:
        # Without centroids the geometric criterion cannot be evaluated, so the pair is
        # not matched. Silently falling back to residue overlap alone would weaken the
        # "both criteria agree" guarantee that the method rests on.
        return False, 0.0, jaccard, float("inf")

    distance = float(np.linalg.norm(first.centroid - second.centroid))
    matched = jaccard >= parameters.min_residue_jaccard and distance <= parameters.centroid_tolerance

    # Combined score orders candidate matches when one pocket could join several groups:
    # high overlap and short centroid distance both push it up.
    proximity = max(0.0, 1.0 - distance / max(parameters.centroid_tolerance, 1e-6))
    return matched, 0.7 * jaccard + 0.3 * proximity, jaccard, distance


def _group_pockets(
    placed: list[PlacedPocket], parameters: ConsensusParameters
) -> list[list[int]]:
    """Greedily cluster pockets across models.

    Candidate pairs are sorted by match quality and merged only when the merge keeps at
    most one pocket per input model in a group -- a single model cannot "agree with
    itself" about one pocket, and allowing it would inflate agreement counts.
    """
    candidates: list[tuple[float, int, int]] = []
    for i in range(len(placed)):
        for j in range(i + 1, len(placed)):
            if placed[i].structure_index == placed[j].structure_index:
                continue
            matched, score, _, _ = _match_score(placed[i], placed[j], parameters)
            if matched:
                candidates.append((score, i, j))

    candidates.sort(key=lambda item: (-item[0], item[1], item[2]))

    group_of: dict[int, int] = {index: index for index in range(len(placed))}
    members: dict[int, list[int]] = {index: [index] for index in range(len(placed))}

    def find(index: int) -> int:
        while group_of[index] != index:
            group_of[index] = group_of[group_of[index]]
            index = group_of[index]
        return index

    for _score, i, j in candidates:
        root_i, root_j = find(i), find(j)
        if root_i == root_j:
            continue
        structures_i = {placed[m].structure_index for m in members[root_i]}
        structures_j = {placed[m].structure_index for m in members[root_j]}
        if structures_i & structures_j:
            continue  # merging would put two pockets from one model in the same group
        group_of[root_j] = root_i
        members[root_i].extend(members[root_j])
        del members[root_j]

    groups = [sorted(group) for group in members.values()]
    groups.sort(key=lambda group: (-len(group), min(group)))
    return groups


# --------------------------------------------------------------------------------------
# Agreement scoring
# --------------------------------------------------------------------------------------


def _pair_weight(
    first: PlacedPocket, second: PlacedPocket, use_confidence: bool
) -> float:
    """Weight for one agreeing pair of models.

    With confidence weighting on, a pocket that both models predicted confidently
    contributes more than one that both predicted marginally: the weight is the product
    of the two models' mean pLDDT over the pocket's lining residues, each rescaled to
    0-1. A model with no confidence data for the pocket contributes a neutral 0.5
    rather than silently counting as a full vote.
    """
    if not use_confidence:
        return 1.0
    first_weight = (first.mean_confidence / 100.0) if first.mean_confidence is not None else 0.5
    second_weight = (
        (second.mean_confidence / 100.0) if second.mean_confidence is not None else 0.5
    )
    return max(0.0, min(1.0, first_weight)) * max(0.0, min(1.0, second_weight))


def _possible_pair_counts(
    families: list[ArchitectureFamily],
) -> tuple[int, int]:
    """How many cross-family and same-family model pairs the input set could produce.

    These are the denominators for the two agreement scores, so that a value of 1.0
    means "every pair of models of this kind that *could* have agreed did agree".
    """
    cross = same = 0
    for i in range(len(families)):
        for j in range(i + 1, len(families)):
            if _is_cross_family(families[i], families[j]):
                cross += 1
            else:
                same += 1
    return cross, same


def _is_cross_family(a: ArchitectureFamily, b: ArchitectureFamily) -> bool:
    """True only for two *known* and *different* families.

    ``UNKNOWN`` never counts as cross-family: if the family of a source has not been
    established, its agreement with anything else must not be presented as independent
    evidence.
    """
    if ArchitectureFamily.UNKNOWN in (a, b):
        return False
    return a != b


def _score_group(
    indices: list[int],
    placed: list[PlacedPocket],
    parameters: ConsensusParameters,
    n_models: int,
    possible_cross: int,
    possible_same: int,
    group_id: int,
) -> ConsensusPocket:
    """Turn one matched group into a scored :class:`ConsensusPocket`."""
    group = [placed[index] for index in indices]
    group.sort(key=lambda item: item.structure_index)

    cross_weight = same_weight = 0.0
    n_cross = n_same = 0
    distances: list[float] = []
    jaccards: list[float] = []

    for i in range(len(group)):
        for j in range(i + 1, len(group)):
            weight = _pair_weight(group[i], group[j], parameters.confidence_weighting)
            if _is_cross_family(group[i].family, group[j].family):
                cross_weight += weight
                n_cross += 1
            else:
                same_weight += weight
                n_same += 1
            _, _, jaccard, distance = _match_score(group[i], group[j], parameters)
            jaccards.append(jaccard)
            if np.isfinite(distance):
                distances.append(distance)

    families = []
    for member in group:
        if member.family not in families:
            families.append(member.family)

    cross_agreement = cross_weight / possible_cross if possible_cross else 0.0
    same_agreement = same_weight / possible_same if possible_same else 0.0

    if n_cross:
        evidence_label = "cross-family"
    elif len(group) > 1:
        evidence_label = "same-family-only"
    else:
        evidence_label = "single-model"

    members = [
        ConsensusMember(
            structure_label=member.structure_label,
            source_id=member.source_id,
            source_display_name=member.source_display_name,
            family=member.family,
            pocket_id=member.pocket.id,
            pocket_rank=member.pocket.rank,
            druggability_score=member.pocket.druggability_score,
            score_provenance=member.pocket.score_provenance,
            volume=member.pocket.volume,
            mean_confidence=(
                round(member.mean_confidence, 1) if member.mean_confidence is not None else None
            ),
            centroid=member.pocket.centroid,
        )
        for member in group
    ]

    return ConsensusPocket(
        group_id=group_id,
        members=members,
        n_models_detecting=len(group),
        n_models_total=n_models,
        families_detecting=families,
        cross_family_support=n_cross > 0,
        same_family_support=n_same > 0,
        n_family_pairs_cross=n_cross,
        n_family_pairs_same=n_same,
        cross_family_agreement=round(min(cross_agreement, 1.0), 3),
        same_family_agreement=round(min(same_agreement, 1.0), 3),
        evidence_label=evidence_label,
        mean_pairwise_centroid_distance=(
            round(float(np.mean(distances)), 2) if distances else None
        ),
        mean_residue_jaccard=round(float(np.mean(jaccards)), 3) if jaccards else None,
        consensus_note=_consensus_note(evidence_label, group, n_models),
    )


def _consensus_note(
    evidence_label: str, group: list[PlacedPocket], n_models: int
) -> str:
    """The sentence appended to the pocket's existing confidence caveat."""
    sources = ", ".join(sorted({member.source_display_name for member in group}))
    if evidence_label == "cross-family":
        return (
            f"Detected independently by models from different architecture families "
            f"({sources}). Because these methods do not share an inference pathway, "
            "their agreement is the strongest robustness signal this pipeline can "
            "offer short of experimental validation."
        )
    if evidence_label == "same-family-only":
        return (
            f"Detected by {len(group)} of {n_models} models, but all of them belong to "
            f"the same architecture family ({sources}). Models within a family can "
            "reproduce each other's systematic errors, so this is weaker evidence than "
            "cross-family agreement and should not be read as independent confirmation."
        )
    return (
        f"Detected in only one of the {n_models} supplied models "
        f"({sources}). It may be genuine and missed elsewhere, or it may be an artefact "
        "of that model; either way it carries no consensus support."
    )


# --------------------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------------------


def compute_consensus(
    structures: list[StructureModel],
    parsed: list[ParsedStructure],
    pocket_sets: list[PocketSet],
    parameters: ConsensusParameters | None = None,
) -> ConsensusReport:
    """Run the cross-model consensus stage.

    Raises
    ------
    SequenceMismatchError
        If the supplied structures are not the same protein to within
        ``parameters.min_sequence_identity``.
    ConsensusError
        If fewer than two structures were supplied, or superposition is impossible.
    """
    parameters = parameters or ConsensusParameters()
    if len(structures) < 2:
        raise ConsensusError(
            "Cross-model consensus needs at least 2 structures of the same protein."
        )

    reference_parsed = parsed[0]
    reference_structure = structures[0]

    mappings: list[dict[ResidueKey, ResidueKey]] = []
    transforms: list[tuple[np.ndarray, np.ndarray] | None] = []
    alignments: list[PairwiseAlignmentSummary] = []
    notes: list[str] = []

    for index, (structure, structure_parsed) in enumerate(
        zip(structures, parsed, strict=True)
    ):
        if index == 0:
            mappings.append(
                {
                    (r["chain"], r["resseq"], r["icode"]): (
                        r["chain"],
                        r["resseq"],
                        r["icode"],
                    )
                    for r in reference_parsed.residues
                }
            )
            transforms.append(None)
            continue

        mapping, identity, aligned_length = map_structures(reference_parsed, structure_parsed)

        # Two unrelated sequences align to nothing at all: the optimal global alignment
        # of dissimilar sequences is all-gaps, which leaves no residue correspondence.
        # Caught here rather than letting superposition fail later with a message about
        # backbone atoms, which would send the user looking in the wrong place.
        minimum_residues = min(
            len(reference_parsed.residues), len(structure_parsed.residues)
        )
        if len(mapping) < max(3, 0.1 * minimum_residues):
            raise SequenceMismatchError(
                "The supplied structures could not be put into residue correspondence "
                "at all.\n"
                f"  Aligning '{reference_structure.label}' and '{structure.label}' "
                f"produced {len(mapping)} corresponding residues out of a possible "
                f"{minimum_residues}, at {identity:.1%} identity.\n"
                "  These are not the same protein. Cross-model consensus compares "
                "pockets between independent predictions of one protein; there is "
                "nothing meaningful to compare here."
            )

        if identity < parameters.min_sequence_identity:
            raise SequenceMismatchError(
                "The supplied structures are not the same protein.\n"
                f"  '{reference_structure.label}' and '{structure.label}' share only "
                f"{identity:.1%} sequence identity over {aligned_length} aligned "
                f"residues, below the required {parameters.min_sequence_identity:.0%}.\n"
                "  Cross-model consensus compares pockets between independent "
                "predictions of the SAME protein. Comparing different proteins would "
                "produce a plausible-looking but meaningless table, so Pocketscribe "
                "refuses rather than guessing.\n"
                "  If these really are the same protein (for example two constructs "
                "with different tags), lower the threshold with "
                "--min-identity, e.g. --min-identity 0.8."
            )

        rotation, translation, rmsd, n_atoms = superpose_onto_reference(
            reference_parsed, structure_parsed, mapping
        )
        mappings.append(mapping)
        transforms.append((rotation, translation))
        alignments.append(
            PairwiseAlignmentSummary(
                label_a=reference_structure.label,
                label_b=structure.label,
                identity=round(identity, 4),
                aligned_length=aligned_length,
                rmsd=round(rmsd, 2),
                n_superposed_atoms=n_atoms,
            )
        )

        if rmsd > 5.0:
            notes.append(
                f"'{structure.label}' superposes onto '{reference_structure.label}' with "
                f"a Ca RMSD of {rmsd:.1f} A. The models agree on sequence but differ "
                "substantially in conformation, so pocket correspondence by centroid "
                "distance is less reliable for this pair."
            )

    placed = _place_pockets(structures, parsed, pocket_sets, mappings, transforms)
    if not placed:
        raise ConsensusError("No pockets were detected in any of the supplied structures.")

    missing_centroids = sum(1 for item in placed if item.centroid is None)
    if missing_centroids:
        notes.append(
            f"{missing_centroids} pocket(s) had no centroid and could not take part in "
            "geometric matching; they are reported as single-model pockets."
        )

    families = [structure.source.family for structure in structures]
    possible_cross, possible_same = _possible_pair_counts(families)

    if possible_cross == 0:
        notes.append(
            "All supplied models belong to the same architecture family, so no "
            "cross-family (independent-method) comparison was possible. Adding a model "
            "from a different family -- for example an ESMFold prediction alongside "
            "AlphaFold2/OpenFold/ColabFold models -- is what makes this stage "
            "informative."
        )

    groups = _group_pockets(placed, parameters)
    consensus_pockets = [
        _score_group(
            indices=indices,
            placed=placed,
            parameters=parameters,
            n_models=len(structures),
            possible_cross=possible_cross,
            possible_same=possible_same,
            group_id=group_id,
        )
        for group_id, indices in enumerate(groups, start=1)
    ]

    consensus_pockets.sort(
        key=lambda group: (
            group.cross_family_support,
            group.cross_family_agreement,
            group.n_models_detecting,
            group.same_family_agreement,
        ),
        reverse=True,
    )
    for position, group in enumerate(consensus_pockets, start=1):
        group.group_id = position

    unique_families: list[ArchitectureFamily] = []
    for family in families:
        if family not in unique_families:
            unique_families.append(family)

    return ConsensusReport(
        parameters=parameters,
        structure_labels=[structure.label for structure in structures],
        alignments=alignments,
        groups=consensus_pockets,
        n_cross_family_pockets=sum(1 for g in consensus_pockets if g.cross_family_support),
        n_same_family_only_pockets=sum(
            1 for g in consensus_pockets if g.evidence_label == "same-family-only"
        ),
        n_single_model_pockets=sum(
            1 for g in consensus_pockets if g.evidence_label == "single-model"
        ),
        families_present=unique_families,
        cross_family_comparison_possible=possible_cross > 0,
        notes=notes,
    )


def apply_consensus_to_pockets(
    report: ConsensusReport, structures: list[StructureModel], pocket_sets: list[PocketSet]
) -> None:
    """Fold consensus findings into the confidence caveat already on each pocket.

    Consensus deliberately does not create a second, disconnected score: it extends the
    sentence a reader is already looking at next to the pocket.
    """
    index_by_label = {structure.label: position for position, structure in enumerate(structures)}

    for group in report.groups:
        for member in group.members:
            position = index_by_label.get(member.structure_label)
            if position is None:
                continue
            for pocket in pocket_sets[position].pockets:
                if pocket.id == member.pocket_id:
                    pocket.confidence.text = (
                        f"{pocket.confidence.text} Cross-model consensus: "
                        f"{group.consensus_note}"
                    ).strip()
                    break
