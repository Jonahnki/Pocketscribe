"""Structured data model for the Pocketscribe pipeline.

Every pipeline stage consumes and produces validated pydantic models rather than
loose dictionaries. Two properties of this module matter architecturally:

1. The narrative synthesis layer (``pocketscribe.narrative``) is handed *only*
   objects defined here, serialised to JSON -- never raw coordinates and never a
   file handle. That is what keeps the optional API-backed layer isolated from the
   deterministic core.
2. Confidence is a first-class field on pockets, not an afterthought. A
   :class:`Pocket` cannot be constructed without a place to record what the
   prediction confidence was in the residues lining it.
"""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

# --------------------------------------------------------------------------------------
# Enumerations
# --------------------------------------------------------------------------------------


class SupportTier(str, Enum):
    """How completely Pocketscribe supports a given structure-prediction source."""

    TIER_1 = "tier-1"
    """Fully supported: parsed, confidence-extracted, auto-detected and unit tested."""

    TIER_2_REFERENCE = "tier-2-reference"
    """Implemented reference adapter for a non-legacy output convention."""

    TIER_2_STUB = "tier-2-stub"
    """Adapter interface declared and documented; parsing not implemented.

    Selecting one of these raises :class:`~pocketscribe.errors.UnsupportedSourceError`.
    """


class ArchitectureFamily(str, Enum):
    """Broad methodological family of a structure-prediction model.

    Used by the cross-model consensus module to distinguish agreement between
    genuinely independent methods from agreement between close relatives that may
    simply reproduce each other's errors.
    """

    MSA_COEVOLUTION = "msa-coevolution"
    """Uses a multiple sequence alignment / co-evolutionary signal (AF2, OpenFold,
    ColabFold, Boltz)."""

    SINGLE_SEQUENCE_LM = "single-sequence-lm"
    """Protein language model operating on a single sequence, no MSA (ESMFold)."""

    UNKNOWN = "unknown"
    """Family not established; treated conservatively (never counted as cross-family)."""


class ConfidenceMetric(str, Enum):
    """Which per-residue confidence quantity a source reports."""

    PLDDT = "pLDDT"
    """Predicted local distance difference test, 0-100 (higher is better)."""

    PLDDT_FRACTIONAL = "pLDDT (0-1)"
    """pLDDT reported on a 0-1 scale; normalised to 0-100 on ingest."""


class CaveatLevel(str, Enum):
    """Severity of the confidence caveat attached to a pocket."""

    NONE = "none"
    LOW = "low"
    MODERATE = "moderate"
    SEVERE = "severe"


class ScoreProvenance(str, Enum):
    """Where a druggability score came from.

    This exists so the report can never present a heuristic score as if it were
    fpocket's published, validated druggability score.
    """

    FPOCKET = "fpocket"
    """fpocket's own druggability score (Schmidtke & Barril 2010)."""

    BUILTIN_HEURISTIC = "builtin-heuristic"
    """Pocketscribe's transparent fallback heuristic. NOT a validated druggability
    model; for pipeline demonstration and offline testing only."""


# --------------------------------------------------------------------------------------
# Structure-level models
# --------------------------------------------------------------------------------------


class SourceInfo(BaseModel):
    """Static description of a structure-prediction source."""

    model_config = ConfigDict(frozen=True)

    id: str = Field(description="CLI identifier, e.g. 'alphafold2'.")
    display_name: str = Field(description="Human-readable name used in the report.")
    tier: SupportTier
    family: ArchitectureFamily
    metric: ConfidenceMetric = ConfidenceMetric.PLDDT
    uses_msa: bool = Field(
        default=True,
        description="False for single-sequence methods; surfaced as a report caveat.",
    )
    notes: str = Field(default="", description="Source-specific interpretation guidance.")
    citation_key: str = Field(default="", description="Key into the README 'Built on' table.")


class ResidueConfidence(BaseModel):
    """Per-residue confidence for one residue of one model."""

    chain: str
    resseq: int
    icode: str = " "
    resname: str
    value: float = Field(description="Confidence on a 0-100 scale (pLDDT convention).")

    @property
    def key(self) -> tuple[str, int, str]:
        """Stable identity of this residue within its structure."""
        return (self.chain, self.resseq, self.icode)

    @property
    def label(self) -> str:
        """Short human-readable residue label, e.g. ``A:LEU45``."""
        return f"{self.chain}:{self.resname}{self.resseq}{self.icode.strip()}"


class LowConfidenceRegion(BaseModel):
    """A contiguous run of residues below a confidence threshold."""

    chain: str
    start: int
    end: int
    threshold: float
    length: int
    mean_value: float

    @property
    def label(self) -> str:
        return f"{self.chain}:{self.start}-{self.end}"


class ConfidenceProfile(BaseModel):
    """Whole-structure summary of per-residue prediction confidence."""

    metric: ConfidenceMetric
    residues: list[ResidueConfidence]
    mean: float
    median: float
    minimum: float
    maximum: float
    fraction_below_70: float
    fraction_below_50: float
    regions_below_70: list[LowConfidenceRegion] = Field(default_factory=list)
    regions_below_50: list[LowConfidenceRegion] = Field(default_factory=list)

    def value_for(self, chain: str, resseq: int, icode: str = " ") -> float | None:
        """Look up the confidence of a single residue, or ``None`` if absent."""
        for residue in self.residues:
            if residue.chain == chain and residue.resseq == resseq and residue.icode == icode:
                return residue.value
        return None


class ChainBreak(BaseModel):
    """A gap in the backbone between two consecutive modelled residues."""

    chain: str
    after_resseq: int
    before_resseq: int
    ca_distance: float


class QCReport(BaseModel):
    """Structure quality-control summary (pipeline stage 1)."""

    n_chains: int
    n_residues: int
    n_atoms: int
    chain_ids: list[str]
    sequence_length_by_chain: dict[str, int]
    chain_breaks: list[ChainBreak] = Field(default_factory=list)
    numbering_gaps: dict[str, list[tuple[int, int]]] = Field(default_factory=dict)
    nonstandard_residues: list[str] = Field(default_factory=list)
    has_hydrogens: bool = False
    notes: list[str] = Field(default_factory=list)


class StructureModel(BaseModel):
    """One parsed input structure together with its provenance and confidence."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    path: str
    file_format: str = Field(description="'pdb' or 'mmcif'.")
    source: SourceInfo
    source_was_auto_detected: bool = False
    detection_evidence: list[str] = Field(default_factory=list)
    sequences: dict[str, str] = Field(description="One-letter sequence per chain ID.")
    confidence: ConfidenceProfile
    qc: QCReport
    label: str = Field(default="", description="Short label used in consensus tables.")

    @property
    def total_residues(self) -> int:
        return self.qc.n_residues


# --------------------------------------------------------------------------------------
# Pocket models
# --------------------------------------------------------------------------------------


class PocketResidue(BaseModel):
    """A residue lining a pocket, annotated with its prediction confidence."""

    chain: str
    resseq: int
    icode: str = " "
    resname: str
    confidence: float | None = None

    @property
    def key(self) -> tuple[str, int, str]:
        return (self.chain, self.resseq, self.icode)

    @property
    def label(self) -> str:
        return f"{self.chain}:{self.resname}{self.resseq}{self.icode.strip()}"


class PocketConfidenceAnnotation(BaseModel):
    """How trustworthy the *prediction underlying* a pocket is.

    This is deliberately separate from the druggability score: a pocket can be
    highly druggable by geometry and simultaneously untrustworthy because the
    residues forming it were predicted with low confidence.
    """

    mean_confidence: float | None = None
    min_confidence: float | None = None
    fraction_below_70: float = 0.0
    fraction_below_50: float = 0.0
    n_residues_scored: int = 0
    overlapping_low_confidence_regions: list[str] = Field(default_factory=list)
    level: CaveatLevel = CaveatLevel.NONE
    text: str = ""


class Pocket(BaseModel):
    """A single detected cavity with geometry, scoring and confidence caveat."""

    id: int = Field(description="1-based pocket index as reported by the backend.")
    rank: int = Field(description="1-based rank after sorting by druggability.")
    volume: float | None = None
    druggability_score: float | None = None
    score_provenance: ScoreProvenance = ScoreProvenance.FPOCKET
    fpocket_score: float | None = Field(
        default=None, description="fpocket's own overall pocket score, when available."
    )
    hydrophobicity_score: float | None = None
    polarity_score: float | None = None
    n_alpha_spheres: int | None = None
    mean_alpha_sphere_radius: float | None = None
    total_sasa: float | None = None
    centroid: tuple[float, float, float] | None = None
    residues: list[PocketResidue] = Field(default_factory=list)
    confidence: PocketConfidenceAnnotation = Field(default_factory=PocketConfidenceAnnotation)
    extra: dict[str, float] = Field(default_factory=dict)

    @property
    def residue_keys(self) -> set[tuple[str, int, str]]:
        return {r.key for r in self.residues}


class PocketSet(BaseModel):
    """All pockets detected in one structure by one backend."""

    backend: str
    backend_version: str = "unknown"
    backend_is_fpocket: bool = True
    structure_label: str = ""
    pockets: list[Pocket] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)

    def top(self, n: int) -> list[Pocket]:
        return self.pockets[:n]


# --------------------------------------------------------------------------------------
# Cross-model consensus models
# --------------------------------------------------------------------------------------


class PairwiseAlignmentSummary(BaseModel):
    """Sequence-level agreement between two input structures."""

    label_a: str
    label_b: str
    identity: float = Field(description="Fraction of aligned columns that are identical (0-1).")
    aligned_length: int
    rmsd: float | None = Field(
        default=None, description="Ca RMSD after Kabsch superposition, in Angstrom."
    )
    n_superposed_atoms: int | None = None


class ConsensusMember(BaseModel):
    """One model's contribution to a consensus pocket group."""

    structure_label: str
    source_id: str
    source_display_name: str
    family: ArchitectureFamily
    pocket_id: int
    pocket_rank: int
    druggability_score: float | None = None
    score_provenance: ScoreProvenance = ScoreProvenance.FPOCKET
    volume: float | None = None
    mean_confidence: float | None = None
    centroid: tuple[float, float, float] | None = None


class ConsensusPocket(BaseModel):
    """A pocket matched across two or more input models.

    ``cross_family_support`` and ``same_family_support`` are reported separately and
    are never pooled into one undifferentiated score: agreement between an MSA-based
    and a single-sequence model is stronger evidence than agreement between two
    members of the same family, which may share systematic errors.
    """

    group_id: int
    members: list[ConsensusMember]
    n_models_detecting: int
    n_models_total: int
    families_detecting: list[ArchitectureFamily]
    cross_family_support: bool
    same_family_support: bool
    n_family_pairs_cross: int = 0
    n_family_pairs_same: int = 0
    cross_family_agreement: float = Field(
        default=0.0, description="Confidence-weighted cross-family agreement, 0-1."
    )
    same_family_agreement: float = Field(
        default=0.0, description="Confidence-weighted same-family agreement, 0-1."
    )
    evidence_label: str = Field(
        default="single-model",
        description="'cross-family' | 'same-family-only' | 'single-model'.",
    )
    mean_pairwise_centroid_distance: float | None = None
    mean_residue_jaccard: float | None = None
    consensus_note: str = ""


class ConsensusParameters(BaseModel):
    """Tunable thresholds for the consensus module, recorded in the methods appendix."""

    min_sequence_identity: float = 0.95
    centroid_tolerance: float = 8.0
    min_residue_jaccard: float = 0.25
    confidence_weighting: bool = True


class ConsensusReport(BaseModel):
    """Full output of the cross-model consensus stage."""

    parameters: ConsensusParameters
    structure_labels: list[str]
    alignments: list[PairwiseAlignmentSummary]
    groups: list[ConsensusPocket]
    n_cross_family_pockets: int = 0
    n_same_family_only_pockets: int = 0
    n_single_model_pockets: int = 0
    families_present: list[ArchitectureFamily] = Field(default_factory=list)
    cross_family_comparison_possible: bool = False
    notes: list[str] = Field(default_factory=list)


# --------------------------------------------------------------------------------------
# MD setup models
# --------------------------------------------------------------------------------------


class GeneratedFile(BaseModel):
    """One file written by the MD setup generator."""

    name: str
    path: str
    kind: str = Field(description="'structure' | 'mdp' | 'script' | 'protocol' | 'notes'.")
    description: str = ""


class MDSetupResult(BaseModel):
    """Everything the MD setup stage produced (it never runs a simulation)."""

    output_dir: str
    force_field: str
    water_model: str
    box_shape: str
    box_padding_nm: float
    salt_concentration_M: float
    target_pocket_rank: int | None = None
    target_pocket_id: int | None = None
    files: list[GeneratedFile] = Field(default_factory=list)
    removed_heteroatoms: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


# --------------------------------------------------------------------------------------
# Narrative + top-level analysis models
# --------------------------------------------------------------------------------------


class NarrativeResult(BaseModel):
    """Output of the optional, API-gated narrative synthesis layer."""

    enabled: bool = False
    text: str = ""
    model: str = ""
    skipped_reason: str = ""
    prompt_token_estimate: int | None = None


class AnalysisResult(BaseModel):
    """The complete deterministic analysis, ready for rendering or JSON export."""

    tool_version: str
    generated_at: str
    structures: list[StructureModel]
    pocket_sets: list[PocketSet]
    consensus: ConsensusReport | None = None
    md_setup: MDSetupResult | None = None
    narrative: NarrativeResult = Field(default_factory=NarrativeResult)
    top_pockets_requested: int = 3
    warnings: list[str] = Field(default_factory=list)
    parameters: dict[str, Any] = Field(default_factory=dict)

    @property
    def is_consensus_run(self) -> bool:
        return len(self.structures) > 1

    @property
    def primary_structure(self) -> StructureModel:
        return self.structures[0]

    @property
    def primary_pockets(self) -> PocketSet:
        return self.pocket_sets[0]
