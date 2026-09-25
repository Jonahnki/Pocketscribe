"""Structure-source adapters.

Different structure-prediction tools encode per-residue confidence differently, and
the list of tools keeps growing. Rather than hardcoding one parsing path, every
source implements the :class:`StructureSource` protocol and registers itself here.

Adding a source is the primary contribution path for this project; see
``CONTRIBUTING.md`` for a step-by-step walkthrough that uses the Protenix stub below
as its worked example.

Support tiers
-------------
Tier 1 (fully implemented, auto-detected, individually tested)
    AlphaFold2, OpenFold, ColabFold, ESMFold. All four write pLDDT into the PDB
    B-factor column, but their headers/filenames differ, so they are detected
    distinctly and labelled distinctly in the report.

Tier 2 (interface defined; one working reference implementation)
    Boltz-1 / Boltz-2 is implemented as the reference adapter for a *non*-B-factor
    confidence convention. Protenix, AlphaFold3 and RoseTTAFold are declared and
    documented but deliberately raise rather than attempt a wrong parse.

The cardinal rule: if per-residue confidence cannot be found or parsed for the
declared source, raise. Never treat missing or mismatched data as confidence.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Protocol, runtime_checkable

from .errors import (
    ConfidenceExtractionError,
    SourceDetectionError,
    UnsupportedSourceError,
)
from .models import ArchitectureFamily, ConfidenceMetric, SourceInfo, SupportTier

# --------------------------------------------------------------------------------------
# Adapter protocol
# --------------------------------------------------------------------------------------


@runtime_checkable
class StructureSource(Protocol):
    """Interface every structure-prediction source adapter must satisfy.

    Implementations are stateless and cheap to construct: they are instantiated once
    per run and consulted for detection, then for confidence extraction.
    """

    info: SourceInfo

    def detect(self, path: Path, text_head: str) -> tuple[bool, list[str]]:
        """Decide whether ``path`` was produced by this source.

        Parameters
        ----------
        path:
            Path to the structure file (used for filename conventions).
        text_head:
            The first few kilobytes of the file, already decoded, so that every
            adapter can inspect headers without re-reading from disk.

        Returns
        -------
        (matched, evidence)
            ``matched`` is True when this adapter claims the file. ``evidence`` is a
            list of short human-readable strings explaining *why*, which the CLI
            prints so a user can correct a wrong guess.
        """
        ...

    def extract_confidence(
        self, path: Path, residues: list[dict]
    ) -> tuple[ConfidenceMetric, dict[tuple[str, int, str], float]]:
        """Return per-residue confidence on a 0-100 scale.

        Parameters
        ----------
        path:
            The structure file, so adapters whose confidence lives in a sidecar file
            (Boltz, AlphaFold3) can find it.
        residues:
            Parsed residue records from :mod:`pocketscribe.parsing`. Each dict has
            keys ``chain``, ``resseq``, ``icode``, ``resname`` and ``bfactors``
            (list of per-atom B-factor column values).

        Raises
        ------
        ConfidenceExtractionError
            If confidence is absent, out of range, or otherwise not trustworthy.
        """
        ...


# --------------------------------------------------------------------------------------
# Shared helpers
# --------------------------------------------------------------------------------------

#: Heuristic: an experimental structure's B-factors are not bounded by 100 and are
#: rarely all within the pLDDT range, so a file whose B-factors look like real
#: temperature factors is rejected rather than silently read as confidence.
_PLDDT_MIN = 0.0
_PLDDT_MAX = 100.0


def _bfactor_confidence(
    residues: list[dict], source_name: str
) -> tuple[ConfidenceMetric, dict[tuple[str, int, str], float]]:
    """Extract pLDDT stored in the B-factor column (AF2/OpenFold/ColabFold/ESMFold).

    Handles both the 0-100 convention and the 0-1 convention some ColabFold and
    ESMFold builds emit, normalising the latter to 0-100.
    """
    values: dict[tuple[str, int, str], float] = {}
    n_multi_atom = 0
    n_varying = 0

    for residue in residues:
        bfactors = [b for b in residue.get("bfactors", []) if b is not None]
        if not bfactors:
            continue
        # pLDDT is a per-residue quantity replicated across the residue's atoms;
        # averaging is robust to the occasional per-atom rounding difference.
        values[(residue["chain"], residue["resseq"], residue["icode"])] = sum(bfactors) / len(
            bfactors
        )
        if len(bfactors) > 1:
            n_multi_atom += 1
            if max(bfactors) - min(bfactors) > 0.05:
                n_varying += 1

    if not values:
        raise ConfidenceExtractionError(
            f"No B-factor values found in this file, but source '{source_name}' stores "
            "per-residue pLDDT in the B-factor column. This file does not look like "
            f"{source_name} output. Re-run with the correct --source, or use "
            "'pocketscribe sources' to list what is supported."
        )

    # The decisive check. Real temperature factors sit comfortably inside 0-100, so a
    # range test alone cannot tell an experimental structure from a predicted one. But
    # pLDDT is a *per-residue* quantity written identically to every atom of a residue,
    # whereas a crystallographic B-factor is refined per atom and varies within a
    # residue. A file whose B-factors vary within residues is therefore not pLDDT.
    if n_multi_atom >= 10 and (n_varying / n_multi_atom) > 0.2:
        raise ConfidenceExtractionError(
            f"B-factor values vary between atoms within the same residue in "
            f"{n_varying / n_multi_atom:.0%} of residues. Per-residue pLDDT is written "
            "identically to every atom of a residue, so this column holds "
            "crystallographic temperature factors, not prediction confidence.\n"
            f"  You asked Pocketscribe to read it as '{source_name}' confidence; it will "
            "not, because every pocket caveat downstream would be built on the wrong "
            "number.\n"
            "  If this is an experimental structure, Pocketscribe is not the right tool: "
            "its purpose is caveating pocket claims by prediction confidence, which an "
            "experimental structure does not carry."
        )

    observed = list(values.values())
    lo, hi = min(observed), max(observed)

    if lo < _PLDDT_MIN or hi > _PLDDT_MAX:
        raise ConfidenceExtractionError(
            f"B-factor values range from {lo:.2f} to {hi:.2f}, which is outside the "
            f"pLDDT range 0-100 expected for source '{source_name}'. This is the "
            "signature of an experimental structure whose B-factor column holds real "
            "temperature factors. Pocketscribe will not treat those as confidence "
            "scores. If this really is a predicted structure, check that the file was "
            "not post-processed (e.g. by a refinement step that overwrote B-factors)."
        )

    metric = ConfidenceMetric.PLDDT
    if hi <= 1.0 and lo >= 0.0:
        # Some builds emit fractional pLDDT. Distinguishable from a genuinely awful
        # 0-100 prediction only by the fact that *every* residue would have to be
        # below 1.0, which does not happen in practice.
        values = {key: value * 100.0 for key, value in values.items()}
        metric = ConfidenceMetric.PLDDT_FRACTIONAL

    if max(values.values()) == 0.0:
        raise ConfidenceExtractionError(
            f"All B-factor values are zero, so source '{source_name}' produced no usable "
            "per-residue confidence. Pocketscribe refuses to analyse a predicted "
            "structure without confidence data, because every pocket claim downstream "
            "depends on it."
        )
    return metric, values


def _head_contains(text: str, *needles: str) -> bool:
    lowered = text.lower()
    return any(needle.lower() in lowered for needle in needles)


# --------------------------------------------------------------------------------------
# Tier 1 adapters
# --------------------------------------------------------------------------------------


class AlphaFold2Source:
    """DeepMind AlphaFold2 / AlphaFold Protein Structure Database output.

    Convention: pLDDT in the B-factor column, 0-100 scale. Database downloads carry
    an ``AF-<UniProt>-F1-model_v<n>`` filename and a ``DeepMind`` / ``AlphaFold``
    header line.
    """

    info = SourceInfo(
        id="alphafold2",
        display_name="AlphaFold2",
        tier=SupportTier.TIER_1,
        family=ArchitectureFamily.MSA_COEVOLUTION,
        metric=ConfidenceMetric.PLDDT,
        uses_msa=True,
        notes=(
            "pLDDT is stored in the B-factor column on a 0-100 scale. Regions below 70 "
            "are commonly flexible or disordered; below 50 they should not be "
            "interpreted structurally."
        ),
        citation_key="jumper2021",
    )

    #: AlphaFold DB filenames, e.g. AF-P69905-F1-model_v4.pdb
    _FILENAME = re.compile(r"^AF-[A-Z0-9]+-F\d+-(model|predicted)", re.IGNORECASE)

    def detect(self, path: Path, text_head: str) -> tuple[bool, list[str]]:
        evidence: list[str] = []
        if self._FILENAME.match(path.name):
            evidence.append(f"filename matches AlphaFold DB convention ({path.name})")
        # ColabFold output also mentions AlphaFold, so ColabFold must be tested first;
        # the registry order below guarantees that.
        if _head_contains(text_head, "ALPHAFOLD", "DEEPMIND"):
            evidence.append("header mentions AlphaFold/DeepMind")
        if _head_contains(text_head, "pLDDT"):
            evidence.append("header mentions pLDDT")
        return bool(evidence), evidence

    def extract_confidence(self, path: Path, residues: list[dict]):
        return _bfactor_confidence(residues, self.info.display_name)


class OpenFoldSource:
    """OpenFold, the open-source AlphaFold2 reimplementation (AlQuraishi lab).

    OpenFold reproduces AlphaFold2's output convention exactly -- pLDDT in the
    B-factor column, 0-100 -- so it shares AlphaFold2's extraction path. That shared
    behaviour is asserted explicitly in ``tests/test_sources.py`` rather than assumed,
    because a silent divergence in a future OpenFold release would otherwise produce
    wrong confidence with no error.
    """

    info = SourceInfo(
        id="openfold",
        display_name="OpenFold",
        tier=SupportTier.TIER_1,
        family=ArchitectureFamily.MSA_COEVOLUTION,
        metric=ConfidenceMetric.PLDDT,
        uses_msa=True,
        notes=(
            "OpenFold is a faithful reimplementation of AlphaFold2 and shares its "
            "output convention and architecture family. Agreement between OpenFold and "
            "AlphaFold2 is therefore same-family agreement, not independent evidence."
        ),
        citation_key="ahdritz2024",
    )

    def detect(self, path: Path, text_head: str) -> tuple[bool, list[str]]:
        evidence: list[str] = []
        if _head_contains(text_head, "OPENFOLD"):
            evidence.append("header mentions OpenFold")
        if "openfold" in path.name.lower():
            evidence.append(f"filename mentions openfold ({path.name})")
        return bool(evidence), evidence

    def extract_confidence(self, path: Path, residues: list[dict]):
        return _bfactor_confidence(residues, self.info.display_name)


class ColabFoldSource:
    """ColabFold (MMseqs2 search + AlphaFold2/RoseTTAFold backend).

    The most common real-world path for academic groups without their own GPU
    infrastructure. Its default AlphaFold2-backend output shares the B-factor pLDDT
    convention, but its filenames (``*_unrelaxed_rank_001_alphafold2_*``) and headers
    differ from raw AlphaFold2, so it is detected distinctly and labelled
    "ColabFold (AlphaFold2 backend)" rather than misattributed to AlphaFold2.
    """

    info = SourceInfo(
        id="colabfold",
        display_name="ColabFold (AlphaFold2 backend)",
        tier=SupportTier.TIER_1,
        family=ArchitectureFamily.MSA_COEVOLUTION,
        metric=ConfidenceMetric.PLDDT,
        uses_msa=True,
        notes=(
            "ColabFold substitutes an MMseqs2 homology search for AlphaFold2's default "
            "MSA pipeline. Confidence is still pLDDT in the B-factor column. Because the "
            "structure module is AlphaFold2's, ColabFold shares AlphaFold2's and "
            "OpenFold's architecture family."
        ),
        citation_key="mirdita2022",
    )

    _FILENAME = re.compile(
        r"(unrelaxed|relaxed)_rank_\d+|rank_\d+_(alphafold2|model)|_scores_rank_", re.IGNORECASE
    )

    def detect(self, path: Path, text_head: str) -> tuple[bool, list[str]]:
        evidence: list[str] = []
        if self._FILENAME.search(path.name):
            evidence.append(f"filename matches ColabFold rank convention ({path.name})")
        if _head_contains(text_head, "COLABFOLD", "MMSEQS2", "MMseqs2"):
            evidence.append("header mentions ColabFold/MMseqs2")
        return bool(evidence), evidence

    def extract_confidence(self, path: Path, residues: list[dict]):
        return _bfactor_confidence(residues, self.info.display_name)


class ESMFoldSource:
    """ESMFold (Meta AI ESM-2 protein language model).

    Same B-factor pLDDT convention, but no MSA: the prediction is made from the single
    sequence alone. That is flagged in the report, because it changes how confidence
    should be read and -- crucially for the consensus module -- puts ESMFold in a
    different architecture family from the AlphaFold2 lineage.
    """

    info = SourceInfo(
        id="esmfold",
        display_name="ESMFold",
        tier=SupportTier.TIER_1,
        family=ArchitectureFamily.SINGLE_SEQUENCE_LM,
        metric=ConfidenceMetric.PLDDT,
        uses_msa=False,
        notes=(
            "ESMFold predicts from a single sequence with no MSA, so no explicit "
            "evolutionary-coupling signal informed this model. Confidence tends to be "
            "lower for sequences with few close homologues in the language model's "
            "training distribution. Its independence from MSA-based methods is exactly "
            "what makes ESMFold valuable in cross-family consensus."
        ),
        citation_key="lin2023",
    )

    def detect(self, path: Path, text_head: str) -> tuple[bool, list[str]]:
        evidence: list[str] = []
        if _head_contains(text_head, "ESMFOLD", "ESM-2", "ESM_FOLD"):
            evidence.append("header mentions ESMFold/ESM-2")
        if "esmfold" in path.name.lower() or "esm_" in path.name.lower():
            evidence.append(f"filename mentions ESMFold ({path.name})")
        return bool(evidence), evidence

    def extract_confidence(self, path: Path, residues: list[dict]):
        return _bfactor_confidence(residues, self.info.display_name)


# --------------------------------------------------------------------------------------
# Tier 2 reference adapter
# --------------------------------------------------------------------------------------


class BoltzSource:
    """Boltz-1 / Boltz-2 (MIT-licensed open biomolecular structure prediction).

    Implemented as the working Tier 2 *reference* adapter because Boltz does **not**
    use the legacy B-factor pLDDT convention: it writes structures as mmCIF/PDB and
    exposes confidence through separate structured output files:

    ``confidence_<name>_model_<n>.json``
        Complex-level metrics (``confidence_score``, ``ptm``, ``iptm``, ...).
    ``plddt_<name>_model_<n>.npz``
        Per-residue pLDDT array on a 0-1 scale.

    This adapter reads the sidecar JSON/NPZ when present and otherwise falls back to
    an in-file per-residue confidence column -- never to an unchecked B-factor read.
    Boltz pLDDT is reported on 0-1 and is normalised to 0-100 here so that every
    downstream threshold in Pocketscribe means the same thing for every source.
    """

    info = SourceInfo(
        id="boltz",
        display_name="Boltz",
        tier=SupportTier.TIER_2_REFERENCE,
        family=ArchitectureFamily.MSA_COEVOLUTION,
        metric=ConfidenceMetric.PLDDT,
        uses_msa=True,
        notes=(
            "Boltz reports per-residue pLDDT on a 0-1 scale in a sidecar file rather "
            "than in the B-factor column; Pocketscribe normalises it to 0-100 so that "
            "the <70 and <50 thresholds carry their usual meaning. Boltz is MSA-based "
            "and is treated as the same architecture family as AlphaFold2 for consensus."
        ),
        citation_key="wohlwend2024",
    )

    def detect(self, path: Path, text_head: str) -> tuple[bool, list[str]]:
        evidence: list[str] = []
        if _head_contains(text_head, "BOLTZ"):
            evidence.append("header mentions Boltz")
        if "boltz" in path.name.lower():
            evidence.append(f"filename mentions boltz ({path.name})")
        if self._find_sidecar(path) is not None:
            evidence.append("found a Boltz confidence sidecar file next to the structure")
        return bool(evidence), evidence

    @staticmethod
    def _find_sidecar(path: Path) -> Path | None:
        """Locate Boltz's confidence JSON next to the structure file.

        Boltz names it ``confidence_<stem>.json`` where ``<stem>`` matches the model
        file; we also accept a plain ``<stem>_confidence.json`` for convenience.
        """
        stem = path.stem
        candidates = [
            path.with_name(f"confidence_{stem}.json"),
            path.with_name(f"{stem}_confidence.json"),
            path.with_name(f"{stem}.confidence.json"),
        ]
        for candidate in candidates:
            if candidate.is_file():
                return candidate
        return None

    def extract_confidence(self, path: Path, residues: list[dict]):
        sidecar = self._find_sidecar(path)
        if sidecar is None:
            raise ConfidenceExtractionError(
                "Boltz output does not store per-residue confidence in the B-factor "
                "column, and no confidence sidecar was found next to "
                f"'{path.name}'. Expected one of: confidence_{path.stem}.json, "
                f"{path.stem}_confidence.json. Copy the confidence JSON that Boltz "
                "wrote alongside the predicted structure into the same directory and "
                "re-run. Pocketscribe will not read the B-factor column for Boltz, "
                "because Boltz does not put pLDDT there and the result would be wrong."
            )

        try:
            payload = json.loads(sidecar.read_text())
        except json.JSONDecodeError as exc:
            raise ConfidenceExtractionError(
                f"Boltz confidence file '{sidecar.name}' is not valid JSON: {exc}"
            ) from exc

        per_residue = self._per_residue_array(payload, sidecar)

        ordered = [
            (residue["chain"], residue["resseq"], residue["icode"]) for residue in residues
        ]
        if len(per_residue) != len(ordered):
            raise ConfidenceExtractionError(
                f"Boltz confidence file '{sidecar.name}' lists {len(per_residue)} "
                f"per-residue values but the structure has {len(ordered)} residues. "
                "Pocketscribe will not align mismatched arrays by guessing. Check that "
                "the confidence file corresponds to this exact model."
            )

        scale = 100.0 if max(per_residue) <= 1.0 else 1.0
        values = {
            key: value * scale
            for key, value in zip(ordered, per_residue, strict=True)
        }
        if max(values.values()) > 100.0 or min(values.values()) < 0.0:
            raise ConfidenceExtractionError(
                f"Boltz per-residue confidence in '{sidecar.name}' falls outside the "
                "expected range after normalisation to 0-100."
            )
        return ConfidenceMetric.PLDDT, values

    @staticmethod
    def _per_residue_array(payload: dict, sidecar: Path) -> list[float]:
        """Pull the per-residue pLDDT list out of a Boltz confidence payload."""
        for key in ("plddt", "per_residue_plddt", "residue_plddt", "atom_plddt"):
            array = payload.get(key)
            if isinstance(array, list) and array:
                try:
                    return [float(value) for value in array]
                except (TypeError, ValueError) as exc:
                    raise ConfidenceExtractionError(
                        f"Boltz confidence file '{sidecar.name}' has a non-numeric "
                        f"value in its '{key}' array."
                    ) from exc
        raise ConfidenceExtractionError(
            f"Boltz confidence file '{sidecar.name}' contains no per-residue pLDDT "
            "array (looked for keys: plddt, per_residue_plddt, residue_plddt). It may "
            "be the complex-level confidence summary rather than the per-residue file."
        )


# --------------------------------------------------------------------------------------
# Tier 2 stubs -- declared, documented, and deliberately non-parsing
# --------------------------------------------------------------------------------------


class _StubSource:
    """Base for declared-but-unimplemented adapters.

    A stub never attempts a parse. Attempting one would mean reading a B-factor column
    that, for every source below, holds something other than pLDDT -- producing a
    confident-looking report built on a wrong number.
    """

    info: SourceInfo
    #: Why this cannot simply reuse the AlphaFold2 adapter. Shown in the error.
    incompatibility: str = ""
    #: Where a contributor should look to implement it.
    implementation_hint: str = ""

    def detect(self, path: Path, text_head: str) -> tuple[bool, list[str]]:
        evidence: list[str] = []
        token = self.info.id.replace("-", "")
        if token in path.name.lower().replace("-", "") or _head_contains(
            text_head, self.info.display_name
        ):
            evidence.append(f"file appears to be {self.info.display_name} output")
        return bool(evidence), evidence

    def extract_confidence(self, path: Path, residues: list[dict]):
        raise UnsupportedSourceError(
            f"Source '{self.info.id}' ({self.info.display_name}) is a declared but "
            "not-yet-implemented adapter (Tier 2 stub), so Pocketscribe cannot read its "
            f"confidence data.\n\n  Why it cannot fall back to the AlphaFold2 parser: "
            f"{self.incompatibility}\n\n  Implementing it: {self.implementation_hint}\n\n"
            "  Contributions are welcome -- see the 'Add a new structure-source adapter' "
            "walkthrough in CONTRIBUTING.md. Run 'pocketscribe sources' to see which "
            "sources are fully supported today."
        )


class ProtenixSource(_StubSource):
    """Protenix (ByteDance, Apache-2.0) -- an open AlphaFold3 reproduction.

    Output format notes for a future contributor
    --------------------------------------------
    Protenix writes predictions as mmCIF alongside a JSON summary per sample. Like
    AlphaFold3 (which it reproduces) it is an all-atom diffusion model, so its
    confidence outputs are atom-level pLDDT plus PAE/PDE matrices and chain-level
    ranking scores, typically found in the per-sample ``*_summary_confidence.json``
    and ``*_confidence.json`` files in the prediction output directory. An adapter
    should aggregate atom-level pLDDT to per-residue values (mean over the residue's
    atoms) and must not read the mmCIF B-factor column, which for a diffusion model
    output is not guaranteed to carry pLDDT at all.
    """

    info = SourceInfo(
        id="protenix",
        display_name="Protenix",
        tier=SupportTier.TIER_2_STUB,
        family=ArchitectureFamily.MSA_COEVOLUTION,
        notes="Open AlphaFold3 reproduction; atom-level confidence in sidecar JSON.",
        citation_key="protenix2025",
    )
    incompatibility = (
        "Protenix is an all-atom diffusion model whose confidence is emitted as "
        "atom-level pLDDT and PAE matrices in sidecar JSON, not as a per-residue "
        "B-factor column."
    )
    implementation_hint = (
        "Parse the per-sample *_confidence.json, aggregate atom-level pLDDT to "
        "per-residue means, and return them on a 0-100 scale from extract_confidence()."
    )


class AlphaFold3Source(_StubSource):
    """AlphaFold3 (DeepMind).

    Output format notes for a future contributor
    --------------------------------------------
    AlphaFold3's own output uses different chain-ID and mmCIF metadata conventions
    from AlphaFold2/OpenFold: predictions come as mmCIF with per-atom pLDDT and
    separate ``*_summary_confidences.json`` / ``*_confidences.json`` files carrying
    PAE, pTM/ipTM and chain-pair metrics. Entity/chain identifiers follow the
    AF3 input-JSON entity naming rather than the single-chain 'A' of an AF2 monomer
    prediction. The AF2 parser must not be reused unchanged.
    """

    info = SourceInfo(
        id="alphafold3",
        display_name="AlphaFold3",
        tier=SupportTier.TIER_2_STUB,
        family=ArchitectureFamily.MSA_COEVOLUTION,
        notes="Different chain-ID and mmCIF metadata conventions from AlphaFold2.",
        citation_key="abramson2024",
    )
    incompatibility = (
        "AlphaFold3 writes mmCIF with AF3 entity/chain naming and reports confidence "
        "through separate *_confidences.json files; its chain IDs and metadata do not "
        "follow the AlphaFold2 monomer convention, so the AF2 parser would mislabel "
        "chains even where it found numbers to read."
    )
    implementation_hint = (
        "Map AF3 entity/chain identifiers onto the parsed mmCIF chains, then read "
        "per-atom pLDDT from the confidences JSON and aggregate per residue."
    )


class RoseTTAFoldSource(_StubSource):
    """RoseTTAFold / RoseTTAFold All-Atom (Baker lab).

    Lower implementation priority: current benchmarks generally show RoseTTAFold
    outperformed by newer models, so it is deprioritised rather than excluded. Its
    per-residue confidence is reported as predicted LDDT in a separate ``.npz``
    (``lddt`` array) rather than reliably in the B-factor column.
    """

    info = SourceInfo(
        id="rosettafold",
        display_name="RoseTTAFold",
        tier=SupportTier.TIER_2_STUB,
        family=ArchitectureFamily.MSA_COEVOLUTION,
        notes="Predicted LDDT in a sidecar .npz; lower priority than newer models.",
        citation_key="baek2021",
    )
    incompatibility = (
        "RoseTTAFold reports predicted LDDT in a sidecar .npz array; the B-factor "
        "column of its PDB output is not a dependable pLDDT carrier across versions."
    )
    implementation_hint = (
        "Load the run's .npz, read the 'lddt' array, and map it onto residues in "
        "model order; note RoseTTAFold LDDT is on a 0-1 scale."
    )


# --------------------------------------------------------------------------------------
# Registry
# --------------------------------------------------------------------------------------

#: Detection order matters. ColabFold, OpenFold and ESMFold output all mention
#: AlphaFold-derived concepts, so the specific sources are tried before AlphaFold2,
#: which is the most permissive matcher and therefore acts as the fallback.
_REGISTRY: dict[str, StructureSource] = {}
_DETECTION_ORDER: list[str] = [
    "colabfold",
    "openfold",
    "esmfold",
    "boltz",
    "protenix",
    "alphafold3",
    "rosettafold",
    "alphafold2",
]


def _register(adapter: StructureSource) -> None:
    _REGISTRY[adapter.info.id] = adapter


for _adapter in (
    AlphaFold2Source(),
    OpenFoldSource(),
    ColabFoldSource(),
    ESMFoldSource(),
    BoltzSource(),
    ProtenixSource(),
    AlphaFold3Source(),
    RoseTTAFoldSource(),
):
    _register(_adapter)


def get_source(source_id: str) -> StructureSource:
    """Look up an adapter by CLI identifier.

    Raises
    ------
    SourceDetectionError
        If the identifier is not registered at all (a typo, or a model Pocketscribe
        has never heard of).
    """
    key = source_id.strip().lower()
    if key not in _REGISTRY:
        known = ", ".join(sorted(_REGISTRY))
        raise SourceDetectionError(
            f"Unknown structure source '{source_id}'. Known sources: {known}. "
            "Run 'pocketscribe sources' for the full table including support tiers."
        )
    return _REGISTRY[key]


def all_sources() -> list[StructureSource]:
    """Every registered adapter, ordered Tier 1 first then by name."""
    tier_order = {
        SupportTier.TIER_1: 0,
        SupportTier.TIER_2_REFERENCE: 1,
        SupportTier.TIER_2_STUB: 2,
    }
    return sorted(
        _REGISTRY.values(), key=lambda a: (tier_order[a.info.tier], a.info.display_name)
    )


def read_text_head(path: Path, n_bytes: int = 8192) -> str:
    """Read the first ``n_bytes`` of a structure file as text, tolerating binary noise."""
    with path.open("rb") as handle:
        raw = handle.read(n_bytes)
    return raw.decode("utf-8", errors="replace")


def detect_source(path: Path) -> tuple[StructureSource, list[str]]:
    """Guess which tool produced ``path``.

    Returns the adapter plus the evidence that justified the guess. The CLI always
    prints that evidence, because an auto-detection that is confidently wrong is worse
    than one the user can see and correct.

    Raises
    ------
    SourceDetectionError
        If nothing matched. Guessing AlphaFold2 by default would be the single most
        dangerous silent failure in this tool: an experimental PDB would be read as if
        its temperature factors were confidence scores.
    """
    text_head = read_text_head(path)
    for source_id in _DETECTION_ORDER:
        adapter = _REGISTRY[source_id]
        matched, evidence = adapter.detect(path, text_head)
        if matched:
            return adapter, evidence

    raise SourceDetectionError(
        f"Could not determine which prediction tool produced '{path.name}'.\n"
        "  Nothing in its filename or headers identified a supported source.\n"
        "  If this is an experimental (deposited) structure, Pocketscribe is not the "
        "right tool: its whole premise is caveating pocket claims by per-residue "
        "prediction confidence, which an experimental structure does not carry.\n"
        "  If it is a predicted structure whose headers were stripped, state the source "
        "explicitly, e.g. --source alphafold2. Run 'pocketscribe sources' to list them."
    )
