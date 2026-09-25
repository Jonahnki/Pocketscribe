"""Per-residue confidence extraction and low-confidence region mapping.

This is the module that makes Pocketscribe what it is. Most pocket-detection tools
treat a predicted structure exactly as they would treat a crystal structure. Here the
per-residue confidence of the prediction is carried through the whole pipeline and
attached to every pocket claim, so that a high druggability score sitting in a
disordered loop is never presented as if it were a high druggability score in a
well-resolved binding site.

Thresholds follow the conventional AlphaFold interpretation bands:

========  ============================================================
pLDDT     Interpretation
========  ============================================================
> 90      Very high; backbone and side chains generally reliable.
70 - 90   Confident backbone; side-chain placement less certain.
50 - 70   Low; treat the local structure cautiously.
< 50      Very low; frequently disordered. Not a basis for a structural claim.
========  ============================================================
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .errors import ConfidenceExtractionError
from .models import (
    CaveatLevel,
    ConfidenceProfile,
    LowConfidenceRegion,
    Pocket,
    PocketConfidenceAnnotation,
    ResidueConfidence,
)
from .parsing import ParsedStructure
from .sources import StructureSource

#: Below this, local structure is "low confidence".
THRESHOLD_LOW = 70.0
#: Below this, local structure is "very low confidence" / likely disordered.
THRESHOLD_VERY_LOW = 50.0
#: Shortest run of consecutive sub-threshold residues worth reporting as a region.
MIN_REGION_LENGTH = 3


def build_confidence_profile(
    structure: ParsedStructure, source: StructureSource
) -> ConfidenceProfile:
    """Extract per-residue confidence for ``structure`` using ``source``'s adapter.

    Raises
    ------
    ConfidenceExtractionError
        Propagated from the adapter, or raised here if the adapter returned confidence
        for too few residues to caveat pockets meaningfully.
    """
    metric, values = source.extract_confidence(Path(structure.path), structure.residues)

    residues: list[ResidueConfidence] = []
    for residue in structure.residues:
        key = (residue["chain"], residue["resseq"], residue["icode"])
        if key not in values:
            continue
        residues.append(
            ResidueConfidence(
                chain=residue["chain"],
                resseq=residue["resseq"],
                icode=residue["icode"],
                resname=residue["resname"],
                value=float(values[key]),
            )
        )

    if not residues:
        raise ConfidenceExtractionError(
            f"Adapter for '{source.info.display_name}' returned no per-residue "
            "confidence that could be matched to residues in the structure."
        )

    coverage = len(residues) / max(len(structure.residues), 1)
    if coverage < 0.5:
        raise ConfidenceExtractionError(
            f"Confidence was recovered for only {coverage:.0%} of residues "
            f"({len(residues)}/{len(structure.residues)}) using the "
            f"'{source.info.display_name}' adapter. Pocketscribe will not caveat pockets "
            "from such sparse data. Check that the declared --source matches the tool "
            "that produced this file."
        )

    array = np.array([r.value for r in residues], dtype=float)
    return ConfidenceProfile(
        metric=metric,
        residues=residues,
        mean=float(np.mean(array)),
        median=float(np.median(array)),
        minimum=float(np.min(array)),
        maximum=float(np.max(array)),
        fraction_below_70=float(np.mean(array < THRESHOLD_LOW)),
        fraction_below_50=float(np.mean(array < THRESHOLD_VERY_LOW)),
        regions_below_70=find_low_confidence_regions(residues, THRESHOLD_LOW),
        regions_below_50=find_low_confidence_regions(residues, THRESHOLD_VERY_LOW),
    )


def find_low_confidence_regions(
    residues: list[ResidueConfidence],
    threshold: float,
    min_length: int = MIN_REGION_LENGTH,
) -> list[LowConfidenceRegion]:
    """Map contiguous sub-threshold runs onto the sequence.

    Runs are broken at chain boundaries and at gaps in residue numbering, so a region
    never silently spans a part of the chain that was not modelled.
    """
    regions: list[LowConfidenceRegion] = []
    current: list[ResidueConfidence] = []
    previous: ResidueConfidence | None = None

    def flush() -> None:
        if len(current) >= min_length:
            values = [r.value for r in current]
            regions.append(
                LowConfidenceRegion(
                    chain=current[0].chain,
                    start=current[0].resseq,
                    end=current[-1].resseq,
                    threshold=threshold,
                    length=len(current),
                    mean_value=round(float(np.mean(values)), 1),
                )
            )
        current.clear()

    for residue in sorted(residues, key=lambda r: (r.chain, r.resseq, r.icode)):
        contiguous = (
            previous is not None
            and residue.chain == previous.chain
            and residue.resseq == previous.resseq + 1
        )
        if residue.value < threshold:
            if not contiguous:
                flush()
            current.append(residue)
        else:
            flush()
        previous = residue
    flush()
    return regions


def annotate_pocket_confidence(
    pocket: Pocket, profile: ConfidenceProfile
) -> PocketConfidenceAnnotation:
    """Cross-reference one pocket's lining residues against the confidence map.

    This is pipeline stage 2's final step: every pocket leaves this function carrying
    a caveat, even when that caveat is "none needed".
    """
    values: list[float] = []
    for residue in pocket.residues:
        value = profile.value_for(residue.chain, residue.resseq, residue.icode)
        residue.confidence = value
        if value is not None:
            values.append(value)

    if not values:
        return PocketConfidenceAnnotation(
            level=CaveatLevel.SEVERE,
            n_residues_scored=0,
            text=(
                "No prediction confidence could be matched to the residues lining this "
                "pocket, so its reliability cannot be assessed. Treat it as unverified."
            ),
        )

    array = np.array(values, dtype=float)
    mean_value = float(np.mean(array))
    min_value = float(np.min(array))
    fraction_below_70 = float(np.mean(array < THRESHOLD_LOW))
    fraction_below_50 = float(np.mean(array < THRESHOLD_VERY_LOW))

    overlapping = _overlapping_regions(pocket, profile)
    level = _caveat_level(mean_value, fraction_below_70, fraction_below_50)
    text = _caveat_text(
        level=level,
        mean_value=mean_value,
        min_value=min_value,
        fraction_below_70=fraction_below_70,
        fraction_below_50=fraction_below_50,
        overlapping=overlapping,
        n_residues=len(values),
    )

    return PocketConfidenceAnnotation(
        mean_confidence=round(mean_value, 1),
        min_confidence=round(min_value, 1),
        fraction_below_70=round(fraction_below_70, 3),
        fraction_below_50=round(fraction_below_50, 3),
        n_residues_scored=len(values),
        overlapping_low_confidence_regions=overlapping,
        level=level,
        text=text,
    )


def _overlapping_regions(pocket: Pocket, profile: ConfidenceProfile) -> list[str]:
    """Labels of flagged low-confidence regions that this pocket touches."""
    labels: list[str] = []
    for region in profile.regions_below_50 + profile.regions_below_70:
        for residue in pocket.residues:
            if residue.chain == region.chain and region.start <= residue.resseq <= region.end:
                label = f"{region.label} (mean {region.mean_value}, <{int(region.threshold)})"
                if label not in labels:
                    labels.append(label)
                break
    return labels


def _caveat_level(
    mean_value: float, fraction_below_70: float, fraction_below_50: float
) -> CaveatLevel:
    """Map pocket-level confidence statistics onto a caveat severity.

    Deliberately conservative: a pocket is escalated on *either* a poor average or a
    substantial minority of untrustworthy residues, because a binding site only needs
    a few badly-placed lining residues to be geometrically wrong.
    """
    if fraction_below_50 >= 0.25 or mean_value < THRESHOLD_VERY_LOW:
        return CaveatLevel.SEVERE
    if fraction_below_50 >= 0.10 or mean_value < THRESHOLD_LOW or fraction_below_70 >= 0.40:
        return CaveatLevel.MODERATE
    if fraction_below_70 >= 0.15 or mean_value < 85.0:
        return CaveatLevel.LOW
    return CaveatLevel.NONE


def _caveat_text(
    level: CaveatLevel,
    mean_value: float,
    min_value: float,
    fraction_below_70: float,
    fraction_below_50: float,
    overlapping: list[str],
    n_residues: int,
) -> str:
    """Compose the sentence that appears next to the pocket in the report."""
    stats = (
        f"Mean pLDDT of the {n_residues} lining residues is {mean_value:.1f} "
        f"(lowest {min_value:.1f}; {fraction_below_70:.0%} below 70, "
        f"{fraction_below_50:.0%} below 50)."
    )

    if level is CaveatLevel.SEVERE:
        judgement = (
            "This pocket sits substantially in poorly-predicted structure. Its geometry, "
            "and therefore its druggability score, should not be relied upon. Do not "
            "commit docking or simulation resources to it without independent evidence "
            "such as an experimental structure or a homologous binding site."
        )
    elif level is CaveatLevel.MODERATE:
        judgement = (
            "A meaningful part of this pocket is predicted with low confidence. The "
            "cavity may be real but its precise shape and volume are uncertain; treat "
            "the druggability score as indicative rather than quantitative."
        )
    elif level is CaveatLevel.LOW:
        judgement = (
            "Most of this pocket is confidently predicted, with some less certain "
            "residues at its periphery. Side-chain placement in those residues may shift "
            "on refinement or simulation."
        )
    else:
        judgement = (
            "The residues lining this pocket are confidently predicted. The usual caveat "
            "still applies: this is a computational model, not an experimental structure, "
            "and side-chain rotamers in apo predictions often differ from the "
            "ligand-bound state."
        )

    if overlapping:
        judgement += " Overlaps flagged low-confidence region(s): " + "; ".join(overlapping) + "."

    return f"{stats} {judgement}"


def confidence_band_counts(profile: ConfidenceProfile) -> dict[str, int]:
    """Residue counts in the four conventional pLDDT interpretation bands."""
    values = np.array([r.value for r in profile.residues], dtype=float)
    return {
        "very_high": int(np.sum(values >= 90)),
        "confident": int(np.sum((values >= 70) & (values < 90))),
        "low": int(np.sum((values >= 50) & (values < 70))),
        "very_low": int(np.sum(values < 50)),
    }
