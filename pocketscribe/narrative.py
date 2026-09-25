"""Optional narrative synthesis (pipeline stage 5).

This is the only module in Pocketscribe that talks to an external API, and it is
deliberately isolated:

* Nothing in the deterministic pipeline imports it except the CLI, which calls it last.
* It receives a JSON summary built from :mod:`pocketscribe.models` -- never coordinates,
  never a file handle, never the structure itself.
* If ``ANTHROPIC_API_KEY`` is unset, or the ``anthropic`` package is not installed, or
  the call fails for any reason, it returns a skipped result and the report renders
  complete without this section.

The report is a scientific artefact whose whole premise is that predicted structures
carry uncertainty. A narrative that overclaims would contradict the rest of the
document, so the prompt below instructs the model to reinforce the caveats rather than
smooth them over, and the rendered section is labelled as AI-generated.
"""

from __future__ import annotations

import json
import os
from typing import Any

from .models import AnalysisResult, NarrativeResult

#: Kept deliberately small: this is a few paragraphs of prose, not a document.
MAX_TOKENS = 1600

DEFAULT_MODEL = "claude-sonnet-4-5"

SYSTEM_PROMPT = """\
You are writing the interpretation section of a structural bioinformatics report for an \
academic drug-discovery group. The report analyses a COMPUTATIONALLY PREDICTED protein \
structure, not an experimental one.

Write 2-4 paragraphs of plain scientific prose. No headings, no bullet lists, no \
markdown formatting, no preamble such as "Here is the interpretation".

What to cover:
- Which pocket or pockets are most worth pursuing, and the specific reason why, citing \
  the numbers you were given.
- What the per-residue confidence data means for interpreting those pockets. A high \
  druggability score in a low-confidence region is not a promising result; say so \
  plainly when the data shows it.
- When cross-model consensus data is present: treat agreement between different \
  architecture families as substantially stronger evidence than agreement within one \
  family, and say which kind you are looking at. Never present same-family agreement as \
  independent confirmation.
- One concrete recommended next step, scaled to what the data actually supports.

Hard constraints:
- Do not overclaim. This is first-pass computational triage, not evidence of binding, \
  druggability in any experimental sense, or therapeutic relevance.
- Never state or imply that a pocket is a confirmed binding site, that a compound would \
  bind, or anything about clinical or therapeutic potential.
- Do not invent numbers, residue names, or findings that are not in the data you were \
  given. If something is absent, say it is absent.
- Distinguish clearly between what the geometry shows and what the prediction \
  confidence permits you to conclude from it.
- If the druggability scores are marked as coming from a heuristic fallback backend \
  rather than fpocket, say so and treat them as ordering hints only, not as validated \
  druggability scores.
- Prefer being useful and specific over being comprehensive. A short, honest paragraph \
  beats a long hedge.
"""


def build_payload(result: AnalysisResult) -> dict[str, Any]:
    """Assemble the JSON summary handed to the model.

    Only derived, structured facts cross this boundary. Coordinates never do -- they are
    not needed for interpretation, they would dominate the context, and keeping them out
    makes the privacy story simple: what leaves the machine is a table of numbers about
    pockets, not the unpublished structure itself.
    """
    payload: dict[str, Any] = {
        "run_type": "cross-model consensus" if result.is_consensus_run else "single structure",
        "structures": [],
        "pocket_backend": {
            "name": result.primary_pockets.backend,
            "is_fpocket": result.primary_pockets.backend_is_fpocket,
            "version": result.primary_pockets.backend_version,
            "warning": (
                None
                if result.primary_pockets.backend_is_fpocket
                else (
                    "Scores come from Pocketscribe's built-in geometric heuristic, NOT "
                    "fpocket's validated druggability model."
                )
            ),
        },
    }

    for structure, pocket_set in zip(
        result.structures, result.pocket_sets, strict=True
    ):
        payload["structures"].append(
            {
                "label": structure.label,
                "source": structure.source.display_name,
                "source_uses_msa": structure.source.uses_msa,
                "architecture_family": structure.source.family.value,
                "n_residues": structure.qc.n_residues,
                "n_chains": structure.qc.n_chains,
                "confidence": {
                    "metric": structure.confidence.metric.value,
                    "mean": round(structure.confidence.mean, 1),
                    "median": round(structure.confidence.median, 1),
                    "fraction_below_70": round(structure.confidence.fraction_below_70, 3),
                    "fraction_below_50": round(structure.confidence.fraction_below_50, 3),
                    "n_low_confidence_regions": len(structure.confidence.regions_below_70),
                },
                "qc_notes": structure.qc.notes,
                "pockets": [
                    {
                        "rank": pocket.rank,
                        "id": pocket.id,
                        "volume_A3": pocket.volume,
                        "druggability_score": pocket.druggability_score,
                        "score_provenance": pocket.score_provenance.value,
                        "hydrophobicity_score": pocket.hydrophobicity_score,
                        "polarity_score": pocket.polarity_score,
                        "n_lining_residues": len(pocket.residues),
                        "lining_residues": [r.label for r in pocket.residues[:40]],
                        "confidence": {
                            "mean_plddt_of_lining_residues": pocket.confidence.mean_confidence,
                            "min_plddt": pocket.confidence.min_confidence,
                            "fraction_below_70": pocket.confidence.fraction_below_70,
                            "fraction_below_50": pocket.confidence.fraction_below_50,
                            "caveat_level": pocket.confidence.level.value,
                            "caveat": pocket.confidence.text,
                        },
                    }
                    for pocket in pocket_set.top(result.top_pockets_requested)
                ],
            }
        )

    if result.consensus is not None:
        payload["consensus"] = {
            "cross_family_comparison_possible": result.consensus.cross_family_comparison_possible,
            "families_present": [f.value for f in result.consensus.families_present],
            "n_cross_family_pockets": result.consensus.n_cross_family_pockets,
            "n_same_family_only_pockets": result.consensus.n_same_family_only_pockets,
            "n_single_model_pockets": result.consensus.n_single_model_pockets,
            "alignments": [
                {
                    "pair": f"{a.label_a} vs {a.label_b}",
                    "sequence_identity": a.identity,
                    "ca_rmsd_A": a.rmsd,
                }
                for a in result.consensus.alignments
            ],
            "pocket_groups": [
                {
                    "group_id": group.group_id,
                    "evidence": group.evidence_label,
                    "n_models_detecting": group.n_models_detecting,
                    "n_models_total": group.n_models_total,
                    "cross_family_agreement": group.cross_family_agreement,
                    "same_family_agreement": group.same_family_agreement,
                    "detected_by": [
                        {
                            "source": m.source_display_name,
                            "family": m.family.value,
                            "pocket_rank": m.pocket_rank,
                            "druggability_score": m.druggability_score,
                        }
                        for m in group.members
                    ],
                }
                for group in result.consensus.groups[:10]
            ],
            "notes": result.consensus.notes,
        }

    if result.md_setup is not None:
        payload["md_setup"] = {
            "force_field": result.md_setup.force_field,
            "water_model": result.md_setup.water_model,
            "target_pocket_rank": result.md_setup.target_pocket_rank,
            "n_files": len(result.md_setup.files),
        }

    return payload


def synthesize(
    result: AnalysisResult,
    model: str = DEFAULT_MODEL,
    api_key: str | None = None,
    enabled: bool = True,
) -> NarrativeResult:
    """Generate the narrative section, or explain cleanly why it was skipped.

    This function never raises. Narrative synthesis is optional by design, and a failure
    here must not cost the user their report.
    """
    if not enabled:
        return NarrativeResult(
            enabled=False, skipped_reason="Narrative synthesis was disabled for this run."
        )

    key = api_key or os.environ.get("ANTHROPIC_API_KEY", "").strip()
    if not key:
        return NarrativeResult(
            enabled=False,
            skipped_reason=(
                "No ANTHROPIC_API_KEY in the environment, so the optional narrative "
                "section was skipped. Everything else in this report is produced by the "
                "deterministic, offline pipeline."
            ),
        )

    try:
        import anthropic
    except ImportError:
        return NarrativeResult(
            enabled=False,
            skipped_reason=(
                "The optional 'anthropic' package is not installed, so narrative "
                "synthesis was skipped. Install it with: pip install 'pocketscribe[narrative]'"
            ),
        )

    payload = build_payload(result)
    payload_json = json.dumps(payload, indent=2, sort_keys=True)

    try:
        client = anthropic.Anthropic(api_key=key)
        response = client.messages.create(
            model=model,
            max_tokens=MAX_TOKENS,
            system=SYSTEM_PROMPT,
            messages=[
                {
                    "role": "user",
                    "content": (
                        "Here is the structured output of the analysis. Write the "
                        "interpretation section.\n\n```json\n" + payload_json + "\n```"
                    ),
                }
            ],
        )
        text = "".join(
            block.text for block in response.content if getattr(block, "type", "") == "text"
        ).strip()
    except Exception as exc:  # noqa: BLE001 - the report must survive any API failure
        return NarrativeResult(
            enabled=False,
            skipped_reason=(
                f"Narrative synthesis failed ({type(exc).__name__}: {exc}). The rest of "
                "this report is unaffected: it is produced entirely offline."
            ),
        )

    if not text:
        return NarrativeResult(
            enabled=False,
            skipped_reason="The narrative model returned no text; the section was skipped.",
        )

    return NarrativeResult(
        enabled=True,
        text=text,
        model=model,
        prompt_token_estimate=len(payload_json) // 4,
    )
