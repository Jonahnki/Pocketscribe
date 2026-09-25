"""Pipeline orchestration.

Wires the stages together in order: parse -> confidence -> pockets -> (consensus) ->
(MD setup) -> (narrative). The CLI is a thin layer over this module, so the whole
pipeline is callable from a notebook or another tool without going through argument
parsing.

The ordering constraint that matters: the narrative stage runs last, takes only the
finished :class:`~pocketscribe.models.AnalysisResult`, and can be removed entirely
without affecting anything above it.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from . import __version__
from .confidence import annotate_pocket_confidence, build_confidence_profile
from .config import RunConfig
from .consensus import apply_consensus_to_pockets, compute_consensus
from .md_setup import generate_md_setup
from .models import (
    AnalysisResult,
    ConsensusParameters,
    PocketSet,
    StructureModel,
)
from .parsing import ParsedStructure, parse_structure
from .pockets import resolve_backend
from .sources import StructureSource, detect_source, get_source


@dataclass
class StructureInput:
    """One ``--pdb path[:source]`` argument."""

    path: Path
    source_id: str = "auto"
    label: str = ""

    @classmethod
    def parse(cls, raw: str) -> StructureInput:
        """Parse ``path`` or ``path:source``.

        Windows drive letters mean a bare ``:`` split is unsafe, so the source is only
        taken from the part after the last colon when that part is not itself a path
        component.
        """
        candidate = Path(raw)
        if candidate.exists():
            return cls(path=candidate)
        if ":" in raw:
            head, _, tail = raw.rpartition(":")
            if head and "/" not in tail and "\\" not in tail:
                return cls(path=Path(head), source_id=tail.strip().lower())
        return cls(path=candidate)


def load_structure(
    item: StructureInput, label: str
) -> tuple[StructureModel, ParsedStructure, StructureSource]:
    """Parse one structure, establish its source, and extract its confidence profile."""
    parsed = parse_structure(item.path)

    auto_detected = item.source_id in {"", "auto"}
    if auto_detected:
        source, evidence = detect_source(item.path)
    else:
        source = get_source(item.source_id)
        evidence = []

    profile = build_confidence_profile(parsed, source)

    structure = StructureModel(
        path=str(item.path),
        file_format=parsed.file_format,
        source=source.info,
        source_was_auto_detected=auto_detected,
        detection_evidence=evidence,
        sequences=parsed.sequences,
        confidence=profile,
        qc=parsed.qc,
        label=label,
    )
    return structure, parsed, source


def run_pipeline(
    inputs: list[StructureInput],
    config: RunConfig,
    md_output_dir: Path | None = None,
    allow_backend_fallback: bool = False,
    narrative_enabled: bool | None = None,
    progress=None,
) -> tuple[AnalysisResult, list[ParsedStructure]]:
    """Run the full analysis.

    Parameters
    ----------
    allow_backend_fallback:
        Only ``demo`` and the test suite set this. It lets the built-in geometric
        backend stand in when fpocket is missing, which must never happen silently in a
        real analysis run.
    progress:
        Optional callable taking a status string, used by the CLI for live output.
    """

    def say(message: str) -> None:
        if progress is not None:
            progress(message)

    warnings: list[str] = []
    structures: list[StructureModel] = []
    parsed_structures: list[ParsedStructure] = []
    sources: list[StructureSource] = []

    for position, item in enumerate(inputs, start=1):
        label = item.label or item.path.name
        say(f"[{position}/{len(inputs)}] Parsing {label}")
        structure, parsed, source = load_structure(item, label)
        structures.append(structure)
        parsed_structures.append(parsed)
        sources.append(source)
        say(
            f"    source: {source.info.display_name}"
            + (" (auto-detected)" if structure.source_was_auto_detected else "")
            + f", {structure.qc.n_residues} residues, "
            f"mean {structure.confidence.metric.value} "
            f"{structure.confidence.mean:.1f}"
        )

    backend, fell_back = resolve_backend(
        config.pockets.backend,
        allow_fallback=allow_backend_fallback,
        fpocket_binary=config.pockets.fpocket_binary,
        min_alpha_spheres=config.pockets.min_alpha_spheres,
    )
    if fell_back:
        message = (
            "fpocket was not found on PATH, so the built-in geometric fallback backend "
            "was used. Its scores are a documented heuristic, NOT fpocket druggability "
            "scores, and are not suitable for scientific interpretation."
        )
        warnings.append(message)
        say(f"    WARNING: {message}")

    pocket_sets: list[PocketSet] = []
    for structure, parsed in zip(structures, parsed_structures, strict=True):
        say(f"Detecting pockets in {structure.label} ({backend.name})")
        pocket_set = backend.detect(parsed)
        pocket_set.structure_label = structure.label
        for pocket in pocket_set.pockets:
            pocket.confidence = annotate_pocket_confidence(pocket, structure.confidence)
        pocket_sets.append(pocket_set)
        say(f"    {len(pocket_set.pockets)} pocket(s) detected")

    consensus = None
    if len(structures) > 1:
        say("Computing cross-model consensus")
        consensus = compute_consensus(
            structures=structures,
            parsed=parsed_structures,
            pocket_sets=pocket_sets,
            parameters=ConsensusParameters(**config.consensus.model_dump()),
        )
        apply_consensus_to_pockets(consensus, structures, pocket_sets)
        say(
            f"    {len(consensus.groups)} pocket group(s); "
            f"{consensus.n_cross_family_pockets} with cross-family support"
        )

    md_setup = None
    if config.md.enabled:
        target = _select_target_pocket(pocket_sets[0], config.md.target_pocket_rank)
        directory = md_output_dir or Path("md_setup")
        say(f"Generating MD setup in {directory}")
        md_setup = generate_md_setup(
            structure=structures[0],
            parsed=parsed_structures[0],
            pocket_set=pocket_sets[0],
            output_dir=directory,
            target_pocket=target,
            force_field=config.md.force_field,
            water_model=config.md.water_model,
            box_shape=config.md.box_shape,
            box_padding_nm=config.md.box_padding_nm,
            salt_concentration_M=config.md.salt_concentration_M,
            temperature_K=config.md.temperature_K,
            production_ns=config.md.production_ns,
        )
        say(f"    {len(md_setup.files)} file(s) written")

    parameters = config.as_parameters()
    # Record what actually ran, not only what was asked for: if the backend fell back,
    # the methods appendix must say so rather than repeating the request.
    parameters["pocket backend"] = (
        f"{backend.name} (requested: {config.pockets.backend})"
        if backend.name != config.pockets.backend
        else backend.name
    )

    result = AnalysisResult(
        tool_version=__version__,
        generated_at=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        structures=structures,
        pocket_sets=pocket_sets,
        consensus=consensus,
        md_setup=md_setup,
        top_pockets_requested=config.pockets.top_pockets,
        warnings=warnings,
        parameters=parameters,
    )

    # Narrative last, and isolated: everything above is already complete and renderable
    # whether or not this stage does anything.
    from .narrative import synthesize  # imported here to keep the core import-light

    enabled = config.narrative.enabled if narrative_enabled is None else narrative_enabled
    say("Narrative synthesis" if enabled else "Narrative synthesis disabled")
    result.narrative = synthesize(result, model=config.narrative.model, enabled=enabled)
    if not result.narrative.enabled:
        say(f"    skipped: {result.narrative.skipped_reason.splitlines()[0]}")

    return result, parsed_structures


def _select_target_pocket(pocket_set: PocketSet, rank: int | None):
    """Pick the pocket the MD setup targets: the requested rank, or the top-ranked one."""
    if not pocket_set.pockets:
        return None
    if rank is None:
        return pocket_set.pockets[0]
    for pocket in pocket_set.pockets:
        if pocket.rank == rank:
            return pocket
    return pocket_set.pockets[0]
