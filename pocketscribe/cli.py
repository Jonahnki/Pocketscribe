"""Command-line interface.

Four commands:

``run``
    Analyse one structure, or several structures of the same protein for cross-model
    consensus.
``demo``
    Run end to end on bundled example structures, with no network access and no
    external downloads.
``sources``
    List every structure-prediction source and its support tier.
``version``
    Print the version.

Errors are printed as plain, specific messages rather than tracebacks: the target user
is a bench scientist, and a Python traceback is not an error message.
"""

from __future__ import annotations

import sys
from pathlib import Path

import typer

from . import __version__
from .citations import CITATIONS
from .config import RunConfig
from .errors import PocketscribeError
from .models import SupportTier
from .pipeline import StructureInput, run_pipeline
from .report.render import render_report
from .sources import all_sources

app = typer.Typer(
    name="pocketscribe",
    help=(
        "Confidence-aware binding-pocket triage and MD setup for predicted protein "
        "structures. Research use only."
    ),
    add_completion=False,
    no_args_is_help=True,
)

_TIER_TEXT = {
    SupportTier.TIER_1: "Tier 1 (full support)",
    SupportTier.TIER_2_REFERENCE: "Tier 2 (reference adapter)",
    SupportTier.TIER_2_STUB: "Tier 2 (stub - not implemented)",
}


def _echo(message: str) -> None:
    typer.echo(message)


def _fail(error: Exception) -> None:
    """Print a clean error and exit non-zero."""
    typer.secho(f"\nError: {error}", fg=typer.colors.RED, err=True)
    raise typer.Exit(code=1)


@app.command()
def run(
    pdb: list[str] = typer.Option(
        ...,
        "--pdb",
        help=(
            "Structure file. Optionally tag it with its source as PATH:SOURCE, e.g. "
            "model.pdb:esmfold. Pass --pdb more than once, with 2+ structures of the "
            "SAME protein, to enable cross-model consensus."
        ),
    ),
    source: str = typer.Option(
        "auto",
        "--source",
        help=(
            "Structure source when it is not tagged per file: alphafold2, openfold, "
            "colabfold, esmfold, boltz, or auto to detect it from the file."
        ),
    ),
    output: Path = typer.Option(Path("report.html"), "--output", "-o", help="Report path."),
    top_pockets: int = typer.Option(3, "--top-pockets", min=1, max=50),
    md_setup: bool = typer.Option(
        False, "--md-setup", help="Generate GROMACS input files for the top pocket."
    ),
    md_dir: Path = typer.Option(
        Path("md_setup"), "--md-dir", help="Directory for the generated MD files."
    ),
    pocket_backend: str = typer.Option(
        "fpocket",
        "--pocket-backend",
        help=(
            "'fpocket' (default, recommended) or 'builtin' (offline geometric fallback "
            "whose scores are a heuristic, not fpocket druggability scores)."
        ),
    ),
    min_identity: float = typer.Option(
        0.95,
        "--min-identity",
        min=0.0,
        max=1.0,
        help="Sequence identity required between structures in consensus mode.",
    ),
    no_narrative: bool = typer.Option(
        False, "--no-narrative", help="Skip the optional AI narrative section entirely."
    ),
    config_file: Path | None = typer.Option(
        None, "--config", help="YAML configuration file; CLI options override it."
    ),
) -> None:
    """Analyse a predicted structure and write an HTML report."""
    try:
        config = RunConfig.from_yaml(config_file) if config_file else RunConfig()

        config.pockets.backend = pocket_backend
        config.pockets.top_pockets = top_pockets
        config.consensus.min_sequence_identity = min_identity
        config.md.enabled = md_setup
        if no_narrative:
            config.narrative.enabled = False

        inputs: list[StructureInput] = []
        for raw in pdb:
            item = StructureInput.parse(raw)
            if item.source_id == "auto" and source != "auto":
                item.source_id = source
            item.label = item.path.name
            inputs.append(item)

        if len(inputs) > 1:
            _echo(
                f"Cross-model consensus mode: {len(inputs)} structures. They must be "
                "independent predictions of the same protein."
            )

        result, parsed = run_pipeline(
            inputs=inputs,
            config=config,
            md_output_dir=md_dir,
            allow_backend_fallback=False,
            progress=_echo,
        )

        _echo(f"Rendering report to {output}")
        path = render_report(result, parsed, output, max_images=min(top_pockets, 3))

        _echo("")
        _summarise(result)
        typer.secho(f"Report written to {path}", fg=typer.colors.GREEN)

    except PocketscribeError as exc:
        _fail(exc)
    except KeyboardInterrupt:
        typer.secho("\nInterrupted.", fg=typer.colors.YELLOW, err=True)
        raise typer.Exit(code=130) from None


@app.command()
def demo(
    output: Path = typer.Option(Path("demo_report.html"), "--output", "-o"),
    consensus: bool = typer.Option(
        False,
        "--consensus",
        help="Run the cross-model consensus path on two bundled example models.",
    ),
    md_setup: bool = typer.Option(False, "--md-setup", help="Also generate MD inputs."),
) -> None:
    """Run end to end on bundled example structures. No network access needed."""
    from .data import demo_consensus_structures, demo_structure

    try:
        if consensus:
            paths = demo_consensus_structures()
            inputs = [
                StructureInput(path=path, source_id=source_id, label=path.name)
                for path, source_id in paths
            ]
            _echo(
                "Demo (consensus): "
                + ", ".join(f"{path.name} [{source_id}]" for path, source_id in paths)
            )
        else:
            path, source_id = demo_structure()
            inputs = [StructureInput(path=path, source_id=source_id, label=path.name)]
            _echo(f"Demo: {path.name} [{source_id}]")

        _echo(
            "These are synthetic structures with known geometry, bundled so the demo "
            "runs offline. They are not real proteins."
        )

        config = RunConfig()
        config.md.enabled = md_setup
        # The demo is allowed to fall back to the built-in backend so it runs on a
        # machine without fpocket. A real analysis run never does this silently.
        result, parsed = run_pipeline(
            inputs=inputs,
            config=config,
            allow_backend_fallback=True,
            progress=_echo,
        )
        path = render_report(result, parsed, output, max_images=3)

        _echo("")
        _summarise(result)
        typer.secho(f"Demo report written to {path}", fg=typer.colors.GREEN)

    except PocketscribeError as exc:
        _fail(exc)


@app.command()
def sources() -> None:
    """List supported structure-prediction sources and their support tiers."""
    _echo("")
    _echo("Pocketscribe structure sources")
    _echo("=" * 78)

    for adapter in all_sources():
        info = adapter.info
        _echo("")
        _echo(f"  {info.id}")
        _echo(f"    name          {info.display_name}")
        _echo(f"    support       {_TIER_TEXT[info.tier]}")
        _echo(f"    family        {info.family.value}")
        _echo(f"    confidence    {info.metric.value}")
        _echo(f"    uses MSA      {'yes' if info.uses_msa else 'no (single sequence)'}")
        citation = CITATIONS.get(info.citation_key)
        if citation:
            _echo(f"    cite          {citation.reference}")
        if info.notes:
            for line in _wrap(info.notes, 70):
                _echo(f"    {line}")

    _echo("")
    _echo("-" * 78)
    _echo(
        "Tier 2 stubs fail with a specific 'not yet implemented' message rather than\n"
        "attempting a parse that would be wrong. Adding one is the most useful\n"
        "contribution you can make: see CONTRIBUTING.md."
    )
    _echo("")


@app.command()
def version() -> None:
    """Print the Pocketscribe version."""
    _echo(f"pocketscribe {__version__}")


def _summarise(result) -> None:
    """Terminal summary of what the report contains."""
    pockets = result.primary_pockets
    _echo("Summary")
    _echo("-" * 60)
    _echo(f"  structure        {result.primary_structure.label}")
    _echo(f"  source           {result.primary_structure.source.display_name}")
    _echo(
        f"  mean pLDDT       {result.primary_structure.confidence.mean:.1f} "
        f"({result.primary_structure.confidence.fraction_below_70:.0%} below 70)"
    )
    _echo(f"  pockets          {len(pockets.pockets)} (backend: {pockets.backend})")

    for pocket in pockets.top(result.top_pockets_requested):
        score = (
            f"{pocket.druggability_score:.3f}"
            if pocket.druggability_score is not None
            else "n/a"
        )
        mean_confidence = (
            f"{pocket.confidence.mean_confidence:.0f}"
            if pocket.confidence.mean_confidence is not None
            else "n/a"
        )
        volume = f"{pocket.volume:.0f} A^3" if pocket.volume is not None else "n/a"
        _echo(f"    #{pocket.rank}  score {score}  volume {volume}")
        _echo(
            f"         mean pLDDT {mean_confidence}  "
            f"caveat: {pocket.confidence.level.value}"
        )

    if result.consensus is not None:
        _echo(
            f"  consensus        {len(result.consensus.groups)} group(s); "
            f"{result.consensus.n_cross_family_pockets} cross-family, "
            f"{result.consensus.n_same_family_only_pockets} same-family only, "
            f"{result.consensus.n_single_model_pockets} single-model"
        )
    if result.md_setup is not None:
        _echo(f"  MD setup         {len(result.md_setup.files)} files in {result.md_setup.output_dir}")
    _echo(
        "  narrative        "
        + ("included" if result.narrative.enabled else "skipped (optional)")
    )
    _echo("")


def _wrap(text: str, width: int) -> list[str]:
    import textwrap

    return textwrap.wrap(text, width=width)


def main() -> None:
    """Entry point used by the console script."""
    try:
        app()
    except PocketscribeError as exc:  # safety net; commands handle their own errors
        typer.secho(f"Error: {exc}", fg=typer.colors.RED, err=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
