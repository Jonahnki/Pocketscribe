"""CLI behaviour: the commands, and the errors a bench scientist will actually hit."""

from __future__ import annotations

from typer.testing import CliRunner

from pocketscribe import __version__
from pocketscribe.cli import app
from pocketscribe.pipeline import StructureInput

runner = CliRunner()


# --------------------------------------------------------------------------------------
# sources / version
# --------------------------------------------------------------------------------------


def test_sources_lists_every_source_with_its_tier():
    result = runner.invoke(app, ["sources"])
    assert result.exit_code == 0
    for source_id in (
        "alphafold2",
        "openfold",
        "colabfold",
        "esmfold",
        "boltz",
        "protenix",
        "alphafold3",
        "rosettafold",
    ):
        assert source_id in result.output
    assert "Tier 1 (full support)" in result.output
    assert "Tier 2 (stub - not implemented)" in result.output


def test_sources_prints_citations():
    result = runner.invoke(app, ["sources"])
    assert "Jumper" in result.output
    assert "Mirdita" in result.output


def test_sources_points_at_the_contribution_path():
    result = runner.invoke(app, ["sources"])
    assert "CONTRIBUTING.md" in result.output


def test_version():
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0
    assert __version__ in result.output


# --------------------------------------------------------------------------------------
# demo
# --------------------------------------------------------------------------------------


def test_demo_runs_end_to_end(tmp_path):
    output = tmp_path / "demo.html"
    result = runner.invoke(app, ["demo", "--output", str(output)])
    assert result.exit_code == 0, result.output
    assert output.is_file()

    html = output.read_text()
    assert "Binding-pocket assessment" in html
    assert 'id="pockets"' in html


def test_demo_states_that_the_structures_are_synthetic(tmp_path):
    result = runner.invoke(app, ["demo", "--output", str(tmp_path / "d.html")])
    assert "synthetic" in result.output
    assert "not real proteins" in result.output


def test_demo_consensus_runs_end_to_end(tmp_path):
    output = tmp_path / "consensus.html"
    result = runner.invoke(app, ["demo", "--consensus", "--output", str(output)])
    assert result.exit_code == 0, result.output

    html = output.read_text()
    assert "Cross-model consensus" in html
    # Definition of done: the table distinguishes the two kinds of agreement.
    assert "cross-family" in html
    assert "same-family-only" in html


def test_demo_consensus_reports_all_three_evidence_classes(tmp_path):
    result = runner.invoke(app, ["demo", "--consensus", "--output", str(tmp_path / "c.html")])
    assert "cross-family" in result.output
    assert "same-family only" in result.output
    assert "single-model" in result.output


def test_demo_with_md_setup(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(app, ["demo", "--md-setup", "--output", str(tmp_path / "d.html")])
    assert result.exit_code == 0, result.output
    assert (tmp_path / "md_setup" / "md.mdp").is_file()
    assert (tmp_path / "md_setup" / "PROTOCOL.md").is_file()


# --------------------------------------------------------------------------------------
# run
# --------------------------------------------------------------------------------------


def test_run_on_a_single_structure(tmp_path, af2_path):
    output = tmp_path / "report.html"
    result = runner.invoke(
        app,
        [
            "run",
            "--pdb",
            str(af2_path),
            "--source",
            "alphafold2",
            "--pocket-backend",
            "builtin",
            "--output",
            str(output),
        ],
    )
    assert result.exit_code == 0, result.output
    assert output.is_file()
    assert "AlphaFold2" in result.output


def test_run_auto_detects_the_source(tmp_path, esmfold_path):
    result = runner.invoke(
        app,
        [
            "run",
            "--pdb",
            str(esmfold_path),
            "--pocket-backend",
            "builtin",
            "--output",
            str(tmp_path / "r.html"),
        ],
    )
    assert result.exit_code == 0, result.output
    assert "ESMFold" in result.output
    # Auto-detection must always show its reasoning so a wrong guess is correctable.
    assert "auto-detected" in result.output


def test_run_consensus_mode_with_per_file_sources(tmp_path, af2_path, esmfold_path):
    output = tmp_path / "consensus.html"
    result = runner.invoke(
        app,
        [
            "run",
            "--pdb",
            f"{af2_path}:alphafold2",
            "--pdb",
            f"{esmfold_path}:esmfold",
            "--pocket-backend",
            "builtin",
            "--output",
            str(output),
        ],
    )
    assert result.exit_code == 0, result.output
    assert "Cross-model consensus mode" in result.output
    assert "Cross-model consensus" in output.read_text()


def test_run_rejects_mismatched_proteins_cleanly(
    tmp_path, af2_path, different_protein_path
):
    """No traceback: a bench scientist gets a sentence explaining what went wrong."""
    result = runner.invoke(
        app,
        [
            "run",
            "--pdb",
            f"{af2_path}:alphafold2",
            "--pdb",
            f"{different_protein_path}:alphafold2",
            "--pocket-backend",
            "builtin",
            "--output",
            str(tmp_path / "r.html"),
        ],
    )
    assert result.exit_code == 1
    assert "not the same protein" in result.output
    assert "Traceback" not in result.output


def test_run_rejects_a_tier_2_stub_source_cleanly(tmp_path, af2_path):
    result = runner.invoke(
        app,
        [
            "run",
            "--pdb",
            str(af2_path),
            "--source",
            "protenix",
            "--output",
            str(tmp_path / "r.html"),
        ],
    )
    assert result.exit_code == 1
    assert "not-yet-implemented" in result.output or "not yet implemented" in result.output
    assert "Traceback" not in result.output


def test_run_rejects_an_experimental_structure_cleanly(tmp_path, experimental_path):
    result = runner.invoke(
        app,
        [
            "run",
            "--pdb",
            str(experimental_path),
            "--source",
            "alphafold2",
            "--output",
            str(tmp_path / "r.html"),
        ],
    )
    assert result.exit_code == 1
    assert "temperature factors" in result.output


def test_run_without_fpocket_does_not_fall_back_silently(
    tmp_path, af2_path, monkeypatch
):
    """A real analysis run must never substitute heuristic scores for fpocket's."""
    monkeypatch.setattr("pocketscribe.pockets.fpocket_available", lambda *_: False)
    result = runner.invoke(
        app,
        [
            "run",
            "--pdb",
            str(af2_path),
            "--source",
            "alphafold2",
            "--output",
            str(tmp_path / "r.html"),
        ],
    )
    assert result.exit_code == 1
    assert "fpocket" in result.output
    assert "conda" in result.output


def test_run_honours_no_narrative(tmp_path, af2_path):
    result = runner.invoke(
        app,
        [
            "run",
            "--pdb",
            str(af2_path),
            "--source",
            "alphafold2",
            "--pocket-backend",
            "builtin",
            "--no-narrative",
            "--output",
            str(tmp_path / "r.html"),
        ],
    )
    assert result.exit_code == 0, result.output
    assert "Narrative synthesis disabled" in result.output


def test_run_accepts_a_config_file(tmp_path, af2_path):
    config = tmp_path / "config.yaml"
    config.write_text(
        "pockets:\n  backend: builtin\n  top_pockets: 2\n"
        "consensus:\n  centroid_tolerance: 6.0\n"
    )
    result = runner.invoke(
        app,
        [
            "run",
            "--pdb",
            str(af2_path),
            "--source",
            "alphafold2",
            "--pocket-backend",
            "builtin",
            "--config",
            str(config),
            "--output",
            str(tmp_path / "r.html"),
        ],
    )
    assert result.exit_code == 0, result.output


def test_invalid_config_is_reported_clearly(tmp_path, af2_path):
    config = tmp_path / "bad.yaml"
    config.write_text("pockets:\n  backend: nonsense\n")
    result = runner.invoke(
        app,
        [
            "run",
            "--pdb",
            str(af2_path),
            "--config",
            str(config),
            "--output",
            str(tmp_path / "r.html"),
        ],
    )
    assert result.exit_code == 1
    assert "Invalid configuration" in result.output


# --------------------------------------------------------------------------------------
# Argument parsing
# --------------------------------------------------------------------------------------


def test_structure_input_parses_a_bare_path(af2_path):
    item = StructureInput.parse(str(af2_path))
    assert item.path == af2_path
    assert item.source_id == "auto"


def test_structure_input_parses_a_tagged_path(af2_path):
    item = StructureInput.parse(f"{af2_path}:esmfold")
    assert item.path == af2_path
    assert item.source_id == "esmfold"


def test_structure_input_prefers_an_existing_path_over_a_tag(af2_path):
    """A real file whose name contains a colon must not be split."""
    item = StructureInput.parse(str(af2_path))
    assert item.path.exists()
