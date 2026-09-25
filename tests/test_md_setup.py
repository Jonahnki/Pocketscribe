"""MD setup generation.

GROMACS is not installed in CI, so these tests check the generated files structurally:
valid `.mdp` syntax, no unknown or duplicated parameters, the CHARMM36-specific
non-bonded settings actually present, and a cleaned structure that `pdb2gmx` could
read. A domain expert should be able to hand these to GROMACS and change only
system-specific values.
"""

from __future__ import annotations

import re
import stat

import pytest

from pocketscribe.confidence import annotate_pocket_confidence
from pocketscribe.md_setup import generate_md_setup
from pocketscribe.pipeline import StructureInput, load_structure
from pocketscribe.pockets import BuiltinGeometricBackend

#: Every parameter name the generated .mdp files are allowed to use. A typo in an .mdp
#: key is silently ignored by some GROMACS versions and fatal in others, so the set is
#: pinned rather than pattern-matched.
KNOWN_MDP_KEYS = {
    "integrator", "dt", "nsteps", "emtol", "emstep", "define", "continuation",
    "nstxout-compressed", "compressed-x-grps", "nstenergy", "nstlog", "nstlist",
    "cutoff-scheme", "pbc", "verlet-buffer-tolerance", "coulombtype", "rcoulomb",
    "pme-order", "fourierspacing", "vdwtype", "vdw-modifier", "rvdw-switch", "rvdw",
    "DispCorr", "constraints", "constraint-algorithm", "lincs-iter", "lincs-order",
    "tcoupl", "tc-grps", "tau-t", "ref-t", "pcoupl", "pcoupltype", "tau-p", "ref-p",
    "compressibility", "refcoord-scaling", "gen-vel", "gen-temp", "gen-seed",
    "comm-mode", "nstcomm",
}


@pytest.fixture
def md_output(tmp_path, af2_path):
    structure, parsed, _ = load_structure(
        StructureInput(path=af2_path, source_id="alphafold2"), label=af2_path.name
    )
    pocket_set = BuiltinGeometricBackend().detect(parsed)
    for pocket in pocket_set.pockets:
        pocket.confidence = annotate_pocket_confidence(pocket, structure.confidence)

    target = pocket_set.pockets[0]
    result = generate_md_setup(
        structure=structure,
        parsed=parsed,
        pocket_set=pocket_set,
        output_dir=tmp_path / "md",
        target_pocket=target,
    )
    return result, tmp_path / "md", target


def _parse_mdp(text: str) -> dict[str, str]:
    """Parse an .mdp file the way GROMACS does, and fail on malformed lines."""
    parameters: dict[str, str] = {}
    for raw_line in text.splitlines():
        line = raw_line.split(";", 1)[0].strip()
        if not line:
            continue
        assert "=" in line, f"malformed .mdp line: {raw_line!r}"
        key, _, value = line.partition("=")
        key = key.strip()
        assert key not in parameters, f"duplicate .mdp parameter: {key}"
        parameters[key] = value.strip()
    return parameters


# --------------------------------------------------------------------------------------
# Files
# --------------------------------------------------------------------------------------


def test_generates_the_expected_file_set(md_output):
    result, directory, _target = md_output
    names = {file.name for file in result.files}
    assert {
        "protein_clean.pdb",
        "minim.mdp",
        "ions.mdp",
        "nvt.mdp",
        "npt.mdp",
        "md.mdp",
        "run_setup.sh",
        "PROTOCOL.md",
        "pocket_selection.txt",
    } <= names
    for file in result.files:
        assert (directory / file.name).is_file()


def test_setup_script_is_executable(md_output):
    _result, directory, _target = md_output
    mode = (directory / "run_setup.sh").stat().st_mode
    assert mode & stat.S_IXUSR


# --------------------------------------------------------------------------------------
# .mdp validity
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("name", ["minim.mdp", "ions.mdp", "nvt.mdp", "npt.mdp", "md.mdp"])
def test_mdp_files_parse_and_use_known_keys(md_output, name):
    _result, directory, _target = md_output
    parameters = _parse_mdp((directory / name).read_text())
    assert parameters, f"{name} defined no parameters"
    unknown = set(parameters) - KNOWN_MDP_KEYS
    assert not unknown, f"{name} uses unknown .mdp keys: {sorted(unknown)}"


@pytest.mark.parametrize("name", ["minim.mdp", "ions.mdp", "nvt.mdp", "npt.mdp", "md.mdp"])
def test_every_mdp_declares_an_integrator(md_output, name):
    _result, directory, _target = md_output
    parameters = _parse_mdp((directory / name).read_text())
    assert parameters["integrator"] in {"steep", "md", "cg", "md-vv"}


@pytest.mark.parametrize("name", ["minim.mdp", "ions.mdp", "nvt.mdp", "npt.mdp", "md.mdp"])
def test_charmm36_nonbonded_settings_are_explicit(md_output, name):
    """Running CHARMM36m on GROMACS defaults is a common, consequential error."""
    _result, directory, _target = md_output
    parameters = _parse_mdp((directory / name).read_text())
    assert parameters["cutoff-scheme"] == "Verlet"
    assert parameters["vdw-modifier"] == "force-switch"
    assert parameters["rvdw-switch"] == "1.0"
    assert parameters["rvdw"] == "1.2"
    assert parameters["rcoulomb"] == "1.2"
    assert parameters["coulombtype"] == "PME"


def test_dynamics_files_constrain_hbonds_for_a_2fs_timestep(md_output):
    """A 2 fs step without h-bond constraints is unstable; the two must travel together."""
    _result, directory, _target = md_output
    for name in ("nvt.mdp", "npt.mdp", "md.mdp"):
        parameters = _parse_mdp((directory / name).read_text())
        assert parameters["dt"] == "0.002"
        assert parameters["constraints"] == "h-bonds"


def test_equilibration_restrains_the_protein_and_production_does_not(md_output):
    _result, directory, _target = md_output
    assert "-DPOSRES" in _parse_mdp((directory / "nvt.mdp").read_text())["define"]
    assert "-DPOSRES" in _parse_mdp((directory / "npt.mdp").read_text())["define"]
    assert "define" not in _parse_mdp((directory / "md.mdp").read_text())


def test_equilibration_order_is_nvt_then_npt(md_output):
    _result, directory, _target = md_output
    nvt = _parse_mdp((directory / "nvt.mdp").read_text())
    npt = _parse_mdp((directory / "npt.mdp").read_text())

    assert nvt["pcoupl"] == "no"  # constant volume first
    assert nvt["gen-vel"] == "yes"  # velocities generated here
    assert nvt["continuation"] == "no"

    assert npt["pcoupl"] != "no"  # pressure coupled second
    assert npt["gen-vel"] == "no"  # velocities carried over
    assert npt["continuation"] == "yes"


def test_production_length_matches_the_requested_time(tmp_path, af2_path):
    structure, parsed, _ = load_structure(
        StructureInput(path=af2_path, source_id="alphafold2"), label=af2_path.name
    )
    pocket_set = BuiltinGeometricBackend().detect(parsed)
    generate_md_setup(
        structure=structure,
        parsed=parsed,
        pocket_set=pocket_set,
        output_dir=tmp_path / "md",
        production_ns=50.0,
    )
    parameters = _parse_mdp((tmp_path / "md" / "md.mdp").read_text())
    # 50 ns at a 2 fs step.
    assert int(parameters["nsteps"]) == 25_000_000


# --------------------------------------------------------------------------------------
# Cleaned structure
# --------------------------------------------------------------------------------------


def test_clean_structure_has_no_hydrogens(md_output):
    """pdb2gmx adds a consistent hydrogen set; leftovers from the model conflict with it."""
    _result, directory, _target = md_output
    for line in (directory / "protein_clean.pdb").read_text().splitlines():
        if line.startswith("ATOM"):
            assert line[76:78].strip().upper() != "H"


def test_clean_structure_keeps_every_protein_residue(md_output, af2_structure):
    _result, directory, _target = md_output
    resseqs = {
        int(line[22:26])
        for line in (directory / "protein_clean.pdb").read_text().splitlines()
        if line.startswith("ATOM")
    }
    assert len(resseqs) == af2_structure.qc.n_residues


def test_clean_structure_is_fixed_width_pdb(md_output):
    """Column alignment matters: pdb2gmx reads by position, not by whitespace."""
    _result, directory, _target = md_output
    for line in (directory / "protein_clean.pdb").read_text().splitlines():
        if line.startswith("ATOM"):
            assert len(line) >= 78
            float(line[30:38]), float(line[38:46]), float(line[46:54])
            int(line[22:26])


def test_clean_structure_is_reparseable(md_output):
    from pocketscribe.parsing import parse_structure

    _result, directory, _target = md_output
    reparsed = parse_structure(directory / "protein_clean.pdb")
    assert reparsed.qc.n_residues > 0


def test_clean_structure_ends_with_ter_and_end(md_output):
    _result, directory, _target = md_output
    lines = (directory / "protein_clean.pdb").read_text().strip().splitlines()
    assert lines[-1] == "END"
    assert "TER" in lines[-3:]


# --------------------------------------------------------------------------------------
# Pocket selection and protocol
# --------------------------------------------------------------------------------------


def test_pocket_selection_lists_the_lining_residues(md_output):
    result, directory, target = md_output
    text = (directory / "pocket_selection.txt").read_text()
    assert "make_ndx" in text

    selections = re.findall(r"^ri ([\d ]+)$", text, flags=re.MULTILINE)
    assert selections
    listed = {int(value) for line in selections for value in line.split()}
    assert listed == {residue.resseq for residue in target.residues}
    assert result.target_pocket_rank == target.rank


def test_pocket_selection_suggests_follow_up_analyses(md_output):
    _result, directory, _target = md_output
    text = (directory / "pocket_selection.txt").read_text()
    assert "gmx rmsf" in text
    assert "gmx sasa" in text


def test_protocol_states_that_no_simulation_was_run(md_output):
    _result, directory, _target = md_output
    text = (directory / "PROTOCOL.md").read_text()
    assert "does not run any simulation" in text


def test_protocol_explains_the_force_field_choice(md_output):
    _result, directory, _target = md_output
    text = (directory / "PROTOCOL.md").read_text()
    assert "CHARMM36m" in text
    assert "ff19SB" in text  # an alternative is offered, not just the default asserted


def test_protocol_warns_about_low_confidence_regions(md_output):
    """The bundled model has a bad region; the protocol must connect that to stability."""
    _result, directory, _target = md_output
    text = (directory / "PROTOCOL.md").read_text()
    assert "pLDDT below 70" in text


def test_setup_script_marks_the_interactive_decisions(md_output):
    _result, directory, _target = md_output
    text = (directory / "run_setup.sh").read_text()
    assert text.startswith("#!/usr/bin/env bash")
    assert "DECISION REQUIRED" in text
    assert "pdb2gmx" in text and "genion" in text and "grompp" in text
    # Choosing the wrong genion group silently ruins the system.
    assert "SOL" in text
