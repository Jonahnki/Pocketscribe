"""Molecular dynamics setup generation (pipeline stage 4).

Pocketscribe does **not** run molecular dynamics. This stage produces the files a
user would otherwise assemble by hand -- a cleaned structure, GROMACS ``.mdp``
parameter files, a preparation script and a written protocol -- so that the handoff
from "we found a pocket" to "we are simulating it" is one command rather than an
afternoon of copying parameters out of tutorials.

Everything here is deterministic template generation. The defaults are the ones a
structural biology group would reasonably start from (CHARMM36m + TIP3P, dodecahedron
box, 0.15 M NaCl, 310 K), and every parameter a user should reconsider for their own
system is commented in place rather than left for them to discover.
"""

from __future__ import annotations

from pathlib import Path

from .errors import MDSetupError
from .models import GeneratedFile, MDSetupResult, Pocket, PocketSet, StructureModel
from .parsing import ParsedStructure

#: Residue names dropped from the MD-ready structure: crystallographic/prediction
#: leftovers that pdb2gmx would either reject or silently mishandle.
_STRIP_RESIDUES = {"HOH", "WAT", "TIP", "TIP3", "SOL", "DOD", "NA", "CL", "SO4", "PO4", "GOL", "EDO", "PEG", "MPD", "ACT", "DMS"}


def generate_md_setup(
    structure: StructureModel,
    parsed: ParsedStructure,
    pocket_set: PocketSet,
    output_dir: Path,
    target_pocket: Pocket | None = None,
    force_field: str = "charmm36m",
    water_model: str = "tip3p",
    box_shape: str = "dodecahedron",
    box_padding_nm: float = 1.0,
    salt_concentration_M: float = 0.15,
    temperature_K: float = 310.0,
    production_ns: float = 100.0,
) -> MDSetupResult:
    """Write a complete, ready-to-run GROMACS setup for ``structure``.

    Parameters
    ----------
    target_pocket:
        When given, an index-group selection and a restraint note are generated for
        the residues lining that pocket, so a user can analyse or restrain the site
        without working out its residue list by hand.
    """
    output_dir = Path(output_dir)
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise MDSetupError(f"Could not create MD output directory '{output_dir}': {exc}") from exc

    files: list[GeneratedFile] = []
    removed: list[str] = []

    clean_path = output_dir / "protein_clean.pdb"
    removed = _write_clean_structure(parsed, clean_path)
    files.append(
        GeneratedFile(
            name=clean_path.name,
            path=str(clean_path),
            kind="structure",
            description=(
                "Protein-only coordinates with waters, ions, ligands and hydrogens "
                "removed, ready for gmx pdb2gmx."
            ),
        )
    )

    mdp_specs = [
        ("minim.mdp", _minim_mdp(), "Steepest-descent energy minimisation."),
        ("ions.mdp", _ions_mdp(), "Dummy parameter file used only by grompp before genion."),
        (
            "nvt.mdp",
            _nvt_mdp(temperature_K),
            "100 ps NVT equilibration with position restraints on the protein.",
        ),
        (
            "npt.mdp",
            _npt_mdp(temperature_K),
            "100 ps NPT equilibration, restraints retained, pressure coupled to 1 bar.",
        ),
        (
            "md.mdp",
            _production_mdp(temperature_K, production_ns),
            f"Unrestrained production run, {production_ns:g} ns at {temperature_K:g} K.",
        ),
    ]
    for name, content, description in mdp_specs:
        path = output_dir / name
        path.write_text(content)
        files.append(
            GeneratedFile(name=name, path=str(path), kind="mdp", description=description)
        )

    script_path = output_dir / "run_setup.sh"
    script_path.write_text(
        _setup_script(
            force_field=force_field,
            water_model=water_model,
            box_shape=box_shape,
            box_padding_nm=box_padding_nm,
            salt_concentration_M=salt_concentration_M,
        )
    )
    script_path.chmod(0o755)
    files.append(
        GeneratedFile(
            name=script_path.name,
            path=str(script_path),
            kind="script",
            description=(
                "System preparation through to production: pdb2gmx, editconf, solvate, "
                "genion, minimisation, NVT, NPT, production."
            ),
        )
    )

    if target_pocket is not None:
        selection_path = output_dir / "pocket_selection.txt"
        selection_path.write_text(_pocket_selection(target_pocket))
        files.append(
            GeneratedFile(
                name=selection_path.name,
                path=str(selection_path),
                kind="notes",
                description=(
                    f"gmx make_ndx selection for the {len(target_pocket.residues)} residues "
                    f"lining pocket {target_pocket.rank}."
                ),
            )
        )

    protocol_path = output_dir / "PROTOCOL.md"
    protocol_path.write_text(
        _protocol(
            structure=structure,
            pocket_set=pocket_set,
            target_pocket=target_pocket,
            force_field=force_field,
            water_model=water_model,
            box_shape=box_shape,
            box_padding_nm=box_padding_nm,
            salt_concentration_M=salt_concentration_M,
            temperature_K=temperature_K,
            production_ns=production_ns,
            removed=removed,
        )
    )
    files.append(
        GeneratedFile(
            name=protocol_path.name,
            path=str(protocol_path),
            kind="protocol",
            description="Step-by-step written protocol with the reasoning behind each choice.",
        )
    )

    notes: list[str] = [
        "Pocketscribe generates these inputs but never runs a simulation; nothing here "
        "has been energy-minimised or equilibrated yet.",
        "CHARMM36m is a sensible general-purpose default for a folded protein. Consider "
        "AMBER ff19SB (with OPC water) for comparison, or a specialised disordered-protein "
        "force field if a large part of your model is low-confidence and likely disordered.",
    ]
    if structure.confidence.fraction_below_70 > 0.3:
        notes.append(
            f"{structure.confidence.fraction_below_70:.0%} of residues in this model have "
            "pLDDT below 70. Long, poorly-predicted loops are the usual cause of an "
            "unstable trajectory; consider trimming or restraining them, and treat the "
            "first part of the production run as extended equilibration."
        )

    return MDSetupResult(
        output_dir=str(output_dir),
        force_field=force_field,
        water_model=water_model,
        box_shape=box_shape,
        box_padding_nm=box_padding_nm,
        salt_concentration_M=salt_concentration_M,
        target_pocket_rank=target_pocket.rank if target_pocket else None,
        target_pocket_id=target_pocket.id if target_pocket else None,
        files=files,
        removed_heteroatoms=removed,
        notes=notes,
    )


def _write_clean_structure(parsed: ParsedStructure, destination: Path) -> list[str]:
    """Write protein-only, hydrogen-free coordinates and report what was removed."""
    removed: set[str] = set()
    lines: list[str] = [
        "REMARK   1 MD-ready structure prepared by Pocketscribe",
        "REMARK   1 Waters, ions, ligands and hydrogens removed; the force field will",
        "REMARK   1 add hydrogens during pdb2gmx. B-factor column retains the source",
        "REMARK   1 model's per-residue confidence for reference.",
    ]
    serial = 1
    last_chain: str | None = None

    for atom in parsed.atoms:
        resname = atom["resname"]
        element = (atom["element"] or "").upper()
        if resname in _STRIP_RESIDUES:
            removed.add(resname)
            continue
        if element == "H":
            continue
        if last_chain is not None and atom["chain"] != last_chain:
            lines.append("TER")
        last_chain = atom["chain"]

        name = atom["name"]
        formatted_name = f"{name:<4}" if len(name) >= 4 else f" {name:<3}"
        x, y, z = atom["coord"]
        lines.append(
            f"ATOM  {serial:>5} {formatted_name}{resname:>4}"
            f" {atom['chain'][:1]}{atom['resseq']:>4}{atom['icode']}   "
            f"{x:>8.3f}{y:>8.3f}{z:>8.3f}"
            f"{atom['occupancy']:>6.2f}{atom['bfactor']:>6.2f}"
            f"          {element:>2}"
        )
        serial += 1

    lines.append("TER")
    lines.append("END")
    destination.write_text("\n".join(lines) + "\n")
    return sorted(removed)


# --------------------------------------------------------------------------------------
# GROMACS .mdp templates
#
# Non-bonded settings follow the CHARMM36 recommendations: Verlet cut-off scheme with a
# force-switched van der Waals potential from 1.0 to 1.2 nm and PME electrostatics with
# a matching 1.2 nm real-space cut-off. Using CHARMM36m with GROMACS defaults instead
# is a common and consequential mistake, so the values are stated explicitly here.
# --------------------------------------------------------------------------------------

_NONBONDED_BLOCK = """\
; ---- Neighbour searching ----
cutoff-scheme            = Verlet
nstlist                  = 20          ; Verlet scheme tunes this at run time
pbc                      = xyz
verlet-buffer-tolerance  = 0.005

; ---- Electrostatics ----
coulombtype              = PME
rcoulomb                 = 1.2
pme-order                = 4
fourierspacing           = 0.12

; ---- Van der Waals (CHARMM36 force-switch; do NOT use GROMACS defaults here) ----
vdwtype                  = cutoff
vdw-modifier             = force-switch
rvdw-switch              = 1.0
rvdw                     = 1.2
DispCorr                 = no          ; force-switch already handles the tail
"""


def _minim_mdp() -> str:
    return f"""\
; Pocketscribe: steepest-descent energy minimisation
; Purpose: remove steric clashes from the predicted model, added hydrogens and the
; solvation step before any dynamics are attempted.
; Reconsider: raise nsteps if the system does not reach emtol; a predicted structure
; with long low-confidence loops sometimes needs more than the default.

integrator               = steep
emtol                    = 1000.0      ; kJ/mol/nm; stop when max force falls below this
emstep                   = 0.01        ; initial step size (nm)
nsteps                   = 50000

{_NONBONDED_BLOCK}\
"""


def _ions_mdp() -> str:
    return f"""\
; Pocketscribe: parameter file used only to build a .tpr for gmx genion.
; No dynamics are run with this file; the values simply need to be valid.

integrator               = steep
emtol                    = 1000.0
emstep                   = 0.01
nsteps                   = 5000

{_NONBONDED_BLOCK}\
"""


def _nvt_mdp(temperature_K: float) -> str:
    return f"""\
; Pocketscribe: NVT equilibration (constant volume, 100 ps)
; Purpose: bring the solvated system to temperature while the protein is held in place
; by position restraints, so water relaxes around the model rather than the model
; being dragged by unequilibrated solvent.
; Reconsider: extend to 250-500 ps for a large system, or if the temperature has not
; settled by the end of the run.

integrator               = md
dt                       = 0.002       ; 2 fs, valid because h-bonds are constrained
nsteps                   = 50000       ; 100 ps
define                   = -DPOSRES    ; position restraints on protein heavy atoms

; ---- Output control ----
nstxout-compressed       = 5000
nstenergy                = 5000
nstlog                   = 5000

{_NONBONDED_BLOCK}
; ---- Bonds ----
constraints              = h-bonds
constraint-algorithm     = lincs
lincs-iter               = 1
lincs-order              = 4
continuation             = no          ; first equilibration step after minimisation

; ---- Temperature coupling ----
tcoupl                   = V-rescale
tc-grps                  = Protein Non-Protein
tau-t                    = 0.1     0.1
ref-t                    = {temperature_K:g}   {temperature_K:g}

; ---- Pressure coupling ----
pcoupl                   = no          ; constant volume in this stage

; ---- Velocity generation ----
gen-vel                  = yes
gen-temp                 = {temperature_K:g}
gen-seed                 = -1          ; random seed; set a fixed value for reproducibility
"""


def _npt_mdp(temperature_K: float) -> str:
    return f"""\
; Pocketscribe: NPT equilibration (constant pressure, 100 ps)
; Purpose: let the box volume and solvent density relax to 1 bar with the protein still
; restrained, so the production run starts from a properly equilibrated density.
; Reconsider: extend to 1 ns for membrane systems or anything with a large solvated
; cavity; check that density has plateaued before moving on.

integrator               = md
dt                       = 0.002
nsteps                   = 50000       ; 100 ps
define                   = -DPOSRES

; ---- Output control ----
nstxout-compressed       = 5000
nstenergy                = 5000
nstlog                   = 5000

{_NONBONDED_BLOCK}
; ---- Bonds ----
constraints              = h-bonds
constraint-algorithm     = lincs
lincs-iter               = 1
lincs-order              = 4
continuation             = yes         ; continues from the NVT run

; ---- Temperature coupling ----
tcoupl                   = V-rescale
tc-grps                  = Protein Non-Protein
tau-t                    = 0.1     0.1
ref-t                    = {temperature_K:g}   {temperature_K:g}

; ---- Pressure coupling ----
pcoupl                   = C-rescale   ; use Berendsen on GROMACS older than 2021
pcoupltype               = isotropic
tau-p                    = 2.0
ref-p                    = 1.0
compressibility          = 4.5e-5
refcoord-scaling         = com

gen-vel                  = no
"""


def _production_mdp(temperature_K: float, production_ns: float) -> str:
    nsteps = int(round(production_ns * 1000.0 / 0.002))
    return f"""\
; Pocketscribe: production MD ({production_ns:g} ns)
; Purpose: the unrestrained trajectory you will actually analyse -- pocket persistence,
; cryptic-pocket opening, and the conformational stability of the binding site.
; Reconsider EVERY one of these for your own system:
;   nsteps        -- {production_ns:g} ns is a starting point, not an answer. Pocket
;                    dynamics questions often need several hundred ns, and independent
;                    replicas (different gen-seed) are worth more than one long run.
;   nstxout-compressed -- 10 ps sampling below; tighten it if you need fast side-chain
;                    motions, loosen it if disk space is the constraint.
;   ref-t         -- {temperature_K:g} K is physiological; use 300 K to compare with
;                    published room-temperature simulations.

integrator               = md
dt                       = 0.002
nsteps                   = {nsteps}    ; {production_ns:g} ns

; ---- Output control ----
nstxout-compressed       = 5000        ; coordinates every 10 ps
compressed-x-grps        = System
nstenergy                = 5000
nstlog                   = 5000

{_NONBONDED_BLOCK}
; ---- Bonds ----
constraints              = h-bonds
constraint-algorithm     = lincs
lincs-iter               = 1
lincs-order              = 4
continuation             = yes

; ---- Temperature coupling ----
tcoupl                   = V-rescale
tc-grps                  = Protein Non-Protein
tau-t                    = 0.1     0.1
ref-t                    = {temperature_K:g}   {temperature_K:g}

; ---- Pressure coupling ----
pcoupl                   = Parrinello-Rahman
pcoupltype               = isotropic
tau-p                    = 2.0
ref-p                    = 1.0
compressibility          = 4.5e-5

; ---- Centre-of-mass motion ----
comm-mode                = Linear
nstcomm                  = 100

gen-vel                  = no
"""


def _setup_script(
    force_field: str,
    water_model: str,
    box_shape: str,
    box_padding_nm: float,
    salt_concentration_M: float,
) -> str:
    """The shell script that drives GROMACS from cleaned PDB to production run."""
    return f"""\
#!/usr/bin/env bash
# Pocketscribe: GROMACS system preparation and equilibration.
#
# Run from inside this directory. Every step is a standard GROMACS command; nothing
# here is Pocketscribe-specific, so you can lift individual steps into your own
# workflow. Read the comments before running: two steps need a decision from you.
#
# Requires: GROMACS 2021 or newer on PATH (gmx or gmx_mpi).

set -euo pipefail

GMX="${{GMX:-gmx}}"

# --------------------------------------------------------------------------------------
# 1. Topology generation
#
# DECISION REQUIRED: pdb2gmx will ask about protonation states for histidines and about
# termini. Defaults are usually fine at pH 7, but if your pocket contains a histidine,
# its protonation state may matter for the question you are asking -- check it.
#
# -ignh strips any hydrogens present in the predicted model so the force field adds its
# own, consistent set.
# --------------------------------------------------------------------------------------
"$GMX" pdb2gmx -f protein_clean.pdb -o processed.gro -p topol.top -i posre.itp \\
    -ff {force_field} -water {water_model} -ignh

# --------------------------------------------------------------------------------------
# 2. Simulation box
#
# A {box_shape} box holds ~71% of the volume of a cube with the same protein-image
# separation, so it cuts the number of water molecules (and therefore the cost of the
# run) by roughly a third for a globular protein.
#
# {box_padding_nm:g} nm padding keeps the protein at least 2 x {box_padding_nm:g} nm from
# its periodic image. Increase it if your protein is elongated or if you expect large
# conformational changes.
# --------------------------------------------------------------------------------------
"$GMX" editconf -f processed.gro -o boxed.gro -c -d {box_padding_nm:g} -bt {box_shape}

# --------------------------------------------------------------------------------------
# 3. Solvation
# --------------------------------------------------------------------------------------
"$GMX" solvate -cp boxed.gro -cs spc216.gro -o solvated.gro -p topol.top

# --------------------------------------------------------------------------------------
# 4. Ions: neutralise the system and add physiological salt ({salt_concentration_M:g} M NaCl)
#
# DECISION REQUIRED: genion asks which group to replace with ions. Choose SOL (water).
# Selecting a protein group here will silently ruin the system.
# --------------------------------------------------------------------------------------
"$GMX" grompp -f ions.mdp -c solvated.gro -p topol.top -o ions.tpr -maxwarn 1
echo SOL | "$GMX" genion -s ions.tpr -o solvated_ions.gro -p topol.top \\
    -pname NA -nname CL -neutral -conc {salt_concentration_M:g}

# --------------------------------------------------------------------------------------
# 5. Energy minimisation
# --------------------------------------------------------------------------------------
"$GMX" grompp -f minim.mdp -c solvated_ions.gro -p topol.top -o em.tpr
"$GMX" mdrun -v -deffnm em

# Check convergence before continuing: the maximum force should be below emtol.
echo Potential | "$GMX" energy -f em.edr -o potential.xvg

# --------------------------------------------------------------------------------------
# 6. NVT equilibration (100 ps, protein restrained)
# --------------------------------------------------------------------------------------
"$GMX" grompp -f nvt.mdp -c em.gro -r em.gro -p topol.top -o nvt.tpr
"$GMX" mdrun -deffnm nvt

# --------------------------------------------------------------------------------------
# 7. NPT equilibration (100 ps, protein restrained)
# --------------------------------------------------------------------------------------
"$GMX" grompp -f npt.mdp -c nvt.gro -r nvt.gro -t nvt.cpt -p topol.top -o npt.tpr
"$GMX" mdrun -deffnm npt

# Confirm the density has plateaued near 1000 kg/m^3 before the production run.
echo Density | "$GMX" energy -f npt.edr -o density.xvg

# --------------------------------------------------------------------------------------
# 8. Production
#
# This is the expensive step. Submit it to your cluster rather than running it here.
# --------------------------------------------------------------------------------------
"$GMX" grompp -f md.mdp -c npt.gro -t npt.cpt -p topol.top -o md.tpr
# "$GMX" mdrun -deffnm md

echo
echo "Setup complete. Inspect potential.xvg and density.xvg, then submit md.tpr."
"""


def _pocket_selection(pocket: Pocket) -> str:
    """A make_ndx selection plus analysis hints for one pocket's residues."""
    by_chain: dict[str, list[int]] = {}
    for residue in pocket.residues:
        by_chain.setdefault(residue.chain, []).append(residue.resseq)

    lines = [
        f"# Pocketscribe: residues lining pocket {pocket.rank} (backend id {pocket.id})",
        f"# {len(pocket.residues)} residues across {len(by_chain)} chain(s).",
        "#",
        "# Create an index group for these residues with:",
        "#",
        "#   gmx make_ndx -f em.gro -o index.ndx",
        "#",
        "# then paste the 'ri' line below at the prompt, followed by 'name <N> Pocket'",
        "# and 'q'. GROMACS renumbers residues consecutively from 1 during pdb2gmx, so",
        "# verify the selection visually before relying on it if your input numbering",
        "# did not start at 1 or contained gaps.",
        "",
    ]
    for chain in sorted(by_chain):
        residue_numbers = " ".join(str(number) for number in sorted(by_chain[chain]))
        lines.append(f"# chain {chain}")
        lines.append(f"ri {residue_numbers}")
        lines.append("")

    lines += [
        "# Useful follow-up analyses once the trajectory exists:",
        "#   gmx rmsf -s md.tpr -f md.xtc -n index.ndx -res      # per-residue flexibility",
        "#   gmx gyrate -s md.tpr -f md.xtc -n index.ndx         # pocket compaction over time",
        "#   gmx sasa -s md.tpr -f md.xtc -n index.ndx           # does the pocket stay open?",
        "#",
        "# For pocket persistence specifically, re-running fpocket on trajectory frames",
        "# (gmx trjconv -sep) and tracking whether the pocket is re-detected is a more",
        "# direct measure than any single geometric proxy.",
    ]
    return "\n".join(lines) + "\n"


def _protocol(
    structure: StructureModel,
    pocket_set: PocketSet,
    target_pocket: Pocket | None,
    force_field: str,
    water_model: str,
    box_shape: str,
    box_padding_nm: float,
    salt_concentration_M: float,
    temperature_K: float,
    production_ns: float,
    removed: list[str],
) -> str:
    """The written protocol: what to run, in what order, and why."""
    pocket_section = ""
    if target_pocket is not None:
        pocket_section = f"""
## The pocket this setup targets

Pocket {target_pocket.rank} (backend id {target_pocket.id}), lined by
{len(target_pocket.residues)} residues.

- Volume: {target_pocket.volume if target_pocket.volume is not None else 'n/a'} A^3
- Druggability score: {target_pocket.druggability_score if target_pocket.druggability_score is not None else 'n/a'} ({target_pocket.score_provenance.value})
- Confidence caveat: {target_pocket.confidence.text}

`pocket_selection.txt` holds a `gmx make_ndx` selection for these residues. Nothing in
the simulation is restricted to the pocket -- the whole protein is simulated -- but the
index group lets you measure what happens to the site specifically.
"""

    low_confidence_note = ""
    # 15% is low enough to catch a model with one substantial disordered loop, which is
    # the usual cause of a trajectory that will not settle.
    if structure.confidence.fraction_below_70 > 0.15:
        low_confidence_note = f"""
> **Before you start.** {structure.confidence.fraction_below_70:.0%} of this model's
> residues have pLDDT below 70 and {structure.confidence.fraction_below_50:.0%} are
> below 50. Poorly-predicted regions are the most common cause of an unstable
> trajectory, and they are usually loops that were never well-ordered to begin with.
> Consider trimming obviously disordered termini, and expect the first tens of
> nanoseconds to behave as extended equilibration rather than production sampling.
"""

    return f"""\
# MD simulation protocol

Generated by Pocketscribe for `{Path(structure.path).name}`
(source: {structure.source.display_name}).

This protocol takes a predicted structure to a running GROMACS simulation. It assumes
you know what molecular dynamics is and what equilibration is for, but not that you
have automated this particular setup before. Every generated file is plain GROMACS
input -- nothing here locks you into Pocketscribe.

**Pocketscribe does not run any simulation.** It writes the inputs; you run them.
{low_confidence_note}{pocket_section}
## What was generated

| File | Purpose |
| --- | --- |
| `protein_clean.pdb` | Protein-only coordinates, hydrogens and heteroatoms stripped |
| `run_setup.sh` | The whole preparation sequence as one script |
| `minim.mdp` | Steepest-descent energy minimisation |
| `ions.mdp` | Dummy parameters so `grompp` can build a `.tpr` for `genion` |
| `nvt.mdp` | 100 ps constant-volume equilibration, protein restrained |
| `npt.mdp` | 100 ps constant-pressure equilibration, protein restrained |
| `md.mdp` | {production_ns:g} ns unrestrained production run |
{"| `pocket_selection.txt` | `make_ndx` selection for the target pocket |" if target_pocket else ""}

## Choices made for you, and when to change them

**Force field: {force_field} with {water_model} water.** CHARMM36m is a reasonable
general-purpose default for a folded globular protein and is well validated for
protein dynamics. It was specifically reparameterised to avoid over-compacting
intrinsically disordered regions, which matters here: a predicted model with large
low-confidence regions is exactly the case where a force field biased toward compact
states will mislead you. Alternatives worth considering are AMBER ff19SB with OPC
water, which some groups prefer for side-chain rotamer accuracy. Do not mix force-field
choices across simulations you intend to compare.

**Non-bonded settings.** The `.mdp` files use the CHARMM36 recommendation: a
force-switched van der Waals potential from 1.0 to 1.2 nm with PME electrostatics at a
matching 1.2 nm cut-off. Running CHARMM36m with GROMACS' default non-bonded settings
is a common and consequential error; the values are stated explicitly in every file so
they survive copy-and-paste.

**Box: {box_shape}, {box_padding_nm:g} nm padding.** A rhombic dodecahedron holds about
71% of the volume of a cube at the same minimum image distance, so it cuts solvent
count and cost by roughly a third. Increase the padding for an elongated protein or if
you expect a large hinge motion.

**Ions: neutralising counter-ions plus {salt_concentration_M:g} M NaCl.** Physiological
ionic strength. Raise it if you are studying a highly charged interface; note that the
neutralising ions alone are not a substitute for real salt.

**Temperature: {temperature_K:g} K.** Physiological. Use 300 K if you want to compare
against the large body of room-temperature simulation literature.

**Timestep: 2 fs with h-bond constraints.** Standard. Do not raise it without switching
to hydrogen mass repartitioning, and do not remove the constraints while keeping 2 fs.

## Running it

```bash
cd {Path('md_setup').name}
./run_setup.sh
```

The script stops for two decisions: histidine protonation and termini during
`pdb2gmx`, and the group to replace with ions during `genion` (choose `SOL`). Both are
marked in the script.

After `run_setup.sh` completes you will have `md.tpr`. The production run is the
expensive step; submit it to your cluster:

```bash
gmx mdrun -deffnm md
```

## Checks before you trust the trajectory

1. **Minimisation converged** -- maximum force below `emtol`, potential energy negative
   and stable. If it did not converge, the model probably has a steric problem that
   minimisation cannot fix; look at the structure.
2. **Temperature settled** during NVT, around {temperature_K:g} K.
3. **Density plateaued** during NPT, near 1000 kg/m^3 for a standard aqueous system.
4. **Backbone RMSD** over the production run should rise and then plateau. A model that
   never plateaus is still relaxing out of its predicted conformation, which is common
   for low-confidence predictions and means your sampling has not started yet.

## What this does and does not tell you about the pocket

A single unbiased trajectory tells you whether the pocket stays open under
thermal motion, how flexible its lining residues are, and whether a cryptic site opens
on this timescale. It does not tell you whether a ligand binds, what the affinity is,
or whether the pocket is druggable in any experimentally meaningful sense. Those
questions need docking, free-energy methods, or an experiment.

Independent replicas started from different velocity seeds are more informative than
one long run of the same total length. Change `gen-seed` in `nvt.mdp` between replicas.

## Removed from the input structure

{', '.join(removed) if removed else 'Nothing -- the input contained protein only.'}

Hydrogens were stripped so that `pdb2gmx` adds a consistent set matching the force
field.

---

*Generated by Pocketscribe. Research use only. A simulation of a predicted structure
inherits every uncertainty of that prediction -- see the confidence caveats in the main
report.*
"""
