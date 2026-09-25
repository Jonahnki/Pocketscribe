#!/usr/bin/env python3
"""Generate the bundled example structures.

The structures Pocketscribe ships for ``pocketscribe demo`` and the test suite are
**synthetic**, not real proteins. That is deliberate:

* the demo and the tests must run with no network access and no external downloads;
* the geometry is known in advance, so a test can assert that a cavity of a particular
  size is found rather than asserting whatever the tool happened to produce;
* nothing is redistributed whose licensing would need checking.

The fold is a chain wound over a spherical shell. It is not protein-like in any
biological sense, but it has the properties the pipeline needs to be exercised
honestly: a continuous backbone with realistic Ca-Ca spacing, a large enclosed interior
cavity that any buriedness-based detector will find, and per-residue confidence values
laid out so that one region is confidently predicted and another is not.

Run this script to regenerate the bundled files:

    python scripts/make_example_structures.py

It is checked in so that the provenance of every bundled file is auditable: nothing in
``pocketscribe/data/example_structures`` is hand-edited.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "pocketscribe" / "data" / "example_structures"

CA_SPACING = 3.8  # Angstrom, the real Ca-Ca distance in a polypeptide

# Backbone geometry, approximated well enough to give plausible atom counts and
# volumes. These are not refined values and the structures are not meant to be
# energy-minimised.
N_OFFSET = 1.45
C_OFFSET = 1.52
O_OFFSET = 1.23
CB_OFFSET = 1.53

#: A repeating pattern with a hydrophobic majority, so the cavity lining has a
#: realistic apolar character.
SEQUENCE_A = "ALVIFLGAVLTAIFVGLASVLAMFGTVLIAVFGSLVTAMLIVFGA"
SEQUENCE_B = "EKDRQNSEKHDRQNGEKDRSQNTEKHDRPQNSEKDRQNGEKHDRS"


#: alpha-helix geometry: Ca atoms sit on a 2.3 A radius helix, rising 1.5 A and turning
#: 100 degrees per residue. That gives a Ca-Ca distance of 3.83 A, matching a real
#: polypeptide closely enough that the chain-break check does not fire.
HELIX_RADIUS = 2.3
HELIX_RISE = 1.5
HELIX_TURN = np.deg2rad(100.0)


def helix_bundle(
    n_helices: int,
    residues_per_helix: int,
    bundle_radius: float,
    loop_length: int = 4,
) -> tuple[np.ndarray, np.ndarray]:
    """Build an antiparallel helix bundle.

    A helix bundle is a real fold class, which matters for a demo: it produces the kind
    of surface clefts between packed secondary-structure elements that a pocket detector
    is meant to find, rather than one implausible balloon of empty space.

    Returns the Ca coordinates and, for each residue, the reference point its side chain
    points away from (the local helix axis for helix residues, the bundle centre for
    loops), so that side chains project outward the way they do in a real bundle.
    """
    ca_points: list[np.ndarray] = []
    axis_points: list[np.ndarray] = []

    for helix_index in range(n_helices):
        angle = 2.0 * np.pi * helix_index / n_helices
        centre = np.array([bundle_radius * np.cos(angle), bundle_radius * np.sin(angle), 0.0])
        # Antiparallel: consecutive helices run in opposite directions, as in a real
        # up-down-up-down bundle.
        direction = 1.0 if helix_index % 2 == 0 else -1.0
        z_start = -direction * (residues_per_helix - 1) * HELIX_RISE / 2.0
        phase = angle + np.pi  # face the helix's polar side outward

        helix: list[np.ndarray] = []
        axes: list[np.ndarray] = []
        for residue_index in range(residues_per_helix):
            theta = phase + residue_index * HELIX_TURN
            z = z_start + direction * residue_index * HELIX_RISE
            helix.append(
                centre
                + np.array(
                    [HELIX_RADIUS * np.cos(theta), HELIX_RADIUS * np.sin(theta), z]
                )
            )
            axes.append(centre + np.array([0.0, 0.0, z]))

        if ca_points:
            # Connect the previous helix's C-terminus to this helix's N-terminus with a
            # loop whose steps are the right length, so the backbone stays continuous.
            loop = _loop_between(ca_points[-1], helix[0], loop_length)
            ca_points.extend(loop)
            axis_points.extend([np.zeros(3)] * len(loop))

        ca_points.extend(helix)
        axis_points.extend(axes)

    return np.array(ca_points), np.array(axis_points)


def two_domain_bundle(
    bundle_a: tuple[int, int, float],
    bundle_b: tuple[int, int, float],
    separation: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Two helix bundles joined by a linker.

    A single bundle produces exactly one cavity -- its central channel -- which makes a
    poor demonstration of a tool whose job is to *rank* pockets. Two domains of
    different packing produce two pockets of clearly different size and quality, which
    is what the ranked table and the confidence caveats are for.

    Each tuple is ``(n_helices, residues_per_helix, bundle_radius)``.
    """
    ca_a, axes_a = helix_bundle(*bundle_a)
    ca_b, axes_b = helix_bundle(*bundle_b)

    shift = np.array([separation, 0.0, 0.0])
    ca_b = ca_b + shift
    axes_b = axes_b + shift

    linker = _loop_between(ca_a[-1], ca_b[0], 4)
    ca = np.vstack([ca_a, np.array(linker), ca_b])
    axes = np.vstack([axes_a, np.zeros((len(linker), 3)), axes_b])
    return ca, axes


def _loop_between(start: np.ndarray, end: np.ndarray, n_points: int) -> list[np.ndarray]:
    """Intermediate Ca positions joining two helices, bowed outward like a real loop."""
    separation = np.linalg.norm(end - start)
    # Enough points that no step exceeds a plausible Ca-Ca distance.
    count = max(n_points, int(np.ceil(separation / CA_SPACING)) - 1)
    outward = np.array([start[0] + end[0], start[1] + end[1], 0.0])
    norm = np.linalg.norm(outward)
    outward = outward / norm if norm > 1e-6 else np.array([1.0, 0.0, 0.0])

    points: list[np.ndarray] = []
    for step in range(1, count + 1):
        fraction = step / (count + 1)
        # A sine bow pushes the loop away from the bundle axis instead of cutting
        # straight through the core.
        bow = np.sin(np.pi * fraction) * 3.0
        points.append(start + (end - start) * fraction + outward * bow)
    return points


def build_residues(
    ca_coords: np.ndarray, sequence: str, outward_reference: np.ndarray | None = None
) -> list[dict]:
    """Place N, CA, C, O and CB around each Ca position.

    Side chains (represented by CB alone) point away from ``outward_reference`` -- the
    local helix axis, so that they pack outward from each helix the way real side chains
    do, partly filling the space between helices and leaving surface clefts behind.
    """
    residues: list[dict] = []
    n_residues = len(ca_coords)
    centre = ca_coords.mean(axis=0)

    for index, ca in enumerate(ca_coords):
        # At the chain ends there is no neighbour on one side, so the residue's own
        # position stands in. Using the *other* neighbour on both sides would give a
        # zero direction vector and collapse N, CA and C onto one point.
        previous = ca_coords[index - 1] if index > 0 else ca
        following = ca_coords[index + 1] if index < n_residues - 1 else ca

        direction = following - previous
        direction /= max(np.linalg.norm(direction), 1e-9)

        reference = (
            outward_reference[index] if outward_reference is not None else centre
        )
        outward = ca - reference
        if np.linalg.norm(outward) < 1e-6:
            outward = ca - centre
        outward /= max(np.linalg.norm(outward), 1e-9)

        perpendicular = np.cross(direction, outward)
        perpendicular /= max(np.linalg.norm(perpendicular), 1e-9)

        residues.append(
            {
                "resname": three_letter(sequence[index % len(sequence)]),
                "resseq": index + 1,
                "atoms": [
                    ("N", "N", ca - direction * N_OFFSET),
                    ("CA", "C", ca),
                    ("C", "C", ca + direction * C_OFFSET),
                    ("O", "O", ca + direction * C_OFFSET + perpendicular * O_OFFSET),
                    ("CB", "C", ca + outward * CB_OFFSET),
                ],
            }
        )
    return residues


ONE_TO_THREE = {
    "A": "ALA", "R": "ARG", "N": "ASN", "D": "ASP", "C": "CYS",
    "Q": "GLN", "E": "GLU", "G": "GLY", "H": "HIS", "I": "ILE",
    "L": "LEU", "K": "LYS", "M": "MET", "F": "PHE", "P": "PRO",
    "S": "SER", "T": "THR", "W": "TRP", "Y": "TYR", "V": "VAL",
}


def three_letter(code: str) -> str:
    return ONE_TO_THREE[code]


def confidence_profile(
    n_residues: int,
    baseline: float,
    low_region: tuple[int, int] | None,
    low_value: float,
    seed: int,
) -> np.ndarray:
    """Per-residue pLDDT with a deliberately poorly-predicted stretch.

    A demo whose every residue is confidently predicted would never exercise the part of
    Pocketscribe that matters most, so one region is always laid out below the
    interpretation thresholds.
    """
    rng = np.random.default_rng(seed)
    values = np.clip(rng.normal(baseline, 3.0, n_residues), 30.0, 98.0)
    if low_region is not None:
        start, end = low_region
        span = end - start
        # A smooth trough rather than a step: real low-confidence regions taper.
        taper = np.sin(np.linspace(0, np.pi, span))
        values[start:end] = np.clip(
            values[start:end] - taper * (baseline - low_value), 20.0, 98.0
        )
    return np.round(values, 2)


def write_pdb(
    path: Path,
    residues: list[dict],
    confidences: np.ndarray,
    header_lines: list[str],
    chain: str = "A",
) -> None:
    """Write a PDB file with per-residue confidence in the B-factor column."""
    lines = list(header_lines)
    serial = 1
    for residue, confidence in zip(residues, confidences, strict=True):
        for name, element, coord in residue["atoms"]:
            formatted_name = f" {name:<3}"
            lines.append(
                f"ATOM  {serial:>5} {formatted_name}{residue['resname']:>4}"
                f" {chain}{residue['resseq']:>4}    "
                f"{coord[0]:>8.3f}{coord[1]:>8.3f}{coord[2]:>8.3f}"
                f"{1.00:>6.2f}{confidence:>6.2f}"
                f"          {element:>2}"
            )
            serial += 1
    lines.append("TER")
    lines.append("END")
    path.write_text("\n".join(lines) + "\n")


def perturb(coords: np.ndarray, magnitude: float, seed: int) -> np.ndarray:
    """Jitter coordinates, then rotate and translate the whole model.

    The rigid-body part matters: it forces the consensus module to actually superpose
    the structures rather than getting the right answer because every model happened to
    be written in the same frame.
    """
    rng = np.random.default_rng(seed)
    jittered = coords + rng.normal(0.0, magnitude, coords.shape)

    angle = 0.7
    rotation = np.array(
        [
            [np.cos(angle), -np.sin(angle), 0.0],
            [np.sin(angle), np.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    return (rotation @ jittered.T).T + np.array([12.0, -5.0, 3.0])


def _write_mmcif_copy(pdb_path: Path, cif_path: Path) -> None:
    """Convert a generated PDB to mmCIF via Biopython's own writer."""
    import warnings

    from Bio.PDB import MMCIFIO, PDBParser

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        structure = PDBParser(QUIET=True).get_structure(pdb_path.stem, str(pdb_path))
    writer = MMCIFIO()
    writer.set_structure(structure)
    writer.save(str(cif_path))


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ---- Protein 1: the demo protein, predicted by four "different tools" -----------
    # Domain 1 packs loosely (a large, well-formed central cavity); domain 2 packs
    # tightly (a small, marginal one). The ranked pocket table therefore has something
    # to rank.
    ca, axes = two_domain_bundle(
        bundle_a=(4, 22, 8.0), bundle_b=(4, 18, 7.0), separation=27.0
    )
    n_residues = len(ca)
    print(f"Protein 1: {n_residues} residues, two four-helix-bundle domains")

    # The low-confidence stretch is placed over the SECOND domain deliberately, so the
    # demo report shows the situation this tool exists for: a pocket that is both lower
    # scoring and sitting in structure the prediction does not support.
    low_region = (int(n_residues * 0.60), int(n_residues * 0.86))

    # AlphaFold2: high confidence overall, one poorly-predicted stretch.
    residues_af2 = build_residues(ca, SEQUENCE_A, axes)
    confidence_af2 = confidence_profile(n_residues, 92.0, low_region, 40.0, seed=1)
    write_pdb(
        OUTPUT_DIR / "AF-SYNTH01-F1-model_v4.pdb",
        residues_af2,
        confidence_af2,
        [
            "HEADER    PREDICTED STRUCTURE                     01-JAN-24   SYN1",
            "TITLE     ALPHAFOLD MONOMER V2.0 PREDICTION FOR SYNTHETIC TEST PROTEIN 1",
            "REMARK   1 SYNTHETIC STRUCTURE GENERATED BY POCKETSCRIBE FOR TESTING.",
            "REMARK   1 NOT A REAL PROTEIN AND NOT A REAL ALPHAFOLD PREDICTION.",
            "REMARK   2 PLDDT IS STORED IN THE B-FACTOR COLUMN ON A 0-100 SCALE.",
        ],
    )

    # OpenFold: same architecture family as AlphaFold2, so a slightly different model of
    # the same fold. Used to demonstrate SAME-family agreement.
    residues_openfold = build_residues(perturb(ca, 0.35, seed=11), SEQUENCE_A, perturb(axes, 0.0, seed=11))
    confidence_openfold = confidence_profile(n_residues, 90.0, low_region, 43.0, seed=2)
    write_pdb(
        OUTPUT_DIR / "synth01_openfold_model_1.pdb",
        residues_openfold,
        confidence_openfold,
        [
            "HEADER    PREDICTED STRUCTURE                     01-JAN-24   SYN1",
            "TITLE     OPENFOLD PREDICTION FOR SYNTHETIC TEST PROTEIN 1",
            "REMARK   1 SYNTHETIC STRUCTURE GENERATED BY POCKETSCRIBE FOR TESTING.",
            "REMARK   2 PLDDT IS STORED IN THE B-FACTOR COLUMN ON A 0-100 SCALE.",
        ],
    )

    # ColabFold: also the AlphaFold2 family, distinguished by its filename and headers.
    residues_colabfold = build_residues(perturb(ca, 0.4, seed=12), SEQUENCE_A, perturb(axes, 0.0, seed=12))
    confidence_colabfold = confidence_profile(n_residues, 88.0, low_region, 45.0, seed=3)
    write_pdb(
        OUTPUT_DIR / "synth01_unrelaxed_rank_001_alphafold2_model_3_seed_000.pdb",
        residues_colabfold,
        confidence_colabfold,
        [
            "HEADER    PREDICTED STRUCTURE                     01-JAN-24   SYN1",
            "TITLE     COLABFOLD V1.5.5 PREDICTION (MMSEQS2 MSA, ALPHAFOLD2 BACKEND)",
            "REMARK   1 SYNTHETIC STRUCTURE GENERATED BY POCKETSCRIBE FOR TESTING.",
            "REMARK   2 PLDDT IS STORED IN THE B-FACTOR COLUMN ON A 0-100 SCALE.",
        ],
    )

    # ESMFold: a different architecture family (single-sequence language model), which
    # is what makes CROSS-family agreement possible in the demo.
    #
    # Its second domain is built more tightly packed, so the marginal cavity there is
    # absent from this model. That disagreement is the point: the large pocket ends up
    # with cross-family support from all three models, while the small one is supported
    # only by AlphaFold2 and OpenFold -- two members of one family that can reproduce
    # each other's errors. The demo report then shows both kinds of evidence side by
    # side, which is exactly the distinction the consensus module exists to draw.
    ca_esmfold, axes_esmfold = two_domain_bundle(
        bundle_a=(4, 22, 8.0), bundle_b=(4, 18, 6.0), separation=27.0
    )
    residues_esmfold = build_residues(
        perturb(ca_esmfold, 0.55, seed=13), SEQUENCE_A, perturb(axes_esmfold, 0.0, seed=13)
    )
    confidence_esmfold = confidence_profile(
        len(ca_esmfold), low_region=low_region, baseline=80.0, low_value=38.0, seed=4
    )
    write_pdb(
        OUTPUT_DIR / "synth01_esmfold.pdb",
        residues_esmfold,
        confidence_esmfold,
        [
            "HEADER    PREDICTED STRUCTURE                     01-JAN-24   SYN1",
            "TITLE     ESMFOLD V1 PREDICTION FOR SYNTHETIC TEST PROTEIN 1",
            "REMARK   1 SYNTHETIC STRUCTURE GENERATED BY POCKETSCRIBE FOR TESTING.",
            "REMARK   1 SINGLE-SEQUENCE PREDICTION; NO MSA WAS USED.",
            "REMARK   2 PLDDT IS STORED IN THE B-FACTOR COLUMN ON A 0-100 SCALE.",
        ],
    )

    # Boltz: the Tier 2 reference adapter. Confidence lives in a sidecar JSON on a 0-1
    # scale, NOT in the B-factor column, so the B-factors here are deliberately written
    # as zeros -- an adapter that wrongly read them would produce obviously wrong output.
    residues_boltz = build_residues(perturb(ca, 0.45, seed=14), SEQUENCE_A, perturb(axes, 0.0, seed=14))
    boltz_confidence = confidence_profile(n_residues, 86.0, low_region, 41.0, seed=5)
    write_pdb(
        OUTPUT_DIR / "boltz_synth01_model_0.pdb",
        residues_boltz,
        np.zeros(n_residues),
        [
            "HEADER    PREDICTED STRUCTURE                     01-JAN-24   SYN1",
            "TITLE     BOLTZ-1 PREDICTION FOR SYNTHETIC TEST PROTEIN 1",
            "REMARK   1 SYNTHETIC STRUCTURE GENERATED BY POCKETSCRIBE FOR TESTING.",
            "REMARK   2 CONFIDENCE IS NOT IN THE B-FACTOR COLUMN. SEE THE SIDECAR JSON.",
        ],
    )
    (OUTPUT_DIR / "confidence_boltz_synth01_model_0.json").write_text(
        json.dumps(
            {
                "confidence_score": 0.84,
                "ptm": 0.81,
                "iptm": 0.0,
                "plddt": [round(float(value) / 100.0, 4) for value in boltz_confidence],
            },
            indent=2,
        )
        + "\n"
    )

    # An mmCIF copy of the AlphaFold2 model, so the mmCIF parsing path is exercised on
    # real output of the writer rather than on a hand-written file.
    _write_mmcif_copy(
        OUTPUT_DIR / "AF-SYNTH01-F1-model_v4.pdb",
        OUTPUT_DIR / "AF-SYNTH01-F1-model_v4.cif",
    )

    # ---- Protein 2: a different protein, for the mismatch-rejection path -----------
    ca_other, axes_other = helix_bundle(
        n_helices=3, residues_per_helix=20, bundle_radius=8.0
    )
    residues_other = build_residues(ca_other, SEQUENCE_B, axes_other)
    confidence_other = confidence_profile(len(ca_other), 89.0, None, 0.0, seed=6)
    print(f"Protein 2: {len(ca_other)} residues, three-helix bundle")
    write_pdb(
        OUTPUT_DIR / "AF-SYNTH02-F1-model_v4.pdb",
        residues_other,
        confidence_other,
        [
            "HEADER    PREDICTED STRUCTURE                     01-JAN-24   SYN2",
            "TITLE     ALPHAFOLD MONOMER V2.0 PREDICTION FOR SYNTHETIC TEST PROTEIN 2",
            "REMARK   1 SYNTHETIC STRUCTURE GENERATED BY POCKETSCRIBE FOR TESTING.",
            "REMARK   1 A DIFFERENT PROTEIN FROM SYNTH01; USED TO TEST THAT CONSENSUS",
            "REMARK   1 REFUSES TO COMPARE NON-IDENTICAL SEQUENCES.",
            "REMARK   2 PLDDT IS STORED IN THE B-FACTOR COLUMN ON A 0-100 SCALE.",
        ],
    )

    # ---- An experimental-style file, for the "do not read B-factors as pLDDT" test --
    # Per-atom temperature factors that vary within each residue, as a refined
    # crystallographic model's would.
    rng = np.random.default_rng(99)
    lines = [
        "HEADER    HYDROLASE                               01-JAN-99   XRAY",
        "TITLE     SYNTHETIC EXPERIMENT-LIKE STRUCTURE (NOT A REAL DEPOSITION)",
        "REMARK   1 B-FACTOR COLUMN HOLDS PER-ATOM TEMPERATURE FACTORS, NOT PLDDT.",
        "REMARK   2 RESOLUTION.    1.80 ANGSTROMS.",
    ]
    serial = 1
    for residue in build_residues(ca[:60], SEQUENCE_A, axes[:60]):
        base = float(rng.uniform(12.0, 45.0))
        for name, element, coord in residue["atoms"]:
            bfactor = round(base + float(rng.normal(0.0, 4.0)), 2)
            lines.append(
                f"ATOM  {serial:>5}  {name:<3}{residue['resname']:>4}"
                f" A{residue['resseq']:>4}    "
                f"{coord[0]:>8.3f}{coord[1]:>8.3f}{coord[2]:>8.3f}"
                f"{1.00:>6.2f}{max(bfactor, 2.0):>6.2f}"
                f"          {element:>2}"
            )
            serial += 1
    lines += ["TER", "END"]
    (OUTPUT_DIR / "experimental_like.pdb").write_text("\n".join(lines) + "\n")

    print(f"\nWrote {len(list(OUTPUT_DIR.glob('*')))} files to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
