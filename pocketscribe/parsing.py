"""Structure parsing, validation and quality control (pipeline stage 1).

Reads PDB and mmCIF, validates that the file is a protein model rather than a
nucleic acid or a bare sequence, extracts one-letter sequences and coordinates, and
produces the :class:`~pocketscribe.models.QCReport` that every later stage annotates
against.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
from Bio.PDB import MMCIFParser, PDBParser
from Bio.PDB.Polypeptide import is_aa

from .errors import InputStructureError
from .models import ChainBreak, QCReport

#: Three-letter to one-letter code for the standard residues, plus the handful of
#: modified residues that prediction tools occasionally emit.
THREE_TO_ONE: dict[str, str] = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
    "MSE": "M", "SEC": "U", "PYL": "O", "HSD": "H", "HSE": "H", "HSP": "H",
    "CYX": "C", "HID": "H", "HIE": "H", "HIP": "H", "ASH": "D", "GLH": "E",
    "LYN": "K",
}

#: Residue names that identify a nucleic acid chain.
_NUCLEIC = {
    "DA", "DC", "DG", "DT", "DI", "A", "C", "G", "U", "I", "RA", "RC", "RG", "RU",
}

#: Cα-Cα distance above which consecutive residues are considered a chain break.
#: Consecutive residues in a real chain sit at ~3.8 A; 4.5 A allows for strain while
#: still catching genuine gaps.
CA_BREAK_THRESHOLD = 4.5


class ParsedStructure:
    """A parsed structure held in the plain form the rest of the pipeline consumes.

    Deliberately not a pydantic model: it carries a numpy coordinate array and the
    Biopython object, neither of which should ever be serialised into the JSON handed
    to the narrative layer.
    """

    def __init__(
        self,
        path: Path,
        file_format: str,
        residues: list[dict],
        atoms: list[dict],
        sequences: dict[str, str],
        qc: QCReport,
        biopython_structure,
    ) -> None:
        self.path = path
        self.file_format = file_format
        #: One dict per residue: chain, resseq, icode, resname, one_letter, bfactors,
        #: ca_coord, atom_indices.
        self.residues = residues
        #: One dict per atom: chain, resseq, icode, resname, name, element, coord, bfactor.
        self.atoms = atoms
        self.sequences = sequences
        self.qc = qc
        self.biopython_structure = biopython_structure

    @property
    def coords(self) -> np.ndarray:
        """``(n_atoms, 3)`` coordinate array in the order of :attr:`atoms`."""
        return np.array([atom["coord"] for atom in self.atoms], dtype=float)

    def ca_coords(self, chain: str | None = None) -> tuple[list[tuple[str, int, str]], np.ndarray]:
        """Cα coordinates and their residue keys, optionally restricted to one chain."""
        keys: list[tuple[str, int, str]] = []
        points: list[np.ndarray] = []
        for residue in self.residues:
            if chain is not None and residue["chain"] != chain:
                continue
            if residue["ca_coord"] is None:
                continue
            keys.append((residue["chain"], residue["resseq"], residue["icode"]))
            points.append(residue["ca_coord"])
        array = np.array(points, dtype=float) if points else np.zeros((0, 3))
        return keys, array

    def residue_by_key(self, key: tuple[str, int, str]) -> dict | None:
        for residue in self.residues:
            if (residue["chain"], residue["resseq"], residue["icode"]) == key:
                return residue
        return None


def _detect_format(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in {".cif", ".mmcif"}:
        return "mmcif"
    if suffix in {".pdb", ".ent"}:
        return "pdb"
    # Fall back to sniffing: mmCIF files start with a data_ block.
    head = path.read_text(errors="replace")[:2048].lstrip()
    if head.startswith("data_") or "_atom_site." in head:
        return "mmcif"
    if head.startswith(("HEADER", "ATOM", "MODEL", "REMARK", "TITLE", "CRYST1")):
        return "pdb"
    raise InputStructureError(
        f"Cannot tell whether '{path.name}' is PDB or mmCIF. Pocketscribe accepts "
        ".pdb, .ent, .cif and .mmcif files containing a protein model."
    )


def parse_structure(path: str | Path) -> ParsedStructure:
    """Parse a PDB or mmCIF protein model.

    Raises
    ------
    InputStructureError
        If the file is missing, unparseable, empty of protein, or is a nucleic acid
        model or a bare sequence rather than a structure.
    """
    path = Path(path)
    if not path.exists():
        raise InputStructureError(f"Structure file not found: {path}")
    if path.is_dir():
        raise InputStructureError(
            f"'{path}' is a directory. Pass a single PDB or mmCIF file; for "
            "cross-model consensus pass --pdb once per model."
        )
    if path.stat().st_size == 0:
        raise InputStructureError(f"Structure file is empty: {path}")

    file_format = _detect_format(path)
    parser = (
        MMCIFParser(QUIET=True) if file_format == "mmcif" else PDBParser(QUIET=True, PERMISSIVE=True)
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            structure = parser.get_structure(path.stem, str(path))
        except Exception as exc:  # Biopython raises a variety of parse errors
            raise InputStructureError(
                f"Could not parse '{path.name}' as {file_format.upper()}: {exc}"
            ) from exc

    try:
        model = next(structure.get_models())
    except StopIteration as exc:
        raise InputStructureError(
            f"'{path.name}' contains no model records. If this is a FASTA sequence "
            "rather than a structure, predict a structure first (e.g. with ColabFold "
            "or ESMFold) and pass the resulting file."
        ) from exc

    residues: list[dict] = []
    atoms: list[dict] = []
    sequences: dict[str, str] = {}
    nonstandard: list[str] = []
    nucleic_residue_count = 0
    total_residue_count = 0
    has_hydrogens = False

    for chain in model:
        chain_id = chain.id.strip() or "A"
        chain_sequence: list[str] = []
        for residue in chain:
            hetflag, resseq, icode = residue.id
            resname = residue.get_resname().strip().upper()
            total_residue_count += 1

            if resname in _NUCLEIC:
                nucleic_residue_count += 1
                continue
            if hetflag.strip() and resname not in THREE_TO_ONE:
                # Waters, ions and ligands: recorded for the MD cleaning step but not
                # part of the protein model.
                continue
            if not (is_aa(resname, standard=False) or resname in THREE_TO_ONE):
                continue

            one_letter = THREE_TO_ONE.get(resname, "X")
            if one_letter == "X" and resname not in THREE_TO_ONE:
                nonstandard.append(resname)

            bfactors: list[float] = []
            atom_indices: list[int] = []
            ca_coord = None
            for atom in residue:
                element = (atom.element or "").strip().upper()
                if element == "H":
                    has_hydrogens = True
                coord = np.array(atom.get_coord(), dtype=float)
                atoms.append(
                    {
                        "chain": chain_id,
                        "resseq": int(resseq),
                        "icode": icode,
                        "resname": resname,
                        "name": atom.get_name().strip(),
                        "element": element or atom.get_name().strip()[:1],
                        "coord": coord,
                        "bfactor": float(atom.get_bfactor()),
                        "occupancy": float(atom.get_occupancy() or 1.0),
                    }
                )
                atom_indices.append(len(atoms) - 1)
                bfactors.append(float(atom.get_bfactor()))
                if atom.get_name().strip() == "CA":
                    ca_coord = coord

            residues.append(
                {
                    "chain": chain_id,
                    "resseq": int(resseq),
                    "icode": icode,
                    "resname": resname,
                    "one_letter": one_letter,
                    "bfactors": bfactors,
                    "ca_coord": ca_coord,
                    "atom_indices": atom_indices,
                }
            )
            chain_sequence.append(one_letter)

        if chain_sequence:
            sequences[chain_id] = "".join(chain_sequence)

    if not residues:
        if nucleic_residue_count:
            raise InputStructureError(
                f"'{path.name}' contains {nucleic_residue_count} nucleic-acid residues and "
                "no protein residues. Pocketscribe analyses protein models only."
            )
        raise InputStructureError(
            f"'{path.name}' contains no recognisable amino-acid residues. Pocketscribe "
            "needs a protein model (single- or multi-chain)."
        )

    if nucleic_residue_count and nucleic_residue_count > len(residues):
        raise InputStructureError(
            f"'{path.name}' is predominantly nucleic acid "
            f"({nucleic_residue_count} nucleotides vs {len(residues)} amino acids). "
            "Pocketscribe analyses protein models only."
        )

    qc = _build_qc(
        residues=residues,
        atoms=atoms,
        sequences=sequences,
        nonstandard=sorted(set(nonstandard)),
        has_hydrogens=has_hydrogens,
        nucleic_residue_count=nucleic_residue_count,
    )

    return ParsedStructure(
        path=path,
        file_format=file_format,
        residues=residues,
        atoms=atoms,
        sequences=sequences,
        qc=qc,
        biopython_structure=structure,
    )


def _build_qc(
    residues: list[dict],
    atoms: list[dict],
    sequences: dict[str, str],
    nonstandard: list[str],
    has_hydrogens: bool,
    nucleic_residue_count: int,
) -> QCReport:
    """Geometry sanity checks: chain breaks, numbering gaps, odd residues."""
    chain_breaks: list[ChainBreak] = []
    numbering_gaps: dict[str, list[tuple[int, int]]] = {}

    by_chain: dict[str, list[dict]] = {}
    for residue in residues:
        by_chain.setdefault(residue["chain"], []).append(residue)

    for chain_id, chain_residues in by_chain.items():
        ordered = sorted(chain_residues, key=lambda r: (r["resseq"], r["icode"]))
        for previous, current in zip(ordered, ordered[1:], strict=False):
            if previous["ca_coord"] is not None and current["ca_coord"] is not None:
                distance = float(np.linalg.norm(current["ca_coord"] - previous["ca_coord"]))
                if distance > CA_BREAK_THRESHOLD:
                    chain_breaks.append(
                        ChainBreak(
                            chain=chain_id,
                            after_resseq=previous["resseq"],
                            before_resseq=current["resseq"],
                            ca_distance=round(distance, 2),
                        )
                    )
            gap = current["resseq"] - previous["resseq"]
            if gap > 1:
                numbering_gaps.setdefault(chain_id, []).append(
                    (previous["resseq"] + 1, current["resseq"] - 1)
                )

    notes: list[str] = []
    if chain_breaks:
        notes.append(
            f"{len(chain_breaks)} backbone discontinuity/discontinuities detected "
            f"(Ca-Ca > {CA_BREAK_THRESHOLD} A). Pockets spanning a break should be "
            "treated with extra caution."
        )
    if numbering_gaps:
        total_gaps = sum(len(gaps) for gaps in numbering_gaps.values())
        notes.append(
            f"{total_gaps} gap(s) in residue numbering, i.e. residues present in the "
            "sequence but absent from the model."
        )
    if nonstandard:
        notes.append(f"Non-standard residue names present: {', '.join(nonstandard)}.")
    if nucleic_residue_count:
        notes.append(
            f"{nucleic_residue_count} nucleic-acid residue(s) were present and were "
            "excluded from the protein analysis."
        )
    if has_hydrogens:
        notes.append(
            "Hydrogens are present in the input; they are stripped before MD topology "
            "generation so that the force field can add its own."
        )

    return QCReport(
        n_chains=len(sequences),
        n_residues=len(residues),
        n_atoms=len(atoms),
        chain_ids=sorted(sequences),
        sequence_length_by_chain={cid: len(seq) for cid, seq in sequences.items()},
        chain_breaks=chain_breaks,
        numbering_gaps=numbering_gaps,
        nonstandard_residues=nonstandard,
        has_hydrogens=has_hydrogens,
        notes=notes,
    )


def concatenated_sequence(sequences: dict[str, str]) -> str:
    """Join per-chain sequences in chain order, for whole-structure comparisons."""
    return "".join(sequences[chain_id] for chain_id in sorted(sequences))
