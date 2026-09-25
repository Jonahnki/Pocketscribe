"""Structure parsing, validation and QC."""

from __future__ import annotations

import pytest

from pocketscribe.errors import InputStructureError
from pocketscribe.parsing import CA_BREAK_THRESHOLD, parse_structure


def test_parses_pdb(af2_structure):
    assert af2_structure.file_format == "pdb"
    assert af2_structure.qc.n_residues == 192
    assert af2_structure.qc.n_chains == 1
    assert af2_structure.qc.chain_ids == ["A"]
    assert af2_structure.qc.n_atoms == 192 * 5


def test_parses_mmcif_to_the_same_model(mmcif_path, af2_structure):
    """The mmCIF copy of a model must parse to the same content as the PDB."""
    parsed = parse_structure(mmcif_path)
    assert parsed.file_format == "mmcif"
    assert parsed.qc.n_residues == af2_structure.qc.n_residues
    assert parsed.sequences == af2_structure.sequences


def test_extracts_one_letter_sequence(af2_structure):
    sequence = af2_structure.sequences["A"]
    assert len(sequence) == 192
    assert set(sequence) <= set("ACDEFGHIKLMNPQRSTVWY")


def test_backbone_is_continuous(af2_structure):
    """The synthetic fold is built with real Ca-Ca spacing, so QC should see no breaks."""
    assert af2_structure.qc.chain_breaks == []


def test_ca_coords_align_with_residue_keys(af2_structure):
    keys, coords = af2_structure.ca_coords()
    assert len(keys) == len(coords) == af2_structure.qc.n_residues
    assert keys[0] == ("A", 1, " ")


def test_chain_break_detection_threshold_is_sane():
    """A guard against someone loosening the threshold past a real Ca-Ca distance."""
    assert 3.8 < CA_BREAK_THRESHOLD < 6.0


def test_missing_file_raises_clearly(tmp_path):
    with pytest.raises(InputStructureError, match="not found"):
        parse_structure(tmp_path / "nope.pdb")


def test_empty_file_raises_clearly(tmp_path):
    path = tmp_path / "empty.pdb"
    path.write_text("")
    with pytest.raises(InputStructureError, match="empty"):
        parse_structure(path)


def test_directory_raises_clearly(tmp_path):
    with pytest.raises(InputStructureError, match="directory"):
        parse_structure(tmp_path)


def test_fasta_is_rejected_with_a_useful_message(tmp_path):
    """A sequence is the most likely wrong input; the error must say what to do."""
    path = tmp_path / "seq.pdb"
    path.write_text(">sp|P69905|HBA_HUMAN\nMVLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSFPTTKTYFPHF\n")
    with pytest.raises(InputStructureError):
        parse_structure(path)


def test_nucleic_acid_model_is_rejected(tmp_path):
    """Pocketscribe analyses proteins; a DNA model must fail rather than yield nothing."""
    lines = ["HEADER    DNA"]
    for index in range(12):
        lines.append(
            f"ATOM  {index + 1:>5}  P    DA A{index + 1:>4}    "
            f"{index * 3.0:>8.3f}{0.0:>8.3f}{0.0:>8.3f}  1.00 20.00           P"
        )
    lines.append("END")
    path = tmp_path / "dna.pdb"
    path.write_text("\n".join(lines) + "\n")
    with pytest.raises(InputStructureError, match="protein"):
        parse_structure(path)


def test_qc_reports_numbering_gaps(tmp_path, af2_path):
    """Residues missing from the model must be surfaced, not silently bridged."""
    kept = []
    for line in af2_path.read_text().splitlines():
        if line.startswith("ATOM"):
            resseq = int(line[22:26])
            if 40 <= resseq <= 50:
                continue
        kept.append(line)
    path = tmp_path / "gapped.pdb"
    path.write_text("\n".join(kept) + "\n")

    parsed = parse_structure(path)
    assert parsed.qc.numbering_gaps
    assert any("numbering" in note for note in parsed.qc.notes)
