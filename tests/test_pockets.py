"""Pocket detection: fpocket output parsing and the offline fallback backend."""

from __future__ import annotations

import pytest

from pocketscribe.errors import PocketDetectionError
from pocketscribe.models import ScoreProvenance
from pocketscribe.pockets import (
    BuiltinGeometricBackend,
    FpocketBackend,
    get_backend,
    parse_fpocket_output,
    rank_pockets,
    resolve_backend,
)

# --------------------------------------------------------------------------------------
# fpocket output parsing (tested against a checked-in format sample)
# --------------------------------------------------------------------------------------


def test_parses_all_pockets(fpocket_out_dir):
    pocket_set = parse_fpocket_output(fpocket_out_dir)
    assert pocket_set.backend == "fpocket"
    assert pocket_set.backend_is_fpocket is True
    assert len(pocket_set.pockets) == 3


def test_parses_fpocket_descriptors(fpocket_out_dir):
    pocket_set = parse_fpocket_output(fpocket_out_dir)
    top = pocket_set.pockets[0]
    assert top.druggability_score == pytest.approx(0.871)
    assert top.volume == pytest.approx(1043.782)
    assert top.fpocket_score == pytest.approx(0.412)
    assert top.n_alpha_spheres == 112
    assert top.hydrophobicity_score == pytest.approx(38.25)
    assert top.polarity_score == pytest.approx(9.0)
    assert top.total_sasa == pytest.approx(241.395)


def test_keeps_secondary_descriptors(fpocket_out_dir):
    """Descriptors Pocketscribe does not surface are still parsed, for the JSON payload."""
    top = parse_fpocket_output(fpocket_out_dir).pockets[0]
    assert top.extra["apolar_sasa"] == pytest.approx(163.283)
    assert top.extra["flexibility"] == pytest.approx(0.231)


def test_scores_are_tagged_as_coming_from_fpocket(fpocket_out_dir):
    """Provenance is what stops a heuristic score being read as fpocket's."""
    for pocket in parse_fpocket_output(fpocket_out_dir).pockets:
        assert pocket.score_provenance is ScoreProvenance.FPOCKET


def test_parses_lining_residues_and_skips_alpha_spheres(fpocket_out_dir):
    top = parse_fpocket_output(fpocket_out_dir).pockets[0]
    labels = {residue.label for residue in top.residues}
    assert labels == {"A:LEU12", "A:VAL15", "A:PHE19", "A:ALA23"}
    # STP records are fpocket's alpha-sphere pseudo-atoms, not protein residues.
    assert all(residue.resname != "STP" for residue in top.residues)


def test_computes_a_centroid_from_lining_atoms(fpocket_out_dir):
    top = parse_fpocket_output(fpocket_out_dir).pockets[0]
    assert top.centroid is not None
    assert all(isinstance(value, float) for value in top.centroid)


def test_pockets_are_ranked_by_druggability(fpocket_out_dir):
    pockets = parse_fpocket_output(fpocket_out_dir).pockets
    assert [pocket.rank for pocket in pockets] == [1, 2, 3]
    scores = [pocket.druggability_score for pocket in pockets]
    assert scores == sorted(scores, reverse=True)


def test_missing_info_file_raises(tmp_path):
    with pytest.raises(PocketDetectionError, match="info.txt"):
        parse_fpocket_output(tmp_path)


def test_empty_info_file_raises(tmp_path):
    (tmp_path / "x_info.txt").write_text("")
    with pytest.raises(PocketDetectionError, match="no pocket records"):
        parse_fpocket_output(tmp_path)


def test_ranking_is_stable_for_equal_scores():
    from pocketscribe.models import Pocket

    pockets = [
        Pocket(id=1, rank=0, volume=100.0, druggability_score=0.5),
        Pocket(id=2, rank=0, volume=900.0, druggability_score=0.5),
    ]
    ranked = rank_pockets(pockets)
    # Volume breaks the tie, so the larger cavity ranks first.
    assert ranked[0].id == 2


# --------------------------------------------------------------------------------------
# Built-in fallback backend
# --------------------------------------------------------------------------------------


def test_builtin_finds_the_known_cavities(af2_structure):
    """The synthetic fold has two domains of known packing; both cavities must be found."""
    pocket_set = BuiltinGeometricBackend().detect(af2_structure)
    assert len(pocket_set.pockets) == 2
    volumes = sorted((pocket.volume for pocket in pocket_set.pockets), reverse=True)
    assert volumes[0] > 500.0
    assert 100.0 < volumes[1] < 500.0


def test_builtin_scores_are_never_labelled_as_fpocket(af2_structure):
    """The single most important guard in this module."""
    pocket_set = BuiltinGeometricBackend().detect(af2_structure)
    assert pocket_set.backend_is_fpocket is False
    for pocket in pocket_set.pockets:
        assert pocket.score_provenance is ScoreProvenance.BUILTIN_HEURISTIC


def test_builtin_warns_loudly_in_its_output(af2_structure):
    pocket_set = BuiltinGeometricBackend().detect(af2_structure)
    assert pocket_set.warnings
    joined = " ".join(pocket_set.warnings)
    assert "not with fpocket" in joined or "NOT fpocket" in joined


def test_builtin_is_deterministic(af2_structure):
    """A report that changes between identical runs is not reproducible."""
    first = BuiltinGeometricBackend().detect(af2_structure)
    second = BuiltinGeometricBackend().detect(af2_structure)
    assert [pocket.volume for pocket in first.pockets] == [
        pocket.volume for pocket in second.pockets
    ]
    assert [pocket.druggability_score for pocket in first.pockets] == [
        pocket.druggability_score for pocket in second.pockets
    ]


def test_builtin_heuristic_is_bounded():
    score = BuiltinGeometricBackend._heuristic_druggability(
        volume=10_000.0, buriedness=7.0, apolar_fraction=1.0
    )
    assert 0.0 <= score <= 1.0
    worst = BuiltinGeometricBackend._heuristic_druggability(
        volume=1.0, buriedness=0.0, apolar_fraction=0.0
    )
    assert 0.0 <= worst < score


def test_builtin_rejects_a_tiny_structure(af2_structure):
    import copy

    tiny = copy.copy(af2_structure)
    tiny.atoms = af2_structure.atoms[:5]
    with pytest.raises(PocketDetectionError, match="too few heavy atoms"):
        BuiltinGeometricBackend().detect(tiny)


# --------------------------------------------------------------------------------------
# Backend selection
# --------------------------------------------------------------------------------------


def test_unknown_backend_name_is_rejected():
    with pytest.raises(PocketDetectionError, match="Unknown pocket backend"):
        get_backend("autodock")


def test_explicit_builtin_is_honoured():
    backend, fell_back = resolve_backend("builtin")
    assert isinstance(backend, BuiltinGeometricBackend)
    assert fell_back is False


def test_analysis_runs_never_fall_back_silently(monkeypatch):
    """Without allow_fallback, a missing fpocket must surface as an error, not a swap."""
    monkeypatch.setattr("pocketscribe.pockets.fpocket_available", lambda *_: False)
    backend, fell_back = resolve_backend("fpocket", allow_fallback=False)
    assert isinstance(backend, FpocketBackend)
    assert fell_back is False


def test_demo_may_fall_back(monkeypatch):
    monkeypatch.setattr("pocketscribe.pockets.fpocket_available", lambda *_: False)
    backend, fell_back = resolve_backend("fpocket", allow_fallback=True)
    assert isinstance(backend, BuiltinGeometricBackend)
    assert fell_back is True


def test_missing_fpocket_error_explains_how_to_install(monkeypatch, af2_structure):
    monkeypatch.setattr("pocketscribe.pockets.fpocket_available", lambda *_: False)
    with pytest.raises(PocketDetectionError) as excinfo:
        FpocketBackend().detect(af2_structure)
    message = str(excinfo.value)
    assert "conda" in message
    assert "--pocket-backend builtin" in message
