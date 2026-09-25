"""Shared test fixtures.

Every test runs offline. The structures analysed are the bundled synthetic ones, and
fpocket parsing is tested against a checked-in format sample, so the suite passes in CI
without fpocket, without GROMACS and without network access.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from pocketscribe.data import EXAMPLE_DIR
from pocketscribe.parsing import parse_structure

FIXTURE_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture(scope="session")
def example_dir() -> Path:
    return EXAMPLE_DIR


@pytest.fixture(scope="session")
def fpocket_out_dir() -> Path:
    return FIXTURE_DIR / "fpocket_out"


@pytest.fixture(scope="session")
def af2_path(example_dir: Path) -> Path:
    return example_dir / "AF-SYNTH01-F1-model_v4.pdb"


@pytest.fixture(scope="session")
def openfold_path(example_dir: Path) -> Path:
    return example_dir / "synth01_openfold_model_1.pdb"


@pytest.fixture(scope="session")
def colabfold_path(example_dir: Path) -> Path:
    return example_dir / "synth01_unrelaxed_rank_001_alphafold2_model_3_seed_000.pdb"


@pytest.fixture(scope="session")
def esmfold_path(example_dir: Path) -> Path:
    return example_dir / "synth01_esmfold.pdb"


@pytest.fixture(scope="session")
def boltz_path(example_dir: Path) -> Path:
    return example_dir / "boltz_synth01_model_0.pdb"


@pytest.fixture(scope="session")
def mmcif_path(example_dir: Path) -> Path:
    return example_dir / "AF-SYNTH01-F1-model_v4.cif"


@pytest.fixture(scope="session")
def different_protein_path(example_dir: Path) -> Path:
    return example_dir / "AF-SYNTH02-F1-model_v4.pdb"


@pytest.fixture(scope="session")
def experimental_path(example_dir: Path) -> Path:
    return example_dir / "experimental_like.pdb"


@pytest.fixture(scope="session")
def af2_structure(af2_path: Path):
    return parse_structure(af2_path)
