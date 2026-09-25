"""Pocket detection (pipeline stage 2).

Pocketscribe does **not** implement a new cavity-detection algorithm. The default and
recommended backend is fpocket, run as an external binary and parsed here; fpocket's
alpha-sphere method and its trained druggability model are the science, and this
module is a wrapper around them.

A second backend, :class:`BuiltinGeometricBackend`, exists so that the bundled demo
and the test suite can run on a machine without fpocket installed. It is a plain
grid-based buriedness scan and it is labelled as such everywhere it appears -- in the
CLI output, in the report, and in the JSON handed to the narrative layer. Its
"druggability" number is an explicitly-documented heuristic, never presented as
fpocket's validated druggability score. Do not use it for scientific interpretation.
"""

from __future__ import annotations

import math
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree

from .errors import PocketDetectionError
from .models import Pocket, PocketResidue, PocketSet, ScoreProvenance
from .parsing import ParsedStructure

#: van der Waals radii (Angstrom) for the elements that occur in protein models.
VDW_RADII: dict[str, float] = {
    "C": 1.70, "N": 1.55, "O": 1.52, "S": 1.80, "P": 1.80,
    "SE": 1.90, "H": 1.20, "F": 1.47, "CL": 1.75, "BR": 1.85, "I": 1.98,
}
DEFAULT_VDW = 1.70

#: Elements counted as apolar when scoring pocket hydrophobicity in the fallback backend.
APOLAR_ELEMENTS = {"C", "S", "SE"}


# --------------------------------------------------------------------------------------
# fpocket backend
# --------------------------------------------------------------------------------------


def fpocket_available(binary: str = "fpocket") -> bool:
    """True when the fpocket executable is on PATH."""
    return shutil.which(binary) is not None


def fpocket_version(binary: str = "fpocket") -> str:
    """Best-effort fpocket version string, recorded in the methods appendix."""
    try:
        completed = subprocess.run(
            [binary, "--version"], capture_output=True, text=True, timeout=30, check=False
        )
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    text = (completed.stdout + completed.stderr).strip()
    match = re.search(r"\d+\.\d+(\.\d+)?", text)
    return match.group(0) if match else (text.splitlines()[0][:60] if text else "unknown")


class FpocketBackend:
    """Runs fpocket on a structure and parses its output directory.

    fpocket writes ``<stem>_out/`` containing ``<stem>_info.txt`` (per-pocket
    descriptors) and ``pockets/pocket*_atm.pdb`` (the atoms lining each pocket). Both
    are parsed here; nothing is recomputed that fpocket already reports.
    """

    name = "fpocket"
    is_fpocket = True

    def __init__(
        self,
        binary: str = "fpocket",
        min_alpha_spheres: int | None = None,
        extra_args: list[str] | None = None,
    ) -> None:
        self.binary = binary
        self.min_alpha_spheres = min_alpha_spheres
        self.extra_args = list(extra_args or [])

    def version(self) -> str:
        return fpocket_version(self.binary)

    def detect(self, structure: ParsedStructure) -> PocketSet:
        if not fpocket_available(self.binary):
            raise PocketDetectionError(
                f"fpocket executable '{self.binary}' was not found on PATH.\n"
                "  fpocket is an external dependency and is not installed by pip.\n"
                "    conda:  conda install -c conda-forge fpocket\n"
                "    Debian/Ubuntu:  sudo apt-get install fpocket\n"
                "    source: https://github.com/Discngine/fpocket\n"
                "  To try the pipeline without installing fpocket, re-run with "
                "--pocket-backend builtin, which uses a transparent grid-based "
                "fallback whose scores are NOT fpocket druggability scores and are not "
                "suitable for scientific interpretation."
            )

        with tempfile.TemporaryDirectory(prefix="pocketscribe-fpocket-") as tmp:
            workdir = Path(tmp)
            # fpocket writes its output next to the input file, so the input is copied
            # into a scratch directory to avoid littering the user's structure folder.
            staged = workdir / f"{Path(structure.path).stem}.pdb"
            _write_pdb(structure, staged, keep_hydrogens=True)

            command = [self.binary, "-f", str(staged)]
            if self.min_alpha_spheres is not None:
                command += ["-i", str(self.min_alpha_spheres)]
            command += self.extra_args

            try:
                completed = subprocess.run(
                    command, capture_output=True, text=True, timeout=1800, check=False
                )
            except subprocess.TimeoutExpired as exc:
                raise PocketDetectionError(
                    f"fpocket timed out after 30 minutes on '{Path(structure.path).name}'."
                ) from exc
            except OSError as exc:
                raise PocketDetectionError(f"Could not execute fpocket: {exc}") from exc

            out_dir = staged.with_name(f"{staged.stem}_out")
            if completed.returncode != 0 or not out_dir.is_dir():
                stderr = (completed.stderr or completed.stdout or "").strip()
                raise PocketDetectionError(
                    f"fpocket failed (exit code {completed.returncode}) on "
                    f"'{Path(structure.path).name}'."
                    + (f"\n  fpocket said: {stderr[:500]}" if stderr else "")
                )

            pocket_set = parse_fpocket_output(out_dir, structure)

        pocket_set.backend_version = self.version()
        pocket_set.structure_label = Path(structure.path).name
        return pocket_set


def parse_fpocket_output(out_dir: Path, structure: ParsedStructure | None = None) -> PocketSet:
    """Parse an ``*_out`` directory produced by fpocket.

    Split out from :class:`FpocketBackend` so that the parser can be unit tested
    against a recorded output directory without fpocket installed.
    """
    out_dir = Path(out_dir)
    info_files = sorted(out_dir.glob("*_info.txt"))
    if not info_files:
        raise PocketDetectionError(
            f"No fpocket '*_info.txt' file in '{out_dir}'. This does not look like an "
            "fpocket output directory."
        )

    descriptors = _parse_info_file(info_files[0])
    atm_files = _sorted_pocket_files(out_dir / "pockets")

    pockets: list[Pocket] = []
    warnings: list[str] = []

    for index, (pocket_number, fields) in enumerate(descriptors):
        residues: list[PocketResidue] = []
        centroid: tuple[float, float, float] | None = None

        # fpocket's info.txt numbers pockets from 1 while the per-pocket files have
        # been 0-based in some releases, so pockets are matched positionally (both
        # listings are in the same rank order) rather than by trusting the index.
        if index < len(atm_files):
            residues, centroid = _parse_pocket_atm(atm_files[index])
        elif structure is not None:
            warnings.append(
                f"No pocket{pocket_number} atom file found; residue list unavailable."
            )

        pockets.append(
            Pocket(
                id=pocket_number,
                rank=0,  # assigned after sorting
                volume=fields.get("volume"),
                druggability_score=fields.get("druggability_score"),
                score_provenance=ScoreProvenance.FPOCKET,
                fpocket_score=fields.get("score"),
                hydrophobicity_score=fields.get("hydrophobicity_score"),
                polarity_score=fields.get("polarity_score"),
                n_alpha_spheres=(
                    int(fields["number_of_alpha_spheres"])
                    if fields.get("number_of_alpha_spheres") is not None
                    else None
                ),
                mean_alpha_sphere_radius=fields.get("mean_alpha_sphere_radius"),
                total_sasa=fields.get("total_sasa"),
                centroid=centroid,
                residues=residues,
                extra={
                    key: value
                    for key, value in fields.items()
                    if key
                    not in {
                        "volume",
                        "druggability_score",
                        "score",
                        "hydrophobicity_score",
                        "polarity_score",
                        "number_of_alpha_spheres",
                        "mean_alpha_sphere_radius",
                        "total_sasa",
                    }
                    and value is not None
                },
            )
        )

    pockets = rank_pockets(pockets)
    return PocketSet(
        backend="fpocket",
        backend_is_fpocket=True,
        pockets=pockets,
        warnings=warnings,
    )


#: Maps the descriptor labels in fpocket's info.txt to snake_case field names.
_INFO_KEY_MAP = {
    "score": "score",
    "druggability score": "druggability_score",
    "number of alpha spheres": "number_of_alpha_spheres",
    "total sasa": "total_sasa",
    "polar sasa": "polar_sasa",
    "apolar sasa": "apolar_sasa",
    "volume": "volume",
    "mean local hydrophobic density": "mean_local_hydrophobic_density",
    "mean alpha sphere radius": "mean_alpha_sphere_radius",
    "mean alp. sph. solvent access": "mean_alpha_sphere_solvent_access",
    "apolar alpha sphere proportion": "apolar_alpha_sphere_proportion",
    "hydrophobicity score": "hydrophobicity_score",
    "volume score": "volume_score",
    "polarity score": "polarity_score",
    "charge score": "charge_score",
    "proportion of polar atoms": "proportion_of_polar_atoms",
    "alpha sphere density": "alpha_sphere_density",
    "cent. of mass - alpha sphere max dist": "com_alpha_sphere_max_dist",
    "flexibility": "flexibility",
}

_POCKET_HEADER = re.compile(r"^\s*Pocket\s+(\d+)\s*:", re.IGNORECASE)


def _parse_info_file(path: Path) -> list[tuple[int, dict[str, float]]]:
    """Parse ``<stem>_info.txt`` into ``(pocket_number, descriptor_dict)`` pairs."""
    entries: list[tuple[int, dict[str, float]]] = []
    current_number: int | None = None
    current: dict[str, float] = {}

    for raw_line in path.read_text(errors="replace").splitlines():
        header = _POCKET_HEADER.match(raw_line)
        if header:
            if current_number is not None:
                entries.append((current_number, current))
            current_number = int(header.group(1))
            current = {}
            continue
        if current_number is None or ":" not in raw_line:
            continue
        label, _, value_text = raw_line.partition(":")
        key = _INFO_KEY_MAP.get(label.strip().lower().rstrip(":"))
        if key is None:
            continue
        try:
            current[key] = float(value_text.strip())
        except ValueError:
            continue

    if current_number is not None:
        entries.append((current_number, current))

    if not entries:
        raise PocketDetectionError(
            f"fpocket info file '{path.name}' contained no pocket records. fpocket may "
            "have found no cavities in this structure."
        )
    return entries


def _sorted_pocket_files(pockets_dir: Path) -> list[Path]:
    """Per-pocket atom files sorted by their embedded integer, not lexically."""
    if not pockets_dir.is_dir():
        return []

    def index_of(path: Path) -> int:
        match = re.search(r"pocket(\d+)_atm", path.name)
        return int(match.group(1)) if match else 10**9

    return sorted(pockets_dir.glob("pocket*_atm.pdb"), key=index_of)


def _parse_pocket_atm(
    path: Path,
) -> tuple[list[PocketResidue], tuple[float, float, float] | None]:
    """Read the lining residues and centroid from a ``pocketN_atm.pdb`` file."""
    seen: dict[tuple[str, int, str], PocketResidue] = {}
    coords: list[tuple[float, float, float]] = []

    for line in path.read_text(errors="replace").splitlines():
        if not line.startswith(("ATOM", "HETATM")):
            continue
        try:
            resname = line[17:20].strip().upper()
            chain = line[21].strip() or "A"
            resseq = int(line[22:26])
            icode = line[26]
            x, y, z = float(line[30:38]), float(line[38:46]), float(line[46:54])
        except (ValueError, IndexError):
            continue
        if resname == "STP":  # fpocket's alpha-sphere pseudo-atoms
            continue
        coords.append((x, y, z))
        key = (chain, resseq, icode)
        if key not in seen:
            seen[key] = PocketResidue(
                chain=chain, resseq=resseq, icode=icode, resname=resname
            )

    centroid = None
    if coords:
        array = np.array(coords, dtype=float)
        centroid = tuple(round(float(v), 3) for v in array.mean(axis=0))  # type: ignore[assignment]

    residues = sorted(seen.values(), key=lambda r: (r.chain, r.resseq, r.icode))
    return residues, centroid


# --------------------------------------------------------------------------------------
# Built-in fallback backend
# --------------------------------------------------------------------------------------


class BuiltinGeometricBackend:
    """Grid-based buriedness scan used when fpocket is unavailable.

    Algorithm (a standard, deliberately simple LIGSITE-style scan -- nothing here is
    presented as novel):

    1. Lay a cubic grid over the structure's bounding box plus padding.
    2. Mark grid points inside any atom's van der Waals radius plus a probe radius as
       occupied; the rest are free.
    3. For each free point, cast rays along 7 axes (3 Cartesian + 4 cubic diagonals)
       and count how many are "protein-solvent-protein" enclosed. That count, 0-7, is
       the point's buriedness.
    4. Keep points with buriedness at or above a threshold, cluster them by grid
       connectivity, and discard clusters below a minimum volume.
    5. Report volume, centroid, lining residues and a documented heuristic score.

    The heuristic score is a transparent weighted combination of pocket volume,
    buriedness and apolar-atom fraction -- see :meth:`_heuristic_druggability`. It is
    **not** fpocket's druggability score, which is a trained model validated against
    known druggable sites, and it is always tagged
    :attr:`~pocketscribe.models.ScoreProvenance.BUILTIN_HEURISTIC` so that no report,
    table or narrative can present it as equivalent.
    """

    name = "builtin"
    is_fpocket = False

    def __init__(
        self,
        spacing: float = 1.0,
        probe_radius: float = 1.4,
        buriedness_threshold: int = 5,
        min_volume: float = 100.0,
        lining_cutoff: float = 5.0,
        max_pockets: int = 20,
    ) -> None:
        self.spacing = spacing
        self.probe_radius = probe_radius
        self.buriedness_threshold = buriedness_threshold
        self.min_volume = min_volume
        self.lining_cutoff = lining_cutoff
        self.max_pockets = max_pockets

    def version(self) -> str:
        return "builtin-1.0"

    def detect(self, structure: ParsedStructure) -> PocketSet:
        heavy_atoms = [
            atom for atom in structure.atoms if (atom["element"] or "").upper() != "H"
        ]
        if len(heavy_atoms) < 20:
            raise PocketDetectionError(
                "Structure has too few heavy atoms for cavity detection "
                f"({len(heavy_atoms)})."
            )

        coords = np.array([atom["coord"] for atom in heavy_atoms], dtype=float)
        radii = np.array(
            [VDW_RADII.get((atom["element"] or "").upper(), DEFAULT_VDW) for atom in heavy_atoms]
        )

        occupied, origin, shape = self._occupancy_grid(coords, radii)
        buriedness = self._buriedness(occupied)
        pocket_mask = (~occupied) & (buriedness >= self.buriedness_threshold)

        if not pocket_mask.any():
            return PocketSet(
                backend=self.name,
                backend_version=self.version(),
                backend_is_fpocket=False,
                structure_label=Path(structure.path).name,
                pockets=[],
                warnings=["No buried cavities passed the built-in backend's thresholds."],
            )

        labels, n_labels = ndimage.label(pocket_mask)
        voxel_volume = self.spacing**3
        tree = cKDTree(coords)

        candidates: list[Pocket] = []
        for label_index in range(1, n_labels + 1):
            voxels = np.argwhere(labels == label_index)
            volume = len(voxels) * voxel_volume
            if volume < self.min_volume:
                continue

            points = origin + voxels * self.spacing
            centroid = points.mean(axis=0)
            mean_buriedness = float(
                buriedness[voxels[:, 0], voxels[:, 1], voxels[:, 2]].mean()
            )

            neighbour_sets = tree.query_ball_point(points, r=self.lining_cutoff)
            atom_indices = sorted({idx for group in neighbour_sets for idx in group})
            if not atom_indices:
                continue

            residues: dict[tuple[str, int, str], PocketResidue] = {}
            apolar = 0
            for idx in atom_indices:
                atom = heavy_atoms[idx]
                if (atom["element"] or "").upper() in APOLAR_ELEMENTS:
                    apolar += 1
                key = (atom["chain"], atom["resseq"], atom["icode"])
                if key not in residues:
                    residues[key] = PocketResidue(
                        chain=atom["chain"],
                        resseq=atom["resseq"],
                        icode=atom["icode"],
                        resname=atom["resname"],
                    )
            apolar_fraction = apolar / len(atom_indices)

            candidates.append(
                Pocket(
                    id=0,
                    rank=0,
                    volume=round(volume, 1),
                    druggability_score=self._heuristic_druggability(
                        volume=volume,
                        buriedness=mean_buriedness,
                        apolar_fraction=apolar_fraction,
                    ),
                    score_provenance=ScoreProvenance.BUILTIN_HEURISTIC,
                    hydrophobicity_score=round(apolar_fraction * 100.0, 1),
                    polarity_score=round((1.0 - apolar_fraction) * 100.0, 1),
                    n_alpha_spheres=None,
                    centroid=tuple(round(float(v), 3) for v in centroid),  # type: ignore[arg-type]
                    residues=sorted(residues.values(), key=lambda r: (r.chain, r.resseq)),
                    extra={
                        "mean_buriedness_0_7": round(mean_buriedness, 2),
                        "n_voxels": float(len(voxels)),
                        "apolar_atom_fraction": round(apolar_fraction, 3),
                    },
                )
            )

        candidates.sort(key=lambda p: (p.druggability_score or 0.0, p.volume or 0.0), reverse=True)
        candidates = candidates[: self.max_pockets]
        for position, pocket in enumerate(candidates, start=1):
            pocket.id = position
        pockets = rank_pockets(candidates)

        return PocketSet(
            backend=self.name,
            backend_version=self.version(),
            backend_is_fpocket=False,
            structure_label=Path(structure.path).name,
            pockets=pockets,
            warnings=[
                "Pockets were detected with Pocketscribe's built-in geometric fallback "
                "backend, not with fpocket. The scores in the 'druggability' column are "
                "a transparent volume/buriedness/hydrophobicity heuristic, NOT fpocket's "
                "validated druggability score, and the two are not comparable. Install "
                "fpocket and re-run before drawing any scientific conclusion."
            ],
        )

    def _occupancy_grid(
        self, coords: np.ndarray, radii: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, tuple[int, int, int]]:
        """Boolean grid marking points inside atom vdW radii plus the probe radius."""
        padding = 4.0
        lower = coords.min(axis=0) - padding
        upper = coords.max(axis=0) + padding
        shape = tuple(int(math.ceil(v)) + 1 for v in (upper - lower) / self.spacing)

        occupied = np.zeros(shape, dtype=bool)
        max_radius = float(radii.max()) + self.probe_radius
        span = int(math.ceil(max_radius / self.spacing))

        indices = np.round((coords - lower) / self.spacing).astype(int)
        offsets = np.stack(
            np.meshgrid(*(np.arange(-span, span + 1),) * 3, indexing="ij"), axis=-1
        ).reshape(-1, 3)
        offset_distances = np.linalg.norm(offsets * self.spacing, axis=1)

        for centre, radius in zip(indices, radii, strict=True):
            cutoff = radius + self.probe_radius
            within = offsets[offset_distances <= cutoff]
            points = within + centre
            valid = np.all((points >= 0) & (points < np.array(shape)), axis=1)
            points = points[valid]
            occupied[points[:, 0], points[:, 1], points[:, 2]] = True

        return occupied, lower, shape

    @staticmethod
    def _buriedness(occupied: np.ndarray) -> np.ndarray:
        """Per-point enclosure count along 7 axes (LIGSITE's protein-solvent-protein scan)."""
        directions = [
            (1, 0, 0), (0, 1, 0), (0, 0, 1),
            (1, 1, 1), (1, 1, -1), (1, -1, 1), (-1, 1, 1),
        ]
        counts = np.zeros(occupied.shape, dtype=np.int8)
        for direction in directions:
            forward = _protein_in_direction(occupied, direction)
            backward = _protein_in_direction(occupied, tuple(-d for d in direction))
            counts += (forward & backward).astype(np.int8)
        return counts

    @staticmethod
    def _heuristic_druggability(
        volume: float, buriedness: float, apolar_fraction: float
    ) -> float:
        """Transparent 0-1 heuristic. Documented in full because it is not a trained model.

        ``0.40 * volume_term + 0.30 * buriedness_term + 0.30 * apolar_term`` where

        * ``volume_term`` is a logistic centred on 300 A^3 with a 150 A^3 width, so
          cavities far below a small-molecule-sized volume score near zero and very
          large cavities saturate rather than dominating;
        * ``buriedness_term`` is the mean 0-7 enclosure count rescaled to 0-1;
        * ``apolar_term`` is the fraction of lining heavy atoms that are carbon,
          sulphur or selenium.

        The weights are a reasonable ordering of what makes a cavity ligandable
        (adequate enclosed volume first, then burial, then hydrophobic character). They
        are not fitted to any benchmark and carry no validation.
        """
        volume_term = 1.0 / (1.0 + math.exp(-(volume - 300.0) / 150.0))
        buriedness_term = min(max(buriedness / 7.0, 0.0), 1.0)
        apolar_term = min(max(apolar_fraction, 0.0), 1.0)
        score = 0.40 * volume_term + 0.30 * buriedness_term + 0.30 * apolar_term
        return round(score, 3)


def _protein_in_direction(occupied: np.ndarray, direction: tuple[int, int, int]) -> np.ndarray:
    """True where an occupied point lies somewhere along ``direction`` from each point.

    Implemented as a cumulative scan over shifted slices, which is O(n) in grid points
    per direction rather than the O(n * ray length) of casting rays one at a time.
    """
    result = np.zeros(occupied.shape, dtype=bool)
    shifted = occupied.copy()
    max_steps = max(occupied.shape)
    for _ in range(max_steps):
        shifted = _shift(shifted, direction)
        if not shifted.any():
            break
        result |= shifted
    return result


def _shift(array: np.ndarray, direction: tuple[int, int, int]) -> np.ndarray:
    """Shift a boolean array by one grid step, filling the exposed face with False."""
    result = array
    for axis, step in enumerate(direction):
        if step == 0:
            continue
        result = np.roll(result, step, axis=axis)
        index = [slice(None)] * 3
        index[axis] = 0 if step > 0 else -1
        result[tuple(index)] = False
    return result


# --------------------------------------------------------------------------------------
# Shared helpers
# --------------------------------------------------------------------------------------


def rank_pockets(pockets: list[Pocket]) -> list[Pocket]:
    """Sort by druggability (descending, volume as tie-break) and assign 1-based ranks."""
    ordered = sorted(
        pockets,
        key=lambda p: (
            p.druggability_score if p.druggability_score is not None else -1.0,
            p.volume or 0.0,
        ),
        reverse=True,
    )
    for rank, pocket in enumerate(ordered, start=1):
        pocket.rank = rank
    return ordered


def get_backend(name: str, **kwargs):
    """Construct a pocket-detection backend by name ('fpocket' or 'builtin')."""
    key = name.strip().lower()
    if key == "fpocket":
        return FpocketBackend(**kwargs)
    if key == "builtin":
        return BuiltinGeometricBackend(**kwargs)
    raise PocketDetectionError(
        f"Unknown pocket backend '{name}'. Available: fpocket (default, recommended), "
        "builtin (offline fallback, heuristic scores only)."
    )


def resolve_backend(
    requested: str,
    allow_fallback: bool = False,
    fpocket_binary: str = "fpocket",
    min_alpha_spheres: int | None = None,
):
    """Pick a backend, optionally degrading to the built-in one when fpocket is absent.

    Returns ``(backend, fell_back)``. ``allow_fallback`` is only ever set by
    ``pocketscribe demo`` and the test suite, so that a normal analysis run never
    silently substitutes heuristic scores for fpocket's: without it, a missing fpocket
    raises with install instructions instead.
    """
    if requested == "builtin":
        return BuiltinGeometricBackend(), False

    fpocket_backend = FpocketBackend(
        binary=fpocket_binary, min_alpha_spheres=min_alpha_spheres
    )
    if fpocket_available(fpocket_binary):
        return fpocket_backend, False
    if allow_fallback:
        return BuiltinGeometricBackend(), True
    # Returned rather than raised here so the caller controls where the error surfaces;
    # FpocketBackend.detect() raises with full install instructions.
    return fpocket_backend, False


def _write_pdb(structure: ParsedStructure, destination: Path, keep_hydrogens: bool = True) -> None:
    """Write the parsed protein back out as a plain PDB file for an external tool."""
    lines: list[str] = []
    serial = 1
    for atom in structure.atoms:
        element = (atom["element"] or "").upper()
        if not keep_hydrogens and element == "H":
            continue
        name = atom["name"]
        # PDB atom-name alignment: names of fewer than 4 characters start in column 14.
        formatted_name = f"{name:<4}" if len(name) >= 4 else f" {name:<3}"
        x, y, z = atom["coord"]
        lines.append(
            f"ATOM  {serial:>5} {formatted_name}{atom['resname']:>4}"
            f" {atom['chain'][:1]}{atom['resseq']:>4}{atom['icode']}   "
            f"{x:>8.3f}{y:>8.3f}{z:>8.3f}"
            f"{atom['occupancy']:>6.2f}{atom['bfactor']:>6.2f}"
            f"          {element:>2}"
        )
        serial += 1
    lines.append("END")
    destination.write_text("\n".join(lines) + "\n")
