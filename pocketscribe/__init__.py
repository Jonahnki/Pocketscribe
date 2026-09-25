"""Pocketscribe: confidence-aware binding-pocket triage for predicted protein structures.

Takes a predicted structure from AlphaFold2, OpenFold, ColabFold, ESMFold or Boltz and
produces one self-contained report covering structure QC, pocket detection,
confidence-caveated druggability, optional cross-model consensus, and ready-to-run
GROMACS input files.

Pocketscribe is an integration contribution. It does not implement a new
pocket-detection algorithm, druggability model or force field -- fpocket, GROMACS,
CHARMM36m and the prediction models do that work, and each is cited in
:mod:`pocketscribe.citations`. The one piece of original method here is the cross-model
consensus module in :mod:`pocketscribe.consensus`.

Research use only. See the disclaimer in every generated report.
"""

from __future__ import annotations

__version__ = "0.1.0"
__author__ = "John Adeyemo Adedeji"
__license__ = "MIT"

__all__ = ["__version__", "__author__", "__license__"]
