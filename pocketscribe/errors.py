"""Exception hierarchy for Pocketscribe.

Every failure mode that a user can plausibly trigger has its own exception type
with a specific, actionable message. The guiding rule throughout the codebase is
"never silently mis-parse": when confidence data, a structure source, or a
cross-model comparison cannot be established with certainty, raise rather than
guess.
"""

from __future__ import annotations


class PocketscribeError(Exception):
    """Base class for all Pocketscribe errors."""


class InputStructureError(PocketscribeError):
    """The supplied file is missing, unreadable, or not a protein model."""


class SourceDetectionError(PocketscribeError):
    """The structure-prediction source could not be determined from the file."""


class UnsupportedSourceError(PocketscribeError):
    """A known source whose adapter is declared but not yet implemented (Tier 2 stub).

    Raised instead of falling back to a different adapter, because a Tier 2 source
    generally does not follow the legacy B-factor pLDDT convention and a wrong parse
    would produce confident-looking nonsense.
    """


class ConfidenceExtractionError(PocketscribeError):
    """Per-residue confidence could not be found or parsed for the declared source."""


class PocketDetectionError(PocketscribeError):
    """Pocket detection failed, or the pocket-detection backend is unavailable."""


class ConsensusError(PocketscribeError):
    """Cross-model consensus could not be computed for the supplied structures."""


class SequenceMismatchError(ConsensusError):
    """Structures supplied for consensus are not the same protein.

    Comparing pockets across different proteins would produce a plausible-looking
    but meaningless table, so this is a hard failure rather than a warning.
    """


class MDSetupError(PocketscribeError):
    """MD input generation failed."""


class ReportRenderError(PocketscribeError):
    """The HTML report could not be rendered."""


class ConfigError(PocketscribeError):
    """A configuration file or CLI option combination is invalid."""
