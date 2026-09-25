"""Run configuration.

Every tunable parameter lives here as a validated pydantic model, can be supplied in a
YAML file, and is echoed into the report's methods appendix. A run that cannot be
described precisely enough to reproduce is not much use in a methods section.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, ValidationError, field_validator

from .errors import ConfigError


class PocketConfig(BaseModel):
    """Pocket-detection settings."""

    backend: str = Field(
        default="fpocket",
        description="'fpocket' (default, recommended), 'builtin' (offline fallback), or 'auto'.",
    )
    fpocket_binary: str = "fpocket"
    min_alpha_spheres: int | None = Field(
        default=None, description="fpocket -i; leave unset for fpocket's own default."
    )
    top_pockets: int = Field(default=3, ge=1, le=50)

    @field_validator("backend")
    @classmethod
    def _known_backend(cls, value: str) -> str:
        if value not in {"fpocket", "builtin", "auto"}:
            raise ValueError(
                f"backend must be 'fpocket', 'builtin' or 'auto', not '{value}'"
            )
        return value


class ConsensusConfig(BaseModel):
    """Cross-model consensus thresholds."""

    min_sequence_identity: float = Field(default=0.95, ge=0.0, le=1.0)
    centroid_tolerance: float = Field(default=8.0, gt=0.0)
    min_residue_jaccard: float = Field(default=0.25, ge=0.0, le=1.0)
    confidence_weighting: bool = True


class MDConfig(BaseModel):
    """Molecular dynamics setup defaults."""

    enabled: bool = False
    force_field: str = "charmm36m"
    water_model: str = "tip3p"
    box_shape: str = "dodecahedron"
    box_padding_nm: float = Field(default=1.0, gt=0.0)
    salt_concentration_M: float = Field(default=0.15, ge=0.0)
    temperature_K: float = Field(default=310.0, gt=0.0)
    production_ns: float = Field(default=100.0, gt=0.0)
    target_pocket_rank: int | None = Field(
        default=None, description="Defaults to the top-ranked pocket."
    )


class NarrativeConfig(BaseModel):
    """Optional narrative synthesis settings."""

    enabled: bool = True
    model: str = "claude-sonnet-4-5"


class RunConfig(BaseModel):
    """Top-level configuration for one Pocketscribe run."""

    pockets: PocketConfig = Field(default_factory=PocketConfig)
    consensus: ConsensusConfig = Field(default_factory=ConsensusConfig)
    md: MDConfig = Field(default_factory=MDConfig)
    narrative: NarrativeConfig = Field(default_factory=NarrativeConfig)

    @classmethod
    def from_yaml(cls, path: str | Path) -> RunConfig:
        """Load configuration from a YAML file, failing loudly on anything invalid."""
        path = Path(path)
        if not path.is_file():
            raise ConfigError(f"Config file not found: {path}")
        try:
            data: Any = yaml.safe_load(path.read_text()) or {}
        except yaml.YAMLError as exc:
            raise ConfigError(f"Config file '{path}' is not valid YAML: {exc}") from exc
        if not isinstance(data, dict):
            raise ConfigError(
                f"Config file '{path}' must contain a mapping at the top level."
            )
        try:
            return cls.model_validate(data)
        except ValidationError as exc:
            raise ConfigError(f"Invalid configuration in '{path}':\n{exc}") from exc

    def as_parameters(self) -> dict[str, Any]:
        """Flattened parameter record for the report's methods appendix."""
        return {
            "pocket backend": self.pockets.backend,
            "top pockets reported": self.pockets.top_pockets,
            "consensus min identity": self.consensus.min_sequence_identity,
            "consensus centroid tolerance (A)": self.consensus.centroid_tolerance,
            "consensus min residue Jaccard": self.consensus.min_residue_jaccard,
            "consensus confidence weighting": self.consensus.confidence_weighting,
            "MD setup generated": self.md.enabled,
        }
