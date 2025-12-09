"""
Temporal metadata models for historical newspaper analysis.

Pydantic models for era definitions and presets used in temporal classification.
"""

from typing import Optional

from pydantic import BaseModel, Field, field_validator, model_validator


class EraRange(BaseModel):
    """
    Year range for an era definition.

    Use None for open-ended ranges:
    - start=None means "from the beginning of time"
    - end=None means "until the end of time"
    """

    start: Optional[int] = Field(
        default=None,
        description="Start year (inclusive). None = open-ended start.",
        ge=1800,
        le=2100,
    )
    end: Optional[int] = Field(
        default=None,
        description="End year (inclusive). None = open-ended end.",
        ge=1800,
        le=2100,
    )

    @model_validator(mode="after")
    def validate_range(self) -> "EraRange":
        """Ensure start <= end when both are provided."""
        if self.start is not None and self.end is not None and self.start > self.end:
            msg = f"Start year ({self.start}) must be <= end year ({self.end})"
            raise ValueError(msg)
        return self

    def contains(self, year: int) -> bool:
        """Check if a year falls within this range."""
        start_ok = self.start is None or year >= self.start
        end_ok = self.end is None or year <= self.end
        return start_ok and end_ok

    def to_tuple(self) -> tuple[Optional[int], Optional[int]]:
        """Convert to tuple format for backwards compatibility."""
        return (self.start, self.end)


class EraDefinition(BaseModel):
    """
    Definition of a single era within a preset.

    Maps an era name to its year range.
    """

    name: str = Field(description="Era identifier (e.g., 'pre_war', 'war', 'post_war')")
    range: EraRange = Field(description="Year range for this era")
    description: Optional[str] = Field(default=None, description="Optional description of this era")

    @classmethod
    def from_tuple(
        cls, name: str, year_range: tuple[Optional[int], Optional[int]]
    ) -> "EraDefinition":
        """Create from legacy tuple format."""
        return cls(name=name, range=EraRange(start=year_range[0], end=year_range[1]))


class EraPreset(BaseModel):
    """
    Complete era preset configuration.

    Defines a named set of eras for temporal classification.
    """

    key: str = Field(description="Preset identifier (e.g., 'wwi', 'war_phases')")
    name: str = Field(description="Human-readable preset name")
    description: str = Field(description="Description of the periodization")
    eras: list[EraDefinition] = Field(description="List of era definitions", min_length=1)

    @field_validator("eras")
    @classmethod
    def validate_unique_era_names(cls, v: list[EraDefinition]) -> list[EraDefinition]:
        """Ensure all era names are unique within a preset."""
        names = [era.name for era in v]
        if len(names) != len(set(names)):
            duplicates = [name for name in names if names.count(name) > 1]
            msg = f"Duplicate era names found: {set(duplicates)}"
            raise ValueError(msg)
        return v

    def get_era_dict(self) -> dict[str, tuple[Optional[int], Optional[int]]]:
        """Convert to legacy dict format for backwards compatibility."""
        return {era.name: era.range.to_tuple() for era in self.eras}

    def get_era_names(self) -> list[str]:
        """Get list of era names in order."""
        return [era.name for era in self.eras]

    def classify_year(self, year: Optional[int]) -> str:
        """
        Classify a year into an era.

        Args:
            year: Year to classify (can be None)

        Returns:
            Era name, or empty string if year is None or doesn't match any era.
        """
        if year is None:
            return ""

        for era in self.eras:
            if era.range.contains(year):
                return era.name

        return ""


# =============================================================================
# Preset Instances
# =============================================================================

WWI_PRESET = EraPreset(
    key="wwi",
    name="World War I Periodization",
    description="Standard WWI periodization: pre-war, war years, post-war",
    eras=[
        EraDefinition(
            name="pre_war",
            range=EraRange(start=None, end=1913),
            description="Pre-war period until end of 1913",
        ),
        EraDefinition(
            name="war",
            range=EraRange(start=1914, end=1918),
            description="World War I years (1914-1918)",
        ),
        EraDefinition(
            name="post_war",
            range=EraRange(start=1919, end=None),
            description="Post-war period from 1919 onwards",
        ),
    ],
)

WWI_BINARY_PRESET = EraPreset(
    key="wwi_binary",
    name="WWI Binary Split",
    description="Simple before/after WWI start (pre/post July 1914)",
    eras=[
        EraDefinition(name="pre", range=EraRange(start=None, end=1913)),
        EraDefinition(name="post", range=EraRange(start=1914, end=None)),
    ],
)

MID_WAR_SPLIT_PRESET = EraPreset(
    key="mid_war_split",
    name="Mid-War Split",
    description="Binary split at 1915/1916 boundary (initial enthusiasm vs. attrition)",
    eras=[
        EraDefinition(name="pre", range=EraRange(start=None, end=1915)),
        EraDefinition(name="post", range=EraRange(start=1916, end=None)),
    ],
)

WAR_PHASES_PRESET = EraPreset(
    key="war_phases",
    name="War Phases",
    description="Detailed war phases: pre-war, early war (movement), late war (attrition), post-war",
    eras=[
        EraDefinition(
            name="pre_war",
            range=EraRange(start=None, end=1913),
            description="Pre-war period",
        ),
        EraDefinition(
            name="early_war",
            range=EraRange(start=1914, end=1915),
            description="War of movement, initial enthusiasm",
        ),
        EraDefinition(
            name="late_war",
            range=EraRange(start=1916, end=1918),
            description="Attrition warfare, disillusionment",
        ),
        EraDefinition(
            name="post_war",
            range=EraRange(start=1919, end=None),
            description="Post-war period",
        ),
    ],
)

DECADES_PRESET = EraPreset(
    key="decades",
    name="Decades",
    description="Simple decade-based periodization",
    eras=[
        EraDefinition(name="1900s", range=EraRange(start=1900, end=1909)),
        EraDefinition(name="1910s", range=EraRange(start=1910, end=1919)),
        EraDefinition(name="1920s", range=EraRange(start=1920, end=1929)),
    ],
)

FIVE_YEAR_PRESET = EraPreset(
    key="five_year",
    name="Five-Year Periods",
    description="Five-year periodization for finer granularity",
    eras=[
        EraDefinition(name="1900_1904", range=EraRange(start=1900, end=1904)),
        EraDefinition(name="1905_1909", range=EraRange(start=1905, end=1909)),
        EraDefinition(name="1910_1914", range=EraRange(start=1910, end=1914)),
        EraDefinition(name="1915_1919", range=EraRange(start=1915, end=1919)),
        EraDefinition(name="1920_1924", range=EraRange(start=1920, end=1924)),
    ],
)


# =============================================================================
# Preset Registry
# =============================================================================

ERA_PRESETS: dict[str, EraPreset] = {
    "wwi": WWI_PRESET,
    "wwi_binary": WWI_BINARY_PRESET,
    "mid_war_split": MID_WAR_SPLIT_PRESET,
    "war_phases": WAR_PHASES_PRESET,
    "decades": DECADES_PRESET,
    "five_year": FIVE_YEAR_PRESET,
}


def get_preset(name: str) -> EraPreset:
    """
    Get an era preset by name.

    Args:
        name: Preset name (e.g., "wwi", "war_phases")

    Returns:
        EraPreset instance.

    Raises:
        ValueError: If preset name is not found.
    """
    if name not in ERA_PRESETS:
        available = ", ".join(ERA_PRESETS.keys())
        msg = f"Unknown preset '{name}'. Available: {available}"
        raise ValueError(msg)
    return ERA_PRESETS[name]


def list_presets() -> dict[str, str]:
    """
    List available presets with their descriptions.

    Returns:
        Dictionary mapping preset names to descriptions.
    """
    return {name: preset.description for name, preset in ERA_PRESETS.items()}
