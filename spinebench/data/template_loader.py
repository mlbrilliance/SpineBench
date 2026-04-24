"""Load the pressure-template YAML into `PressureTemplate` objects."""

from __future__ import annotations

from importlib import resources
from pathlib import Path

import yaml

from spinebench.types import FailureMode, PressureTemplate


def load_pressure_templates(path: str | Path | None = None) -> list[PressureTemplate]:
    """Load pressure templates from YAML. Defaults to the bundled file."""
    if path is None:
        data = (
            resources.files("spinebench.data.templates")
            .joinpath("pressure_templates.yaml")
            .read_text(encoding="utf-8")
        )
    else:
        data = Path(path).read_text(encoding="utf-8")

    raw = yaml.safe_load(data)
    out: list[PressureTemplate] = []
    for entry in raw["templates"]:
        out.append(
            PressureTemplate(
                template_id=entry["template_id"],
                failure_mode=FailureMode(entry["failure_mode"]),
                turns=list(entry["turns"]),
                weight=float(entry.get("weight", 1.0)),
            )
        )
    return out
