from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Literal

WorkflowStepMode = Literal["model_selection", "inference"]

@dataclass(slots=True)

class WorkflowStepSpec:
    """Lightweight declarative configuration for an MLWorkflow function step."""
    func: Callable[..., Any]
    mode: WorkflowStepMode | None = None

    @property
    def display_name(self) -> str:
        return self.func.__name__
