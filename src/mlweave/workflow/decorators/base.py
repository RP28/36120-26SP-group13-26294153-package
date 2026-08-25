from __future__ import annotations

from abc import ABC
from typing import Any

from mlweave.exceptions import MLWeaveConfigurationError
from mlweave.workflow.core.builder import WorkflowStepBuilder
from mlweave.workflow.core.specs import WorkflowStepMode, WorkflowStepSpec


class BaseWorkflowDecorator(ABC):
    """Common plumbing shared by MLWorkflow function decorators."""

    def _get_spec(self, obj: Any) -> WorkflowStepSpec:
        if isinstance(obj, WorkflowStepBuilder):
            return obj.spec

        if callable(obj):
            return WorkflowStepSpec(func=obj)

        raise TypeError(
            "mlweave workflow decorators can only be applied to a callable "
            "or an mlweave workflow-step builder."
        )


class WorkflowFinalizingDecorator(BaseWorkflowDecorator):
    """Base class for decorators that finalize an MLWorkflow step mode."""

    mode: WorkflowStepMode

    def __call__(self, obj: Any) -> WorkflowStepBuilder:
        spec = self._get_spec(obj)

        if spec.mode is not None and spec.mode != self.mode:
            raise MLWeaveConfigurationError(
                f"'{spec.display_name}' is already declared as a "
                f"{spec.mode!r} workflow step and cannot also be declared "
                f"as {self.mode!r}."
            )

        spec.mode = self.mode

        if isinstance(obj, WorkflowStepBuilder):
            return obj

        return WorkflowStepBuilder(spec)
