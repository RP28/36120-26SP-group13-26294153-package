from __future__ import annotations

from functools import update_wrapper
from typing import Any

from mlweave.exceptions import MLWeaveConfigurationError
from mlweave.workflow.core.inference import MLWeaveInferenceStep
from mlweave.workflow.core.selection import MLWeaveModelSelectionPolicy
from mlweave.workflow.core.specs import WorkflowStepSpec


class WorkflowStepBuilder:
    """Lazy factory that materializes one configured MLWorkflow component."""

    def __init__(self, spec: WorkflowStepSpec) -> None:
        self.spec = spec
        update_wrapper(self, spec.func)

    def __call__(self, *args: Any, **kwargs: Any):
        if self.spec.mode == "model_selection":
            return MLWeaveModelSelectionPolicy(
                selection_func=self.spec.func,
                call_args=tuple(args),
                call_kwargs=dict(kwargs),
            )

        if self.spec.mode == "inference":
            return MLWeaveInferenceStep(
                inference_func=self.spec.func,
                call_args=tuple(args),
                call_kwargs=dict(kwargs),
            )

        raise MLWeaveConfigurationError(
            f"'{self.spec.display_name}' is not a finalized MLWorkflow step."
        )

    def __repr__(self) -> str:
        mode = self.spec.mode or "unfinalized"
        return (
            f"<WorkflowStepBuilder {self.spec.display_name!r} "
            f"mode={mode!r}>"
        )
