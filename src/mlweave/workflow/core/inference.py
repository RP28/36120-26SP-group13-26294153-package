from __future__ import annotations

from typing import Any, Callable
from sklearn.base import BaseEstimator
from mlweave.workflow.core.context import WorkflowContext

class MLWeaveInferenceStep(BaseEstimator):
    """Runtime component for a user-defined workflow inference function."""
    def __init__(
        self,
        inference_func: Callable[..., Any],
        call_args: tuple[Any, ...] = (),
        call_kwargs: dict[str, Any] | None = None,
    ) -> None:
        self.inference_func = inference_func
        self.call_args = call_args
        self.call_kwargs = call_kwargs

    def run(self, context: WorkflowContext):
        """Execute the configured inference function against workflow state."""
        if not isinstance(context, WorkflowContext):
            raise TypeError(
                "MLWeaveInferenceStep.run() expects a WorkflowContext instance."
            )
        return self.inference_func(
            context,
            *self.call_args,
            **dict(self.call_kwargs or {}),
        )
