from __future__ import annotations

from functools import update_wrapper
from typing import Any, Callable
from mlweave.exceptions import MLWeaveConfigurationError
from mlweave.pipeline.core.specs import PipelineStepSpec
from mlweave.pipeline.core.split import MLWeaveSplitStep
from mlweave.pipeline.core.step import MLWeavePipelineStep

class PendingPipelineStep:
    """Hold pipeline decorator configuration until a finalizer is applied."""
    def __init__(self, spec: PipelineStepSpec) -> None:
        self.spec = spec
        update_wrapper(self, spec.transform_func)

    def __call__(self, *args, **kwargs):
        raise MLWeaveConfigurationError(
            f"'{self.spec.transform_func.__name__}' uses mlweave pipeline decorators but is missing @pipeline_step, "
            "@stateful_pipeline_step, or @split_step. Decorate the function with the appropriate finalizer before using it.")

    def __repr__(self) -> str:
        return (
            f"<PendingPipelineStep {self.spec.transform_func.__name__!r}: missing pipeline finalizer>")

class PipelineStepBuilder:
    """Lazy factory that creates an sklearn-compatible component when called."""
    def __init__(self, spec: PipelineStepSpec) -> None:
        self.spec = spec
        update_wrapper(self, spec.transform_func)

    def __call__(self, *args: Any, **kwargs: Any):
        if self.spec.mode is None:
            raise MLWeaveConfigurationError(
                f"'{self.spec.transform_func.__name__}' is not finalized. Add @pipeline_step, @stateful_pipeline_step, or @split_step.")
        if self.spec.mode == "stateful" and self.spec.fit_func is None:
            raise MLWeaveConfigurationError(
                f"'{self.spec.transform_func.__name__}' uses @stateful_pipeline_step but has no fit function. "
                f"Define one with @{self.spec.transform_func.__name__}.fit.")
        common = dict(
            call_args=tuple(args),
            call_kwargs=dict(kwargs),
            validators=tuple(sorted(self.spec.validators, key=lambda item: item.priority)),
            output_validators=tuple(sorted(self.spec.output_validators, key=lambda item: item.priority)),
            tracking=self.spec.tracking,
            description=self.spec.description,
            tags=tuple(sorted(self.spec.tags)),
            condition_column=self.spec.condition_column,
            column_condition=self.spec.column_condition)
        if self.spec.mode == "split":
            return MLWeaveSplitStep(split_func=self.spec.transform_func, **common)
        return MLWeavePipelineStep(
            transform_func=self.spec.transform_func, fit_func=self.spec.fit_func,
            stateful=self.spec.mode == "stateful", **common)

    def fit(self, func: Callable[..., Any]) -> Callable[..., Any]:
        """Register the fit function for an @stateful_pipeline_step."""
        if self.spec.mode != "stateful":
            raise MLWeaveConfigurationError(
                f"'{self.spec.transform_func.__name__}' is not a stateful pipeline step. Use @stateful_pipeline_step before registering fit.")
        if self.spec.fit_func is not None:
            raise MLWeaveConfigurationError(
                f"A fit function is already registered for '{self.spec.transform_func.__name__}'.")
        self.spec.fit_func = func
        return func

    def __repr__(self) -> str:
        mode = self.spec.mode or "unfinalized"
        return (
            f"<PipelineStepBuilder {self.spec.transform_func.__name__!r} "
            f"mode={mode!r}>"
        )
