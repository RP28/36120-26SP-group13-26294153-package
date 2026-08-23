from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Literal

from sklearn.base import BaseEstimator

from mlweave.exceptions import MLWeaveConfigurationError
from mlweave.pipeline.core.builder import PendingPipelineStep, PipelineStepBuilder
from mlweave.pipeline.core.specs import (
    OutputValidationSpec,
    PipelineStepSpec,
    ValidationSpec,
    WrappedStepSpec,
)
from mlweave.pipeline.core.wrapped import MLWeaveWrappedStep


PipelineConfigSpec = PipelineStepSpec | WrappedStepSpec


class BasePipelineDecorator(ABC):
    """Common plumbing shared by all mlweave pipeline decorators."""

    def _get_spec(self, obj: Any) -> PipelineConfigSpec:
        if isinstance(obj, (PendingPipelineStep, PipelineStepBuilder)):
            return obj.spec

        if isinstance(obj, MLWeaveWrappedStep):
            return obj.spec

        if isinstance(obj, BaseEstimator):
            raise MLWeaveConfigurationError(
                f"{obj.__class__.__name__} is an existing sklearn estimator. "
                "Call wrap_step(estimator) before applying mlweave pipeline "
                "configuration decorators."
            )

        if callable(obj):
            return PipelineStepSpec(transform_func=obj)

        raise TypeError(
            "mlweave pipeline decorators can only be applied to a callable, "
            "an mlweave pipeline-step specification, or wrap_step(...) output."
        )

    def _preserve_state(self, obj: Any, spec: PipelineConfigSpec):
        # Reuse the same object while decorators are stacked/applied. Function
        # steps keep one pending/builder wrapper, while wrapped sklearn steps
        # remain the same estimator wrapper instance.
        if isinstance(
            obj,
            (PendingPipelineStep, PipelineStepBuilder, MLWeaveWrappedStep),
        ):
            return obj
        if isinstance(spec, WrappedStepSpec):
            raise MLWeaveConfigurationError(
                "Wrapped sklearn step configuration lost its wrapper instance."
            )
        return PendingPipelineStep(spec)


class PipelineConfigurationDecorator(BasePipelineDecorator):
    """Base class for decorators that modify pipeline component configuration."""

    def __call__(self, obj: Any):
        spec = self._get_spec(obj)
        self.configure(spec)
        return self._preserve_state(obj, spec)

    @abstractmethod
    def configure(self, spec: PipelineConfigSpec) -> None:
        """Mutate the shared pipeline-component specification."""


class PipelineValidationDecorator(PipelineConfigurationDecorator):
    """Base class for validation-oriented configuration decorators."""


class PipelineInputValidationDecorator(PipelineValidationDecorator):
    """Base class for validations performed before fit/transform execution."""

    @abstractmethod
    def validation_spec(self) -> ValidationSpec:
        """Return the input-validation configuration contributed by the decorator."""

    def configure(self, spec: PipelineConfigSpec) -> None:
        spec.validators.append(self.validation_spec())
        spec.validators.sort(key=lambda item: item.priority)


class PipelineOutputValidationDecorator(PipelineValidationDecorator):
    """Base class for validations performed after transform execution."""

    @abstractmethod
    def validation_spec(self) -> OutputValidationSpec:
        """Return the output-validation configuration contributed by the decorator."""

    def configure(self, spec: PipelineConfigSpec) -> None:
        spec.output_validators.append(self.validation_spec())
        spec.output_validators.sort(key=lambda item: item.priority)


class PipelineMetadataDecorator(PipelineConfigurationDecorator):
    """Base class for decorators that only attach descriptive metadata."""


class PipelineFinalizingDecorator(BasePipelineDecorator):
    """Base class for decorators that finalize a function pipeline-step mode."""

    mode: Literal["stateless", "stateful", "split"]

    def __call__(self, obj: Any) -> PipelineStepBuilder:
        spec = self._get_spec(obj)

        if isinstance(spec, WrappedStepSpec):
            raise MLWeaveConfigurationError(
                f"'{spec.display_name}' is already an sklearn transformer "
                "adapted with wrap_step(...). Pipeline finalizer decorators are only "
                "for Python functions."
            )

        if spec.mode is not None and spec.mode != self.mode:
            raise MLWeaveConfigurationError(
                f"'{spec.display_name}' is already declared as "
                f"a {spec.mode!r} pipeline step and cannot also be declared "
                f"as {self.mode!r}."
            )

        spec.mode = self.mode

        if isinstance(obj, PipelineStepBuilder):
            return obj

        return PipelineStepBuilder(spec)
