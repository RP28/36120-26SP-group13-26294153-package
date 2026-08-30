from __future__ import annotations

from typing import Any, Callable
from mlweave.pipeline.core.specs import ValidationSpec
from mlweave.pipeline.decorators.base import PipelineInputValidationDecorator

class ValidateDecorator(PipelineInputValidationDecorator):
    """Attach a validator and optional arguments to a pipeline step."""
    def __init__(
        self,
        validator: Callable[..., Any],
        *args: Any,
        stage: str = "both",
        **kwargs: Any,
    ) -> None:
        if stage not in {"fit", "transform", "both"}:
            raise ValueError("stage must be 'fit', 'transform', or 'both'.")
        self.validator = validator
        self.args = args
        self.kwargs = kwargs
        self.stage = stage

    def validation_spec(self) -> ValidationSpec:
        return ValidationSpec(
            validator=self.validator,
            args=self.args,
            kwargs=self.kwargs,
            stage=self.stage,
            priority=100,
        )

validate = ValidateDecorator
