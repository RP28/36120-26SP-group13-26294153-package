from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, FrozenSet, Literal


PipelineStepMode = Literal["stateless", "stateful"]
ValidationStage = Literal["fit", "transform", "both"]
SnapshotField = Literal["row_count", "column_count", "columns"]


@dataclass(slots=True, frozen=True)
class ValidationSpec:
    """Configuration for one validator attached to a pipeline component."""

    validator: Callable[..., Any]
    args: tuple[Any, ...] = ()
    kwargs: dict[str, Any] = field(default_factory=dict)
    stage: ValidationStage = "both"
    priority: int = 100


@dataclass(slots=True, frozen=True)
class OutputValidationSpec:
    """Configuration for one validator that runs after transformation.

    ``snapshot_fields`` declares exactly which pieces of input structure must
    be captured before transformation. This keeps cheap output contracts cheap:
    for example, ``@no_missing_output`` does not need any input snapshot.
    """

    validator: Callable[..., Any]
    args: tuple[Any, ...] = ()
    kwargs: dict[str, Any] = field(default_factory=dict)
    priority: int = 100
    snapshot_fields: FrozenSet[SnapshotField] = frozenset()


@dataclass(slots=True)
class PipelineStepSpec:
    """Lightweight declarative configuration built for function pipeline steps."""

    transform_func: Callable[..., Any]
    fit_func: Callable[..., Any] | None = None
    validators: list[ValidationSpec] = field(default_factory=list)
    output_validators: list[OutputValidationSpec] = field(default_factory=list)
    tracking: bool = False
    description: str | None = None
    tags: set[str] = field(default_factory=set)
    mode: PipelineStepMode | None = None

    @property
    def display_name(self) -> str:
        return self.transform_func.__name__


@dataclass(slots=True)
class WrappedStepSpec:
    """Configuration attached to an existing sklearn transformer wrapper.

    The sklearn estimator itself intentionally lives on ``MLWeaveWrappedStep``
    rather than in this spec so sklearn can expose/tune nested parameters as
    ``estimator__<parameter>``.
    """

    component_name: str
    validators: list[ValidationSpec] = field(default_factory=list)
    output_validators: list[OutputValidationSpec] = field(default_factory=list)
    tracking: bool = False
    description: str | None = None
    tags: set[str] = field(default_factory=set)

    @property
    def display_name(self) -> str:
        return self.component_name
