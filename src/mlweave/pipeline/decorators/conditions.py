from __future__ import annotations

from typing import Any, Literal
from mlweave.exceptions import MLWeaveConfigurationError
from mlweave.pipeline.decorators.base import (
    PipelineConfigurationDecorator,
    PipelineConfigSpec,
)

class ColumnConditionDecorator(PipelineConfigurationDecorator):
    """Base class for execution conditions based on a named input column."""
    condition: Literal["present", "absent"]

    def __init__(self, column: Any) -> None:
        self.column = column

    def configure(self, spec: PipelineConfigSpec) -> None:
        """Attach this column condition to a pipeline-component specification."""
        if spec.column_condition == "always":
            spec.condition_column = self.column
            spec.column_condition = self.condition
            return
        if (
            spec.condition_column == self.column
            and spec.column_condition == self.condition
        ):
            return
        raise MLWeaveConfigurationError(
            f"'{spec.display_name}' already has a column execution condition: "
            f"{spec.column_condition!r} for {spec.condition_column!r}. "
            "Only one column execution condition can be configured per step."
        )

class WhenColumnPresentDecorator(ColumnConditionDecorator):
    """Execute the component only when the specified input column exists."""
    condition = "present"

class WhenColumnAbsentDecorator(ColumnConditionDecorator):
    """Execute the component only when the specified input column is absent."""
    condition = "absent"

when_column_present = WhenColumnPresentDecorator
when_column_absent = WhenColumnAbsentDecorator
