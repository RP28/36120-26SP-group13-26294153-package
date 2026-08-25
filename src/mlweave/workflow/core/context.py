from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from mlweave.exceptions import MLWeaveValidationError


@dataclass(slots=True)
class WorkflowContext:
    """Runtime state exposed to an ``@inference_step`` function."""
    model: Any
    search: Any
    preprocessing: Any
    training_data: Any
    X_parts: tuple[Any, ...]
    y_parts: tuple[Any, ...]
    partition_names: tuple[str, ...]
    best_index: int
    best_params: dict[str, Any]
    training_metrics: dict[str, float]
    validation_metrics: dict[str, float]
    inference_data: Any | None = None
    X_inference: Any | None = None
    inference_index: Any | None = None

    def partition(self, name: str) -> tuple[Any, Any]:
        """Return ``(X, y)`` for one named fitted partition."""
        try:
            index = self.partition_names.index(name)
        except ValueError as exc:
            raise MLWeaveValidationError(
                f"Unknown workflow partition {name!r}. Available partitions: "
                f"{self.partition_names}."
            ) from exc

        return self.X_parts[index], self.y_parts[index]

    @property
    def X_train(self):
        return self.partition("train")[0]

    @property
    def y_train(self):
        return self.partition("train")[1]

    @property
    def X_validation(self):
        return self.partition("validation")[0]

    @property
    def y_validation(self):
        return self.partition("validation")[1]

    @property
    def X_test(self):
        return self.partition("test")[0]

    @property
    def y_test(self):
        return self.partition("test")[1]
