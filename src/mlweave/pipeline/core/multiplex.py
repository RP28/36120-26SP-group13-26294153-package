from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Callable, TypeVar

from mlweave.exceptions import MLWeaveValidationError


T = TypeVar("T")


def is_multiplexed(value: Any) -> bool:
    """Return whether ``value`` uses MLWeave's tuple partition convention."""
    return isinstance(value, tuple)


def partition_count(X: Any) -> int:
    """Return the number of partitions represented by ``X``."""
    return len(X) if is_multiplexed(X) else 1


def row_count(value: Any) -> int | None:
    """Return a cheap sample count when one can be determined."""
    shape = getattr(value, "shape", None)
    if shape is not None and len(shape) >= 1:
        try:
            return int(shape[0])
        except (TypeError, ValueError):
            pass

    try:
        return len(value)
    except TypeError:
        return None


def shape_of(value: Any) -> Any:
    """Return a compact shape description without materialising data."""
    if is_multiplexed(value):
        return tuple(getattr(part, "shape", None) for part in value)
    return getattr(value, "shape", None)


def validate_multiplex(
    X: tuple[Any, ...],
    y: tuple[Any, ...] | None = None,
    *,
    require_multiple: bool = True,
) -> None:
    """Validate MLWeave's partition tuple contract.

    Partition ``0`` is always treated as training data. Remaining partitions
    are non-training datasets such as validation or test data.
    """
    if not isinstance(X, tuple):
        raise MLWeaveValidationError("Multiplexed X must be a tuple.")

    minimum = 2 if require_multiple else 1
    if len(X) < minimum:
        raise MLWeaveValidationError(
            f"Multiplexed X must contain at least {minimum} partition(s); "
            "partition 0 is the training dataset."
        )

    if any(part is None for part in X):
        raise MLWeaveValidationError("Multiplexed X cannot contain None partitions.")

    if y is None:
        return

    if not isinstance(y, tuple):
        raise MLWeaveValidationError(
            "When multiplexed X is paired with y, y must also be a tuple."
        )

    if len(X) != len(y):
        raise MLWeaveValidationError(
            "Multiplexed X and y must contain the same number of partitions: "
            f"got {len(X)} X partition(s) and {len(y)} y partition(s)."
        )

    for index, (X_part, y_part) in enumerate(zip(X, y)):
        x_rows = row_count(X_part)
        y_rows = row_count(y_part)
        if x_rows is None or y_rows is None:
            continue
        if x_rows != y_rows:
            label = "training" if index == 0 else f"partition {index}"
            raise MLWeaveValidationError(
                f"Multiplexed {label} X/y row counts differ: "
                f"{x_rows} != {y_rows}."
            )


def training_value(value: T | tuple[T, ...] | None) -> T | None:
    """Return partition zero for multiplexed input, otherwise the value itself."""
    if isinstance(value, tuple):
        if not value:
            raise MLWeaveValidationError(
                "A multiplexed tuple cannot be empty; partition 0 must be training data."
            )
        return value[0]
    return value


def map_partitions(func: Callable[[Any], T], X: tuple[Any, ...]) -> tuple[T, ...]:
    """Apply ``func`` independently to each partition without copying inputs."""
    return tuple(func(part) for part in X)


def partition_parameter(value: Any, index: int, count: int) -> Any:
    """Select aligned tuple metadata for one partition when unambiguous.

    Ordinary non-tuple values are shared across partitions. A tuple whose length
    exactly matches the dataset partition count is interpreted as partitioned
    metadata (for example per-split sample weights).
    """
    if isinstance(value, tuple) and len(value) == count:
        return value[index]
    return value


def partition_mapping(
    params: Mapping[str, Any] | None,
    index: int,
    count: int,
) -> dict[str, Any]:
    """Return one partition's view of runtime parameters without data copies."""
    if not params:
        return {}
    return {
        key: partition_parameter(value, index, count)
        for key, value in params.items()
    }
