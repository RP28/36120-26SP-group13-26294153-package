from __future__ import annotations

from collections import Counter
from collections.abc import Iterable
from typing import Any
import numpy as np
from scipy import sparse
from mlweave.exceptions import MLWeaveValidationError
from mlweave.pipeline.core.specs import OutputValidationSpec, ValidationSpec
from mlweave.pipeline.decorators.base import (
    PipelineInputValidationDecorator,
    PipelineOutputValidationDecorator,
)

_SCAN_CHUNK_SIZE = 1_000_000
_MAX_INVALID_VALUES_IN_ERROR = 10

def _columns_index(X: Any):
    """Return the native columns object without copying it."""
    return getattr(X, "columns", None)

def _column_names(X: Any) -> tuple[Any, ...] | None:
    """Snapshot column labels only when a later contract really needs them."""
    columns = _columns_index(X)
    if columns is None:
        return None
    return tuple(columns)

def _row_count(X: Any) -> int | None:
    shape = getattr(X, "shape", None)
    if shape is not None and len(shape) >= 1:
        return int(shape[0])
    try:
        return len(X)
    except TypeError:
        return None

def _column_count(X: Any) -> int | None:
    shape = getattr(X, "shape", None)
    if shape is not None and len(shape) >= 2:
        return int(shape[1])
    columns = _columns_index(X)
    if columns is not None:
        return len(columns)
    return None

def _pandas_like_contains_missing(X: Any) -> bool | None:
    """Check pandas-like objects in bounded vectorised row chunks.
    A full ``DataFrame.isna()`` creates a boolean object as large as the input.
    Instead, choose the number of rows from an element budget so each temporary
    missing-value mask stays around ``_SCAN_CHUNK_SIZE`` booleans. This keeps
    peak memory bounded while retaining pandas/NumPy vectorised speed.
    """
    iloc = getattr(X, "iloc", None)
    isna = getattr(X, "isna", None)
    shape = getattr(X, "shape", None)
    if iloc is None or not callable(isna) or shape is None:
        return None
    row_count = int(shape[0]) if len(shape) >= 1 else 0
    column_count = int(shape[1]) if len(shape) >= 2 else 1
    rows_per_chunk = max(1, _SCAN_CHUNK_SIZE // max(1, column_count))
    for start in range(0, row_count, rows_per_chunk):
        chunk = X.iloc[start : start + rows_per_chunk]
        missing = chunk.isna()
        to_numpy = getattr(missing, "to_numpy", None)
        if callable(to_numpy):
            if bool(to_numpy(copy=False).any()):
                return True
            continue
        any_method = getattr(missing, "any", None)
        if callable(any_method):
            result = any_method()
            if hasattr(result, "any"):
                result = result.any()
            if bool(result):
                return True
    return False

def _ndarray_contains_missing(array: np.ndarray) -> bool:
    """Missing-value scan that bounds temporary allocations for large arrays."""
    flat = array.reshape(-1)
    if np.issubdtype(array.dtype, np.number):
        for start in range(0, flat.size, _SCAN_CHUNK_SIZE):
            chunk = flat[start : start + _SCAN_CHUNK_SIZE]
            if bool(np.isnan(chunk).any()):
                return True
        return False
    if np.issubdtype(array.dtype, np.datetime64) or np.issubdtype(
        array.dtype, np.timedelta64
    ):
        for start in range(0, flat.size, _SCAN_CHUNK_SIZE):
            chunk = flat[start : start + _SCAN_CHUNK_SIZE]
            if bool(np.isnat(chunk).any()):
                return True
        return False
    if np.issubdtype(array.dtype, np.str_) or np.issubdtype(array.dtype, np.bytes_):
        return False
    for start in range(0, flat.size, _SCAN_CHUNK_SIZE):
        chunk = flat[start : start + _SCAN_CHUNK_SIZE]
        for value in chunk:
            if value is None:
                return True
            try:
                if bool(np.isnan(value)):
                    return True
            except (TypeError, ValueError):
                pass
    return False

def _contains_missing(X: Any) -> bool:
    """Return whether X contains missing values with bounded peak memory."""
    pandas_result = _pandas_like_contains_missing(X)
    if pandas_result is not None:
        return pandas_result
    if sparse.issparse(X):
        return _ndarray_contains_missing(np.asarray(X.data))
    return _ndarray_contains_missing(np.asarray(X))

def _validate_required_columns(X: Any, columns: tuple[Any, ...]) -> None:
    available = _columns_index(X)
    if available is None:
        raise MLWeaveValidationError(
            "@requires_columns can only validate inputs that expose a "
            "'columns' attribute, such as a pandas DataFrame."
        )
    missing = [column for column in columns if column not in available]
    if missing:
        raise MLWeaveValidationError(f"Required columns are missing: {missing}.")

def _validate_no_missing_input(X: Any) -> None:
    if _contains_missing(X):
        raise MLWeaveValidationError("Pipeline step input contains missing values.")

def _invalid_values_from_pandas_series(
    series: Any,
    allowed: tuple[Any, ...],
) -> list[Any] | None:
    """Return a few invalid values, scanning large Series in row chunks."""
    isin = getattr(series, "isin", None)
    iloc = getattr(series, "iloc", None)
    if not callable(isin) or iloc is None:
        return None
    length = len(series)
    for start in range(0, length, _SCAN_CHUNK_SIZE):
        chunk = series.iloc[start : start + _SCAN_CHUNK_SIZE]
        invalid_mask = ~chunk.isin(allowed)
        if not bool(invalid_mask.any()):
            continue
        invalid = chunk[invalid_mask]
        unique = getattr(invalid, "unique", None)
        if callable(unique):
            return list(unique()[:_MAX_INVALID_VALUES_IN_ERROR])
        return list(invalid[:_MAX_INVALID_VALUES_IN_ERROR])
    return []

def _validate_allowed_values(
    X: Any,
    column: Any,
    allowed: tuple[Any, ...],
) -> None:
    available = _columns_index(X)
    if available is None:
        raise MLWeaveValidationError(
            "@allowed_values can only validate inputs that expose a "
            "'columns' attribute, such as a pandas DataFrame."
        )
    if column not in available:
        raise MLWeaveValidationError(
            f"Column {column!r} required by @allowed_values is missing."
        )
    series = X[column]
    invalid_values = _invalid_values_from_pandas_series(series, allowed)
    if invalid_values is None:
        allowed_lookup = frozenset(allowed)
        invalid_values = []
        seen = set()
        for value in series:
            if value in allowed_lookup or value in seen:
                continue
            seen.add(value)
            invalid_values.append(value)
            if len(invalid_values) >= _MAX_INVALID_VALUES_IN_ERROR:
                break
    if invalid_values:
        suffix = (
            " (showing up to "
            f"{_MAX_INVALID_VALUES_IN_ERROR} invalid values)"
        )
        allowed_preview = list(allowed[:_MAX_INVALID_VALUES_IN_ERROR])
        allowed_suffix = (
            " ..." if len(allowed) > _MAX_INVALID_VALUES_IN_ERROR else ""
        )
        raise MLWeaveValidationError(
            f"Column {column!r} contains values outside the allowed set: "
            f"{invalid_values}{suffix}. Allowed values include: "
            f"{allowed_preview}{allowed_suffix}."
        )

def _validate_preserve_rows(input_snapshot: dict[str, Any], output: Any) -> None:
    before = input_snapshot.get("row_count")
    after = _row_count(output)
    if before is None or after is None:
        raise MLWeaveValidationError(
            "@preserve_rows could not determine the number of rows."
        )
    if before != after:
        raise MLWeaveValidationError(
            f"Pipeline step changed the number of rows from {before} to {after}."
        )

def _validate_preserve_columns(
    input_snapshot: dict[str, Any],
    output: Any,
) -> None:
    before_names = input_snapshot.get("columns")
    after_names = _column_names(output)
    if before_names is not None and after_names is not None:
        if before_names == after_names:
            return
        before_counts = Counter(before_names)
        after_counts = Counter(after_names)
        if before_counts != after_counts:
            removed = list((before_counts - after_counts).elements())
            added = list((after_counts - before_counts).elements())
            raise MLWeaveValidationError(
                "Pipeline step changed the columns. "
                f"Removed: {removed}; added: {added}."
            )
        return
    before_count = input_snapshot.get("column_count")
    after_count = _column_count(output)
    if before_count is None or after_count is None:
        raise MLWeaveValidationError(
            "@preserve_columns could not determine the number of columns."
        )
    if before_count != after_count:
        raise MLWeaveValidationError(
            "Pipeline step changed the number of columns from "
            f"{before_count} to {after_count}."
        )

def _validate_no_missing_output(
    input_snapshot: dict[str, Any],
    output: Any,
) -> None:
    del input_snapshot
    if _contains_missing(output):
        raise MLWeaveValidationError("Pipeline step output contains missing values.")

class RequiresColumnsDecorator(PipelineInputValidationDecorator):
    """Require named columns before fit or transform execution."""
    def __init__(self, columns: Iterable[Any]) -> None:
        self.columns = tuple(columns)
        if not self.columns:
            raise ValueError("@requires_columns requires at least one column.")

    def validation_spec(self) -> ValidationSpec:
        return ValidationSpec(
            validator=_validate_required_columns,
            args=(self.columns,),
            stage="both",
            priority=10,
        )

class NoMissingInputDecorator(PipelineInputValidationDecorator):
    """Reject inputs containing missing values."""
    def validation_spec(self) -> ValidationSpec:
        return ValidationSpec(
            validator=_validate_no_missing_input,
            stage="both",
            priority=20,
        )

class AllowedValuesDecorator(PipelineInputValidationDecorator):
    """Restrict one input column to a declared set of values."""
    def __init__(self, column: Any, values: Iterable[Any]) -> None:
        self.column = column
        self.values = tuple(values)
        if not self.values:
            raise ValueError("@allowed_values requires at least one allowed value.")

    def validation_spec(self) -> ValidationSpec:
        return ValidationSpec(
            validator=_validate_allowed_values,
            args=(self.column, self.values),
            stage="both",
            priority=30,
        )

class PreserveRowsDecorator(PipelineOutputValidationDecorator):
    """Ensure transform does not add or remove rows."""
    def validation_spec(self) -> OutputValidationSpec:
        return OutputValidationSpec(
            validator=_validate_preserve_rows,
            priority=10,
            snapshot_fields=frozenset({"row_count"}),
        )

class PreserveColumnsDecorator(PipelineOutputValidationDecorator):
    """Ensure transform does not add, remove, or rename columns."""
    def validation_spec(self) -> OutputValidationSpec:
        return OutputValidationSpec(
            validator=_validate_preserve_columns,
            priority=20,
            snapshot_fields=frozenset({"columns", "column_count"}),
        )

class NoMissingOutputDecorator(PipelineOutputValidationDecorator):
    """Reject transformed outputs containing missing values."""
    def validation_spec(self) -> OutputValidationSpec:
        return OutputValidationSpec(
            validator=_validate_no_missing_output,
            priority=30,
        )

requires_columns = RequiresColumnsDecorator
no_missing_input = NoMissingInputDecorator()
allowed_values = AllowedValuesDecorator
preserve_rows = PreserveRowsDecorator()
preserve_columns = PreserveColumnsDecorator()
no_missing_output = NoMissingOutputDecorator()
