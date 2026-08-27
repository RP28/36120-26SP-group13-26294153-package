from __future__ import annotations

from time import perf_counter
from typing import Any, Callable, Iterable
import numpy as np
from sklearn.base import BaseEstimator
from mlweave.exceptions import MLWeaveValidationError
from mlweave.pipeline.core.multiplex import shape_of, validate_multiplex
from mlweave.pipeline.core.specs import ColumnCondition, OutputValidationSpec, SnapshotField, ValidationSpec

class MLWeaveSplitStep(BaseEstimator):
    """Fit-time boundary that converts one dataset into ordered partitions."""
    def __init__(self,
        split_func: Callable[..., Any],
        call_args: tuple[Any, ...] = (),
        call_kwargs: dict[str, Any] | None = None,
        validators: tuple[ValidationSpec, ...] = (),
        output_validators: tuple[OutputValidationSpec, ...] = (),
        tracking: bool = False,
        description: str | None = None,
        tags: tuple[str, ...] = (),
        condition_column: Any | None = None,
        column_condition: ColumnCondition = "always",
    ) -> None:
        self.split_func = split_func
        self.call_args = call_args
        self.call_kwargs = call_kwargs
        self.validators = validators
        self.output_validators = output_validators
        self.tracking = tracking
        self.description = description
        self.tags = tags
        self.condition_column = condition_column
        self.column_condition = column_condition
        self._snapshot_fields = self._collect_snapshot_fields(output_validators)

    def fit(self, X, y=None, **fit_params):
        """Satisfy sklearn's transformer protocol without splitting eagerly."""
        del X, y, fit_params
        self._mlweave_is_fitted_ = True
        return self

    def transform(self, X, **params):
        """Pass inference data through unchanged; splitting is fit-time only."""
        del params
        return X

    def fit_transform(self, X, y=None, **fit_params):
        """Split X directly and retain y partitions for direct-step inspection.
        MLWeave ``Pipeline`` consumes both X and y partitions. Direct use of a
        split step can only return X because sklearn's transformer contract has
        a single transformed-data return value.
        """
        X_parts, y_parts = self.split(X, y, **fit_params)
        self.split_y_ = y_parts
        return X_parts

    def inverse_transform(self, X, **params):
        """A split boundary has no feature transformation to invert."""
        del params
        return X

    def set_output(self, *, transform=None):
        """Accept sklearn set_output propagation; split itself changes no format."""
        del transform
        return self

    def get_feature_names_out(self, input_features=None):
        """Return feature names produced at the split boundary when known."""
        output_names = getattr(self, "feature_names_out_", None)
        if output_names is not None:
            return output_names
        if input_features is not None:
            return np.asarray(input_features, dtype=object)
        feature_names = getattr(self, "feature_names_in_", None)
        if feature_names is not None:
            return feature_names
        raise AttributeError(
            "input_features must be provided before this split step is fitted."
        )

    def split(self, X, y=None, **fit_params):
        """Execute and validate the split function."""
        if not self._should_execute(X):
            self._print_track_skip(X)
            return X, y
        started = perf_counter() if self.tracking else None
        self._validate(X)
        input_snapshot = self._snapshot_input(X, self._snapshot_fields)
        kwargs = dict(self.call_kwargs or {})
        overlap = kwargs.keys() & fit_params.keys()
        if overlap:
            raise TypeError(
                "Split-step configured arguments conflict with runtime fit "
                f"parameters: {sorted(overlap)}."
            )
        kwargs.update(fit_params)
        output = self.split_func(X, y, *self.call_args, **kwargs)
        X_parts, y_parts = self._normalise_output(output, input_y=y)
        self._capture_feature_names(X, X_parts[0])
        for part in X_parts:
            self._validate_output(input_snapshot, part)
        self._mlweave_is_fitted_ = True
        self._print_track_event(X, X_parts, y_parts, started)
        return X_parts, y_parts

    def _capture_feature_names(self, input_X, training_X) -> None:
        input_shape = getattr(input_X, "shape", None)
        if input_shape is not None and len(input_shape) >= 2:
            self.n_features_in_ = int(input_shape[1])
        input_columns = getattr(input_X, "columns", None)
        if input_columns is not None:
            self.feature_names_in_ = np.asarray(input_columns, dtype=object)
        output_columns = getattr(training_X, "columns", None)
        if output_columns is not None:
            self.feature_names_out_ = np.asarray(output_columns, dtype=object)

    def _normalise_output(self, output, *, input_y):
        if not isinstance(output, tuple):
            raise MLWeaveValidationError(
                "@split_step functions must return a tuple. The first X "
                "partition is always treated as training data."
            )
        if (
            len(output) == 2
            and isinstance(output[0], tuple)
            and isinstance(output[1], tuple)
        ):
            X_parts, y_parts = output
        else:
            X_parts = output
            y_parts = None
        if input_y is not None and y_parts is None:
            raise MLWeaveValidationError(
                "The split function received y but did not return y partitions. "
                "Return ((X_train, ...), (y_train, ...))."
            )
        validate_multiplex(X_parts, y_parts, require_multiple=True)
        return X_parts, y_parts

    def _should_execute(self, X) -> bool:
        if self.column_condition == "always":
            return True
        columns = getattr(X, "columns", None)
        if columns is None:
            raise MLWeaveValidationError(
                "Column-conditioned split execution requires an input with a "
                "'columns' attribute, such as a pandas DataFrame."
            )
        present = self.condition_column in columns
        if self.column_condition == "present":
            return present
        if self.column_condition == "absent":
            return not present
        raise MLWeaveValidationError(
            f"Unsupported column condition: {self.column_condition!r}."
        )

    def _validate(self, X) -> None:
        for validation in self.validators:
            if validation.stage in ("fit", "transform", "both"):
                validation.validator(X, *validation.args, **validation.kwargs)

    def _validate_output(self, input_snapshot, output) -> None:
        for validation in self.output_validators:
            validation.validator(
                input_snapshot,
                output,
                *validation.args,
                **validation.kwargs,
            )

    def _print_track_event(self, X, X_parts, y_parts, started) -> None:
        if not self.tracking:
            return
        duration = perf_counter() - started
        print(
            f"[mlweave.track] {self.split_func.__name__} | split | "
            f"input={shape_of(X)} -> X={shape_of(X_parts)}"
            + (f" | y={shape_of(y_parts)}" if y_parts is not None else "")
            + f" | {duration:.6f}s"
        )

    def _print_track_skip(self, X) -> None:
        if self.tracking:
            print(
                f"[mlweave.track] {self.split_func.__name__} | split | "
                f"skipped | input={shape_of(X)}"
            )

    @staticmethod
    def _collect_snapshot_fields(
        validators: Iterable[OutputValidationSpec],
    ) -> frozenset[SnapshotField]:
        fields: set[SnapshotField] = set()
        for validation in validators:
            fields.update(validation.snapshot_fields)
        return frozenset(fields)

    @staticmethod
    def _snapshot_input(X, fields: frozenset[SnapshotField]) -> dict[str, Any]:
        if not fields:
            return {}
        snapshot: dict[str, Any] = {}
        shape = getattr(X, "shape", None)
        if "row_count" in fields:
            if shape is not None and len(shape) >= 1:
                snapshot["row_count"] = int(shape[0])
            else:
                try:
                    snapshot["row_count"] = len(X)
                except TypeError:
                    snapshot["row_count"] = None
        if "column_count" in fields:
            if shape is not None and len(shape) >= 2:
                snapshot["column_count"] = int(shape[1])
            else:
                columns = getattr(X, "columns", None)
                snapshot["column_count"] = (
                    len(columns) if columns is not None else None
                )
        if "columns" in fields:
            columns = getattr(X, "columns", None)
            snapshot["columns"] = tuple(columns) if columns is not None else None
        return snapshot
