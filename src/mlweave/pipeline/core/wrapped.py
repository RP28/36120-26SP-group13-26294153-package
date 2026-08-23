from __future__ import annotations

from time import perf_counter
from typing import Any, Iterable

from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.utils.validation import check_is_fitted

from mlweave.exceptions import MLWeaveConfigurationError, MLWeaveValidationError
from mlweave.pipeline.core.multiplex import (
    is_multiplexed,
    partition_mapping,
    shape_of,
    validate_multiplex,
)
from mlweave.pipeline.core.specs import (
    OutputValidationSpec,
    SnapshotField,
    ValidationSpec,
    WrappedStepSpec,
)


class MLWeaveWrappedStep(TransformerMixin, BaseEstimator):
    """Add MLWeave contracts/metadata around an existing sklearn transformer.

    In multiplex mode the wrapped estimator is fitted only on partition zero
    and that fitted instance transforms every remaining partition.
    """

    def __init__(self, estimator: BaseEstimator, spec: WrappedStepSpec) -> None:
        self.estimator = estimator
        self.spec = spec

    @property
    def n_features_in_(self):
        """Number of input features seen by the fitted wrapped estimator."""
        check_is_fitted(self, "estimator_")
        return self.estimator_.n_features_in_

    @property
    def feature_names_in_(self):
        """Input feature names exposed by the fitted wrapped estimator."""
        check_is_fitted(self, "estimator_")
        return self.estimator_.feature_names_in_

    @property
    def description(self) -> str | None:
        return self.spec.description

    @property
    def tags(self) -> tuple[str, ...]:
        return tuple(sorted(self.spec.tags))

    def fit(self, X, y=None, **fit_params):
        if is_multiplexed(X):
            y_parts = y if is_multiplexed(y) else None
            validate_multiplex(X, y_parts, require_multiple=False)
            count = len(X)
            train_y = y[0] if y_parts is not None else y
            return self._fit_single(
                X[0],
                train_y,
                partition_mapping(fit_params, 0, count),
            )
        return self._fit_single(X, y, fit_params)

    def transform(self, X, **transform_params):
        if is_multiplexed(X):
            validate_multiplex(X, require_multiple=False)
            count = len(X)
            return tuple(
                self._transform_single(
                    part,
                    partition_mapping(transform_params, index, count),
                )
                for index, part in enumerate(X)
            )
        return self._transform_single(X, transform_params)

    def fit_transform(self, X, y=None, **fit_params):
        if not is_multiplexed(X):
            return self._fit_transform_single(X, y, fit_params)

        y_parts = y if is_multiplexed(y) else None
        validate_multiplex(X, y_parts, require_multiple=False)
        count = len(X)
        train_y = y[0] if y_parts is not None else y
        train_result = self._fit_transform_single(
            X[0],
            train_y,
            partition_mapping(fit_params, 0, count),
        )
        return (
            train_result,
            *(
                self._transform_single(part, {})
                for part in X[1:]
            ),
        )

    def _fit_single(self, X, y, fit_params: dict[str, Any]):
        if not self._should_execute(X):
            self._print_track_skip("fit", X)
            return self

        started = perf_counter() if self.spec.tracking else None
        self._validate(X, stage="fit")
        self.estimator_ = clone(self.estimator)
        self.estimator_.fit(X, y, **fit_params)
        self._print_track_event(
            stage="fit",
            input_shape=shape_of(X),
            output_shape=None,
            started=started,
        )
        return self

    def _transform_single(self, X, transform_params: dict[str, Any]):
        if not self._should_execute(X):
            self._print_track_skip("transform", X)
            return X

        started = perf_counter() if self.spec.tracking else None
        check_is_fitted(self, "estimator_")
        self._validate(X, stage="transform")
        snapshot_fields = self._collect_snapshot_fields(self.spec.output_validators)
        input_snapshot = self._snapshot_input(X, snapshot_fields)

        result = self.estimator_.transform(X, **transform_params)
        self._validate_output(input_snapshot, result)
        self._print_track_event(
            stage="transform",
            input_shape=shape_of(X),
            output_shape=shape_of(result),
            started=started,
        )
        return result

    def _fit_transform_single(self, X, y, fit_params: dict[str, Any]):
        if not self._should_execute(X):
            self._print_track_skip("fit_transform", X)
            return X

        started = perf_counter() if self.spec.tracking else None
        self._validate_fit_transform(X)
        snapshot_fields = self._collect_snapshot_fields(self.spec.output_validators)
        input_snapshot = self._snapshot_input(X, snapshot_fields)

        self.estimator_ = clone(self.estimator)
        fit_transform = getattr(self.estimator_, "fit_transform", None)
        if callable(fit_transform):
            result = fit_transform(X, y, **fit_params)
        else:
            self.estimator_.fit(X, y, **fit_params)
            result = self.estimator_.transform(X)

        self._validate_output(input_snapshot, result)
        self._print_track_event(
            stage="fit_transform",
            input_shape=shape_of(X),
            output_shape=shape_of(result),
            started=started,
        )
        return result

    def inverse_transform(self, X, **params):
        check_is_fitted(self, "estimator_")
        method = getattr(self.estimator_, "inverse_transform", None)
        if not callable(method):
            raise AttributeError(
                f"{self.estimator_.__class__.__name__} does not provide "
                "inverse_transform()."
            )
        if is_multiplexed(X):
            validate_multiplex(X, require_multiple=False)
            count = len(X)
            return tuple(
                method(part, **partition_mapping(params, index, count))
                for index, part in enumerate(X)
            )
        return method(X, **params)

    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self, "estimator_")
        method = getattr(self.estimator_, "get_feature_names_out", None)
        if not callable(method):
            raise AttributeError(
                f"{self.estimator_.__class__.__name__} does not provide "
                "get_feature_names_out()."
            )
        return method() if input_features is None else method(input_features)

    def _print_track_event(
        self,
        *,
        stage: str,
        input_shape,
        output_shape,
        started: float | None,
    ) -> None:
        if not self.spec.tracking:
            return

        duration = perf_counter() - started
        name = self.spec.display_name
        if output_shape is None:
            print(
                f"[mlweave.track] {name} | {stage} | "
                f"input={input_shape} | {duration:.6f}s"
            )
            return
        print(
            f"[mlweave.track] {name} | {stage} | "
            f"input={input_shape} -> output={output_shape} | {duration:.6f}s"
        )

    def _print_track_skip(self, stage: str, X) -> None:
        if self.spec.tracking:
            print(
                f"[mlweave.track] {self.spec.display_name} | {stage} | skipped | "
                f"input={shape_of(X)}"
            )

    def _should_execute(self, X) -> bool:
        condition = self.spec.column_condition
        if condition == "always":
            return True

        columns = getattr(X, "columns", None)
        if columns is None:
            raise MLWeaveValidationError(
                "Column-conditioned pipeline execution requires an input that "
                "exposes a 'columns' attribute, such as a pandas DataFrame."
            )

        column_present = self.spec.condition_column in columns
        if condition == "present":
            return column_present
        if condition == "absent":
            return not column_present
        raise MLWeaveValidationError(f"Unsupported column condition: {condition!r}.")

    def _validate(self, X, stage: str) -> None:
        for validation in self.spec.validators:
            if validation.stage in (stage, "both"):
                self._run_validation(validation, X)

    def _validate_fit_transform(self, X) -> None:
        for validation in self.spec.validators:
            if validation.stage in ("fit", "transform", "both"):
                self._run_validation(validation, X)

    @staticmethod
    def _run_validation(validation: ValidationSpec, X) -> None:
        validation.validator(X, *validation.args, **validation.kwargs)

    def _validate_output(self, input_snapshot, output) -> None:
        for validation in self.spec.output_validators:
            validation.validator(
                input_snapshot,
                output,
                *validation.args,
                **validation.kwargs,
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


def wrap_step(estimator: BaseEstimator) -> MLWeaveWrappedStep:
    """Wrap an sklearn transformer so MLWeave decorators can configure it."""
    if not isinstance(estimator, BaseEstimator):
        raise TypeError("wrap_step() expects an sklearn BaseEstimator instance.")
    if not callable(getattr(estimator, "fit", None)):
        raise MLWeaveConfigurationError(
            "wrap_step() requires an estimator that provides fit()."
        )
    if not callable(getattr(estimator, "transform", None)):
        raise MLWeaveConfigurationError(
            "wrap_step() is for sklearn transformer steps and requires transform()."
        )

    return MLWeaveWrappedStep(
        estimator=estimator,
        spec=WrappedStepSpec(component_name=estimator.__class__.__name__),
    )
