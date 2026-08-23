from __future__ import annotations

from time import perf_counter
from typing import Any, Iterable

from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.utils.validation import check_is_fitted

from mlweave.exceptions import MLWeaveConfigurationError, MLWeaveValidationError
from mlweave.pipeline.core.specs import (
    OutputValidationSpec,
    SnapshotField,
    ValidationSpec,
    WrappedStepSpec,
)


class MLWeaveWrappedStep(TransformerMixin, BaseEstimator):
    """Add mlweave contracts/metadata around an existing sklearn transformer.

    ``estimator`` remains a normal sklearn constructor parameter, so nested
    parameters remain discoverable as ``estimator__...`` for GridSearchCV and
    similar tools. The original estimator is cloned during fitting; fitting the
    wrapper therefore does not mutate the instance supplied to ``wrap_step``.
    """

    def __init__(self, estimator: BaseEstimator, spec: WrappedStepSpec) -> None:
        # Keep constructor arguments unchanged for sklearn.clone().
        self.estimator = estimator
        self.spec = spec

    @property
    def description(self) -> str | None:
        return self.spec.description

    @property
    def tags(self) -> tuple[str, ...]:
        return tuple(sorted(self.spec.tags))

    def fit(self, X, y=None, **fit_params):
        """Validate and fit a fresh clone of the wrapped sklearn transformer."""
        if not self._should_execute(X):
            self._print_track_skip("fit", X)
            return self

        started = perf_counter() if self.spec.tracking else None
        self._validate(X, stage="fit")

        self.estimator_ = clone(self.estimator)
        self.estimator_.fit(X, y, **fit_params)

        self._print_track_event(
            stage="fit",
            input_shape=getattr(X, "shape", None),
            output_shape=None,
            started=started,
        )
        return self

    def transform(self, X):
        """Validate input, delegate transform, then enforce output contracts."""
        if not self._should_execute(X):
            self._print_track_skip("transform", X)
            return X

        started = perf_counter() if self.spec.tracking else None
        check_is_fitted(self, "estimator_")
        self._validate(X, stage="transform")

        snapshot_fields = self._collect_snapshot_fields(self.spec.output_validators)
        input_snapshot = self._snapshot_input(X, snapshot_fields)

        result = self.estimator_.transform(X)
        self._validate_output(input_snapshot, result)

        self._print_track_event(
            stage="transform",
            input_shape=getattr(X, "shape", None),
            output_shape=getattr(result, "shape", None),
            started=started,
        )
        return result

    def fit_transform(self, X, y=None, **fit_params):
        """Delegate sklearn's fit_transform while scanning input validators once.

        Using the wrapped estimator's own ``fit_transform`` preserves estimator-
        specific behaviour (for example cross-fitting transformers) instead of
        forcing all sklearn components through a hand-written fit+transform.
        """
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
            input_shape=getattr(X, "shape", None),
            output_shape=getattr(result, "shape", None),
            started=started,
        )
        return result

    def inverse_transform(self, X):
        """Delegate inverse transformation when supported by the estimator."""
        check_is_fitted(self, "estimator_")
        method = getattr(self.estimator_, "inverse_transform", None)
        if not callable(method):
            raise AttributeError(
                f"{self.estimator_.__class__.__name__} does not provide "
                "inverse_transform()."
            )
        return method(X)

    def get_feature_names_out(self, input_features=None):
        """Delegate sklearn feature-name discovery when available."""
        check_is_fitted(self, "estimator_")
        method = getattr(self.estimator_, "get_feature_names_out", None)
        if not callable(method):
            raise AttributeError(
                f"{self.estimator_.__class__.__name__} does not provide "
                "get_feature_names_out()."
            )
        if input_features is None:
            return method()
        return method(input_features)

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
        if not self.spec.tracking:
            return

        print(
            f"[mlweave.track] {self.spec.display_name} | {stage} | skipped | "
            f"input={getattr(X, 'shape', None)}"
        )

    def _should_execute(self, X) -> bool:
        """Return whether the wrapped sklearn step should execute."""
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

        raise MLWeaveValidationError(
            f"Unsupported column condition: {condition!r}."
        )

    def _validate(self, X, stage: str) -> None:
        for validation in self.spec.validators:
            if validation.stage not in (stage, "both"):
                continue
            self._run_validation(validation, X)

    def _validate_fit_transform(self, X) -> None:
        # Every attached input validator that is relevant to either fit or
        # transform sees the same X. Run each spec once before delegating to the
        # wrapped estimator's potentially specialised fit_transform().
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
    def _snapshot_input(
        X,
        fields: frozenset[SnapshotField],
    ) -> dict[str, Any]:
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
    """Wrap an existing sklearn transformer so mlweave decorators can configure it.

    Examples
    --------
    ``scaler = wrap_step(StandardScaler())``
    ``scaler = no_missing_input(scaler)``
    ``scaler = preserve_rows(scaler)``
    """
    if not isinstance(estimator, BaseEstimator):
        raise TypeError("wrap_step() expects an sklearn BaseEstimator instance.")

    if not callable(getattr(estimator, "fit", None)):
        raise MLWeaveConfigurationError(
            "wrap_step() requires an estimator that provides fit()."
        )

    if not callable(getattr(estimator, "transform", None)):
        raise MLWeaveConfigurationError(
            "wrap_step() currently supports sklearn transformer steps and "
            "therefore requires transform()."
        )

    return MLWeaveWrappedStep(
        estimator=estimator,
        spec=WrappedStepSpec(component_name=estimator.__class__.__name__),
    )
