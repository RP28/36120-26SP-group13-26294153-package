from __future__ import annotations

import inspect
from time import perf_counter
from typing import Any, Callable, Iterable
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted
from mlweave.exceptions import MLWeaveConfigurationError, MLWeaveValidationError
from mlweave.pipeline.core.multiplex import (
    is_multiplexed,
    partition_mapping,
    shape_of,
    validate_multiplex,
)
from mlweave.pipeline.core.specs import (
    ColumnCondition,
    OutputValidationSpec,
    SnapshotField,
    ValidationSpec,
)

class MLWeavePipelineStep(TransformerMixin, BaseEstimator):
    """Single sklearn-compatible estimator used by MLWeave function steps.
    Besides normal sklearn inputs, the runtime understands MLWeave partition
    tuples. Partition zero is training data; stateful fitting happens only on
    that partition while transformation is applied to every partition.
    """
    def __init__(
        self,
        transform_func: Callable[..., Any],
        fit_func: Callable[..., Any] | None = None,
        call_args: tuple[Any, ...] = (),
        call_kwargs: dict[str, Any] | None = None,
        validators: tuple[ValidationSpec, ...] = (),
        output_validators: tuple[OutputValidationSpec, ...] = (),
        tracking: bool = False,
        description: str | None = None,
        tags: tuple[str, ...] = (),
        stateful: bool = False,
        condition_column: Any | None = None,
        column_condition: ColumnCondition = "always",
        inference_only: bool = False,
        runtime_inputs: tuple[str, ...] | None = None,
    ) -> None:
        self.transform_func = transform_func
        self.fit_func = fit_func
        self.call_args = call_args
        self.call_kwargs = call_kwargs
        self.validators = validators
        self.output_validators = output_validators
        self.tracking = tracking
        self.description = description
        self.tags = tags
        self.stateful = stateful
        self.condition_column = condition_column
        self.column_condition = column_condition
        self.inference_only = inference_only
        self.runtime_inputs = runtime_inputs
        self._validate_runtime_configuration()
        self._snapshot_fields = self._collect_snapshot_fields(output_validators)

    def _validate_runtime_configuration(self) -> None:
        if self.runtime_inputs is not None and not self.inference_only:
            raise MLWeaveConfigurationError(
                "@runtime_inputs is only valid with @inference_step."
            )
        if not self.inference_only:
            return
        if self.stateful or self.fit_func is not None:
            raise MLWeaveConfigurationError("Inference steps must be stateless.")
        signature = inspect.signature(self.transform_func)
        parameters = list(signature.parameters.values())
        if not parameters or parameters[0].kind not in (
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
        ):
            raise MLWeaveConfigurationError(
                "An inference step must accept pipeline data as its first positional parameter."
            )
        names = self.runtime_inputs or ()
        if any(not isinstance(name, str) or not name for name in names):
            raise MLWeaveConfigurationError("Runtime input names must be non-empty strings.")
        if len(set(names)) != len(names):
            raise MLWeaveConfigurationError("Runtime input names must be unique.")
        try:
            bound = signature.bind_partial(None, *self.call_args, **(self.call_kwargs or {}))
        except TypeError as exc:
            raise MLWeaveConfigurationError(f"Invalid inference-step configuration: {exc}") from exc
        for name in names:
            if name not in signature.parameters:
                raise MLWeaveConfigurationError(
                    f"Runtime input {name!r} is missing from the function signature."
                )
            if name == parameters[0].name:
                raise MLWeaveConfigurationError(
                    "Runtime inputs cannot include the first (pipeline data) parameter."
                )
            # These names are already bound by Pipeline's public inference methods.
            if name in {"self", "X", "y", "sample_weight"}:
                raise MLWeaveConfigurationError(
                    f"Runtime input {name!r} conflicts with a public inference parameter."
                )
            if signature.parameters[name].kind not in (
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                inspect.Parameter.KEYWORD_ONLY,
            ):
                raise MLWeaveConfigurationError(
                    f"Runtime input {name!r} must be an explicit keyword-capable parameter."
                )
            if name in bound.arguments:
                raise MLWeaveConfigurationError(
                    f"Runtime input {name!r} is already bound as a fixed constructor argument."
                )

    def _runtime_kwargs(self, values: dict[str, Any]) -> dict[str, Any]:
        parameters = inspect.signature(self.transform_func).parameters
        missing = [
            name for name in self.runtime_inputs or ()
            if name not in values and parameters[name].default is inspect.Parameter.empty
        ]
        if missing:
            raise MLWeaveValidationError(
                f"Inference step {self.transform_func.__name__!r} requires runtime inputs: {missing}."
            )
        return {name: values[name] for name in self.runtime_inputs or () if name in values}

    def fit(self, X, y=None, **fit_params):
        """Fit on one dataset, or on partition zero for multiplexed input."""
        if is_multiplexed(X):
            y_parts = y if is_multiplexed(y) else None
            validate_multiplex(X, y_parts, require_multiple=False)
            count = len(X)
            train_y = y[0] if y_parts is not None else y
            train_params = partition_mapping(fit_params, 0, count)
            return self._fit_single(X[0], train_y, train_params)
        return self._fit_single(X, y, fit_params)

    def transform(self, X, **runtime_values):
        """Transform one dataset or independently transform every partition."""
        if runtime_values and not self.inference_only:
            raise TypeError("Ordinary MLWeave steps do not accept runtime transform inputs.")
        if is_multiplexed(X):
            validate_multiplex(X, require_multiple=False)
            return tuple(
                self._transform_single(part, partition_mapping(runtime_values, index, len(X)))
                for index, part in enumerate(X)
            )
        return self._transform_single(X, runtime_values)

    def fit_transform(self, X, y=None, **fit_params):
        """Fit/transform training data and transform all non-training partitions.
        For ordinary inputs this preserves the existing optimized validation
        path. For multiplexed input, partition zero is fitted exactly once and
        the learned state is reused for the remaining partitions.
        """
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
        if count == 1:
            return (train_result,)
        return (
            train_result,
            *(self._transform_single(part) for part in X[1:]),
        )

    def _fit_single(self, X, y, fit_params: dict[str, Any]):
        if not self._should_execute(X):
            self._print_track_skip("fit", X)
            return self
        started = perf_counter() if self.tracking else None
        self._capture_input_features(X)
        self._validate(X, stage="fit")
        self._fit_core(X, y, fit_params)
        self._print_track_event(
            stage="fit",
            input_shape=shape_of(X),
            output_shape=None,
            started=started,
        )
        return self

    def _transform_single(self, X, runtime_values=None):
        if not self._should_execute(X):
            self._print_track_skip("transform", X)
            return X
        started = perf_counter() if self.tracking else None
        if self.stateful:
            check_is_fitted(self, "_mlweave_is_fitted_")
        self._validate(X, stage="transform")
        kwargs = self._runtime_kwargs(runtime_values or {}) if self.inference_only else {}
        result = self._transform_core(X, kwargs)
        self._capture_output_features(result)
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
        started = perf_counter() if self.tracking else None
        self._capture_input_features(X)
        self._validate_fit_transform_pre_fit(X)
        self._fit_core(X, y, fit_params)
        self._validate_transform_only(X)
        result = self._transform_core(X)
        self._capture_output_features(result)
        self._print_track_event(
            stage="fit_transform",
            input_shape=shape_of(X),
            output_shape=shape_of(result),
            started=started,
        )
        return result

    def _should_execute(self, X) -> bool:
        if self.column_condition == "always":
            return True
        columns = getattr(X, "columns", None)
        if columns is None:
            raise MLWeaveValidationError(
                "Column-conditioned pipeline execution requires an input that "
                "exposes a 'columns' attribute, such as a pandas DataFrame."
            )
        column_present = self.condition_column in columns
        if self.column_condition == "present":
            return column_present
        if self.column_condition == "absent":
            return not column_present
        raise MLWeaveValidationError(
            f"Unsupported column condition: {self.column_condition!r}."
        )

    def _fit_core(self, X, y, fit_params: dict[str, Any]) -> None:
        if not self.stateful:
            if fit_params:
                raise TypeError(
                    "Stateless MLWeave pipeline steps cannot consume fit "
                    f"parameters: {sorted(fit_params)}. Use "
                    "@stateful_pipeline_step when fitting metadata is needed."
                )
            return
        kwargs = dict(self.call_kwargs or {})
        overlap = kwargs.keys() & fit_params.keys()
        if overlap:
            raise TypeError(
                "Configured step arguments conflict with runtime fit parameters: "
                f"{sorted(overlap)}."
            )
        kwargs.update(fit_params)
        state = self.fit_func(
            X,
            y,
            *self.call_args,
            **kwargs,
        )
        if state is not None:
            if not isinstance(state, dict):
                raise TypeError(
                    "A stateful pipeline step's fit function must return None "
                    "or a dict of learned attributes."
                )
            for name, value in state.items():
                learned_name = name if name.endswith("_") else f"{name}_"
                setattr(self, learned_name, value)
        self._mlweave_is_fitted_ = True

    def _transform_core(self, X, runtime_kwargs=None):
        input_snapshot = self._snapshot_input(X, self._snapshot_fields)
        kwargs = dict(self.call_kwargs or {})
        kwargs.update(runtime_kwargs or {})
        if self.stateful:
            result = self.transform_func(X, self, *self.call_args, **kwargs)
        else:
            result = self.transform_func(X, *self.call_args, **kwargs)
        self._validate_output(input_snapshot, result)
        return result

    def get_feature_names_out(self, input_features=None):
        """Return learned output column names when the transform exposes them."""
        output_names = getattr(self, "feature_names_out_", None)
        if output_names is not None:
            return output_names
        preserves_columns = any(
            "columns" in validation.snapshot_fields
            for validation in self.output_validators
        )
        if preserves_columns:
            if input_features is not None:
                return np.asarray(input_features, dtype=object)
            feature_names = getattr(self, "feature_names_in_", None)
            if feature_names is not None:
                return feature_names
        raise AttributeError(
            "Output feature names are unavailable. Return a pandas DataFrame "
            "during fitting or use @preserve_columns for one-to-one transforms."
        )

    def _capture_input_features(self, X) -> None:
        shape = getattr(X, "shape", None)
        if shape is not None and len(shape) >= 2:
            self.n_features_in_ = int(shape[1])
        columns = getattr(X, "columns", None)
        if columns is not None:
            self.feature_names_in_ = np.asarray(columns, dtype=object)

    def _capture_output_features(self, output) -> None:
        columns = getattr(output, "columns", None)
        if columns is not None:
            self.feature_names_out_ = np.asarray(columns, dtype=object)

    def _print_track_event(
        self,
        *,
        stage: str,
        input_shape,
        output_shape,
        started: float | None,
    ) -> None:
        if not self.tracking:
            return
        duration = perf_counter() - started
        name = self.transform_func.__name__
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
        if self.tracking:
            print(
                f"[mlweave.track] {self.transform_func.__name__} | {stage} | "
                f"skipped | input={shape_of(X)}"
            )

    def _validate(self, X, stage: str) -> None:
        for validation in self.validators:
            if validation.stage not in (stage, "both"):
                continue
            self._run_validation(validation, X)

    def _validate_fit_transform_pre_fit(self, X) -> None:
        for validation in self.validators:
            if validation.stage in ("fit", "both"):
                self._run_validation(validation, X)

    def _validate_transform_only(self, X) -> None:
        for validation in self.validators:
            if validation.stage == "transform":
                self._run_validation(validation, X)

    @staticmethod
    def _run_validation(validation: ValidationSpec, X) -> None:
        validation.validator(X, *validation.args, **validation.kwargs)

    def _validate_output(self, input_snapshot, output) -> None:
        for validation in self.output_validators:
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
