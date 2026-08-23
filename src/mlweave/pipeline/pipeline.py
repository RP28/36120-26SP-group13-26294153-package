from __future__ import annotations

import inspect
from collections.abc import Iterable, Mapping
from contextlib import nullcontext
from typing import Any

from sklearn.base import _fit_context, clone
from sklearn.pipeline import Pipeline as SklearnPipeline
from sklearn.pipeline import _fit_transform_one
from sklearn.utils.metaestimators import available_if
from sklearn.utils.validation import check_memory

from mlweave.exceptions import MLWeaveConfigurationError, MLWeaveValidationError
from mlweave.pipeline.core.multiplex import (
    is_multiplexed,
    partition_mapping,
    shape_of,
    validate_multiplex,
)
from mlweave.pipeline.core.split import MLWeaveSplitStep

try:
    from sklearn.utils._user_interface import _print_elapsed_time
except ImportError:  # pragma: no cover - compatibility fallback
    _print_elapsed_time = None


_HAS_LEGACY_XT = "Xt" in inspect.signature(
    SklearnPipeline.inverse_transform
).parameters


def _final_estimator_has(attr: str):
    """Mirror sklearn's method-availability contract for final estimators."""

    def check(self):
        getattr(self._final_estimator, attr)
        return True

    return check


class Pipeline(SklearnPipeline):
    """sklearn ``Pipeline`` plus opt-in MLWeave multiplexing.

    Ordinary sklearn usage delegates directly to sklearn's implementation.
    Extended multiplex execution is activated only when the pipeline contains
    an ``@split_step`` component or when ``fit`` receives matching X/y tuples.

    In multiplex mode, tuple element zero is always training data. Intermediate
    transformers are fit/fit_transformed on partition zero and then transform
    every remaining partition using the same fitted state. The final estimator
    is fit only on partition zero.
    """

    @_fit_context(prefer_skip_nested_validation=False)
    def fit(self, X, y=None, **params):
        if not self._needs_multiplex_fit(X, y):
            self._clear_multiplex_state()
            return super().fit(X, y, **params)

        self._fit_extended(X, y, params=params, caller="fit")
        return self

    @available_if(lambda self: self._can_fit_transform())
    @_fit_context(prefer_skip_nested_validation=False)
    def fit_transform(self, X, y=None, **params):
        if not self._needs_multiplex_fit(X, y):
            self._clear_multiplex_state()
            return super().fit_transform(X, y, **params)

        return self._fit_extended(
            X,
            y,
            params=params,
            caller="fit_transform",
        )

    @available_if(_final_estimator_has("fit_predict"))
    @_fit_context(prefer_skip_nested_validation=False)
    def fit_predict(self, X, y=None, **params):
        if not self._needs_multiplex_fit(X, y):
            self._clear_multiplex_state()
            return super().fit_predict(X, y, **params)

        return self._fit_extended(
            X,
            y,
            params=params,
            caller="fit_predict",
        )

    @available_if(_final_estimator_has("predict"))
    def predict(self, X, **params):
        return self._map_runtime_method("predict", X, params)

    @available_if(_final_estimator_has("predict_proba"))
    def predict_proba(self, X, **params):
        return self._map_runtime_method("predict_proba", X, params)

    @available_if(_final_estimator_has("predict_log_proba"))
    def predict_log_proba(self, X, **params):
        return self._map_runtime_method("predict_log_proba", X, params)

    @available_if(_final_estimator_has("decision_function"))
    def decision_function(self, X, **params):
        return self._map_runtime_method("decision_function", X, params)

    @available_if(_final_estimator_has("score_samples"))
    def score_samples(self, X):
        if not self._should_map_runtime_tuple(X):
            return super().score_samples(X)
        validate_multiplex(X, require_multiple=False)
        return tuple(SklearnPipeline.score_samples(self, part) for part in X)

    @available_if(lambda self: self._can_transform())
    def transform(self, X, **params):
        return self._map_runtime_method("transform", X, params)

    @available_if(lambda self: self._can_inverse_transform())
    def inverse_transform(self, X=None, *, Xt=None, **params):
        target = X if X is not None else Xt
        if target is None:
            # Delegate argument validation/deprecation behaviour to sklearn.
            return self._call_super_inverse(X, Xt, params)

        if not self._should_map_runtime_tuple(target):
            return self._call_super_inverse(X, Xt, params)

        validate_multiplex(target, require_multiple=False)
        count = len(target)
        return tuple(
            self._call_super_inverse(
                part,
                None,
                partition_mapping(params, index, count),
            )
            for index, part in enumerate(target)
        )

    @available_if(_final_estimator_has("score"))
    def score(self, X, y=None, sample_weight=None, **params):
        if not self._should_map_runtime_tuple(X):
            return super().score(
                X,
                y,
                sample_weight=sample_weight,
                **params,
            )

        y_parts = y if is_multiplexed(y) else None
        if y is not None and y_parts is None:
            raise MLWeaveValidationError(
                "Scoring multiplexed X requires y to be a matching tuple."
            )
        validate_multiplex(X, y_parts, require_multiple=False)
        count = len(X)

        results = []
        for index, X_part in enumerate(X):
            y_part = y[index] if y_parts is not None else y
            weight_part = (
                sample_weight[index]
                if isinstance(sample_weight, tuple) and len(sample_weight) == count
                else sample_weight
            )
            results.append(
                SklearnPipeline.score(
                    self,
                    X_part,
                    y_part,
                    sample_weight=weight_part,
                    **partition_mapping(params, index, count),
                )
            )
        return tuple(results)

    def exclude_steps(self, *step_names: str | Iterable[str]) -> Pipeline:
        """Return a cloned pipeline with selected steps set to ``passthrough``."""
        names = self._normalise_step_names(step_names)
        self._validate_step_names(names)

        result = clone(self)
        if names:
            result.set_params(**{name: "passthrough" for name in names})
        return result

    def excluding(self, *step_names: str | Iterable[str]) -> Pipeline:
        """Alias for :meth:`exclude_steps`."""
        return self.exclude_steps(*step_names)

    def describe(self) -> list[dict[str, Any]]:
        """Return lightweight information about configured pipeline steps."""
        description: list[dict[str, Any]] = []
        for index, (name, step) in enumerate(self.steps):
            if self._is_passthrough(step):
                description.append(
                    {
                        "index": index,
                        "name": name,
                        "type": "passthrough",
                        "description": None,
                        "tags": (),
                    }
                )
                continue

            description.append(
                {
                    "index": index,
                    "name": name,
                    "type": step.__class__.__name__,
                    "description": getattr(step, "description", None),
                    "tags": tuple(getattr(step, "tags", ())),
                }
            )
        return description

    def clear_multiplex_data(self) -> Pipeline:
        """Release retained final partition references after an extended fit."""
        for name in ("multiplex_X_", "multiplex_y_"):
            if hasattr(self, name):
                delattr(self, name)
        return self

    def _fit_extended(self, X, y, *, params: dict[str, Any], caller: str):
        self.steps = list(self.steps)
        self._validate_steps()
        self._validate_split_configuration()

        if is_multiplexed(X):
            if not is_multiplexed(y):
                raise MLWeaveValidationError(
                    "External multiplex fitting requires both X and y tuples. "
                    "Tuple element 0 is training data."
                )
            validate_multiplex(X, y, require_multiple=False)

        routed_params = self._check_method_params(method=caller, props=params)
        memory = check_memory(self.memory)
        fit_transform_cached = memory.cache(_fit_transform_one)

        current_X, current_y = X, y
        split_seen = is_multiplexed(current_X)

        for step_idx, (name, transformer) in enumerate(self.steps[:-1]):
            if self._is_passthrough(transformer):
                with self._elapsed_context(step_idx):
                    continue

            if isinstance(transformer, MLWeaveSplitStep):
                if is_multiplexed(current_X):
                    raise MLWeaveConfigurationError(
                        "A @split_step cannot run after data is already "
                        "multiplexed. Only one split boundary is allowed."
                    )

                fitted_splitter = self._clone_for_pipeline(transformer, memory)
                step_params = self._method_params(
                    routed_params[name],
                    "fit_transform",
                )
                with self._elapsed_context(step_idx):
                    current_X, current_y = fitted_splitter.split(
                        current_X,
                        current_y,
                        **step_params,
                    )
                self.steps[step_idx] = (name, fitted_splitter)
                split_seen = is_multiplexed(current_X)
                continue

            if is_multiplexed(current_X):
                current_X, fitted = self._fit_transform_multiplex_step(
                    transformer=transformer,
                    X_parts=current_X,
                    y_parts=current_y,
                    step_params=routed_params[name],
                    step_idx=step_idx,
                    memory=memory,
                    fit_transform_cached=fit_transform_cached,
                )
            else:
                current_X, fitted = self._fit_transform_single_step(
                    transformer=transformer,
                    X=current_X,
                    y=current_y,
                    step_params=routed_params[name],
                    step_idx=step_idx,
                    memory=memory,
                    fit_transform_cached=fit_transform_cached,
                )
            self.steps[step_idx] = (name, fitted)

        result = self._fit_final_extended(
            current_X,
            current_y,
            routed_params=routed_params,
            caller=caller,
        )

        self._mlweave_multiplex_fitted_ = bool(
            split_seen or is_multiplexed(current_X)
        )
        if caller == "fit_transform" and is_multiplexed(result):
            # The final transformer has produced a newer partition tuple.
            self.multiplex_X_ = result
            self.multiplex_y_ = current_y
        elif is_multiplexed(current_X):
            # Retain only the latest partition objects; no stage history or copies.
            self.multiplex_X_ = current_X
            self.multiplex_y_ = current_y
        else:
            self.clear_multiplex_data()

        return self if caller == "fit" else result

    def _fit_transform_single_step(
        self,
        *,
        transformer,
        X,
        y,
        step_params,
        step_idx,
        memory,
        fit_transform_cached,
    ):
        fitted_input = self._clone_for_pipeline(transformer, memory)
        return fit_transform_cached(
            fitted_input,
            X,
            y,
            weight=None,
            message_clsname="Pipeline",
            message=self._log_message(step_idx),
            params=step_params,
        )

    def _fit_transform_multiplex_step(
        self,
        *,
        transformer,
        X_parts,
        y_parts,
        step_params,
        step_idx,
        memory,
        fit_transform_cached,
    ):
        y_tuple = y_parts if is_multiplexed(y_parts) else None
        validate_multiplex(X_parts, y_tuple, require_multiple=False)
        count = len(X_parts)
        train_y = y_parts[0] if y_tuple is not None else y_parts
        fitted_input = self._clone_for_pipeline(transformer, memory)

        # Time the whole multiplex step once. sklearn's helper normally owns
        # the verbose timing for a single dataset, but here evaluation
        # partitions are transformed after the training fit-transform.
        with self._elapsed_context(step_idx):
            train_X, fitted = fit_transform_cached(
                fitted_input,
                X_parts[0],
                train_y,
                weight=None,
                message_clsname="Pipeline",
                message=None,
                params=self._partition_step_params(step_params, 0, count),
            )

            transform_params = self._method_params(step_params, "transform")
            transformed = [train_X]
            append = transformed.append
            for index in range(1, count):
                append(
                    fitted.transform(
                        X_parts[index],
                        **partition_mapping(transform_params, index, count),
                    )
                )
        return tuple(transformed), fitted

    def _fit_final_extended(self, X, y, *, routed_params, caller: str):
        final_name, final_estimator = self.steps[-1]
        if self._is_passthrough(final_estimator):
            if caller == "fit_predict":
                raise AttributeError(
                    "fit_predict is unavailable when the final estimator is passthrough."
                )
            return X

        if isinstance(final_estimator, MLWeaveSplitStep):
            raise MLWeaveConfigurationError(
                "@split_step cannot be the final pipeline step."
            )

        if is_multiplexed(X):
            y_tuple = y if is_multiplexed(y) else None
            validate_multiplex(X, y_tuple, require_multiple=False)
            count = len(X)
            train_X = X[0]
            train_y = y[0] if y_tuple is not None else y
        else:
            count = 1
            train_X, train_y = X, y

        with self._elapsed_context(len(self.steps) - 1):
            if caller == "fit":
                fit_params = self._method_params(routed_params[final_name], "fit")
                final_estimator.fit(
                    train_X,
                    train_y,
                    **partition_mapping(fit_params, 0, count),
                )
                return self

            if caller == "fit_predict":
                fit_predict_params = self._method_params(
                    routed_params[final_name],
                    "fit_predict",
                )
                return final_estimator.fit_predict(
                    train_X,
                    train_y,
                    **partition_mapping(fit_predict_params, 0, count),
                )

            # fit_transform
            fit_transform_params = self._method_params(
                routed_params[final_name],
                "fit_transform",
            )
            transform_params = self._method_params(
                routed_params[final_name],
                "transform",
            )
            method = getattr(final_estimator, "fit_transform", None)
            if callable(method):
                train_result = method(
                    train_X,
                    train_y,
                    **partition_mapping(fit_transform_params, 0, count),
                )
            else:
                fit_params = self._method_params(routed_params[final_name], "fit")
                final_estimator.fit(
                    train_X,
                    train_y,
                    **partition_mapping(fit_params, 0, count),
                )
                train_result = final_estimator.transform(
                    train_X,
                    **partition_mapping(transform_params, 0, count),
                )

        if not is_multiplexed(X):
            return train_result

        outputs = [train_result]
        append = outputs.append
        for index in range(1, count):
            append(
                final_estimator.transform(
                    X[index],
                    **partition_mapping(transform_params, index, count),
                )
            )
        result = tuple(outputs)
        self.multiplex_X_ = result
        self.multiplex_y_ = y
        return result

    def _map_runtime_method(
        self,
        method_name: str,
        X,
        params: Mapping[str, Any],
    ):
        if not self._should_map_runtime_tuple(X):
            return getattr(SklearnPipeline, method_name)(self, X, **params)

        validate_multiplex(X, require_multiple=False)
        count = len(X)
        method = getattr(SklearnPipeline, method_name)
        return tuple(
            method(
                self,
                part,
                **partition_mapping(params, index, count),
            )
            for index, part in enumerate(X)
        )

    def _call_super_inverse(self, X, Xt, params):
        if _HAS_LEGACY_XT:
            return SklearnPipeline.inverse_transform(
                self,
                X,
                Xt=Xt,
                **params,
            )
        target = X if X is not None else Xt
        return SklearnPipeline.inverse_transform(self, target, **params)

    def _needs_multiplex_fit(self, X, y) -> bool:
        if self._has_active_split_step():
            return True
        return is_multiplexed(X) and is_multiplexed(y)

    def _should_map_runtime_tuple(self, X) -> bool:
        if not is_multiplexed(X):
            return False
        return bool(
            getattr(self, "_mlweave_multiplex_fitted_", False)
            or self._has_active_split_step()
        )

    def _has_active_split_step(self) -> bool:
        return any(isinstance(step, MLWeaveSplitStep) for _, step in self.steps)

    def _validate_split_configuration(self) -> None:
        split_indices = [
            index
            for index, (_, step) in enumerate(self.steps)
            if isinstance(step, MLWeaveSplitStep)
        ]
        if len(split_indices) > 1:
            raise MLWeaveConfigurationError(
                "A Pipeline can contain only one active @split_step."
            )
        if split_indices and split_indices[0] == len(self.steps) - 1:
            raise MLWeaveConfigurationError(
                "@split_step cannot be the final pipeline step."
            )

    @staticmethod
    def _is_passthrough(step) -> bool:
        return step is None or (isinstance(step, str) and step == "passthrough")

    @staticmethod
    def _clone_for_pipeline(transformer, memory):
        if hasattr(memory, "location") and memory.location is None:
            return transformer
        return clone(transformer)

    @staticmethod
    def _method_params(step_params, method: str) -> dict[str, Any]:
        if step_params is None:
            return {}
        if isinstance(step_params, Mapping):
            values = step_params.get(method, {})
        else:
            values = getattr(step_params, method, {})
        return dict(values or {})

    @classmethod
    def _partition_step_params(cls, step_params, index: int, count: int):
        if step_params is None:
            return {}
        items = step_params.items() if isinstance(step_params, Mapping) else []
        if not items:
            # sklearn's Bunch is Mapping, but keep a defensive fallback.
            names = (
                "fit",
                "fit_transform",
                "fit_predict",
                "transform",
                "predict",
                "predict_proba",
                "predict_log_proba",
                "decision_function",
                "score",
                "inverse_transform",
            )
            items = (
                (name, getattr(step_params, name, {}))
                for name in names
                if hasattr(step_params, name)
            )
        return {
            method: partition_mapping(values, index, count)
            for method, values in items
        }

    def _elapsed_context(self, step_idx: int):
        if _print_elapsed_time is None:
            return nullcontext()
        return _print_elapsed_time("Pipeline", self._log_message(step_idx))

    def _clear_multiplex_state(self) -> None:
        self._mlweave_multiplex_fitted_ = False
        self.clear_multiplex_data()

    @staticmethod
    def _normalise_step_names(
        step_names: tuple[str | Iterable[str], ...],
    ) -> tuple[str, ...]:
        if len(step_names) == 1 and not isinstance(step_names[0], str):
            candidate = step_names[0]
            names = tuple(candidate) if isinstance(candidate, Iterable) else (candidate,)
        else:
            names = tuple(step_names)

        if any(not isinstance(name, str) or not name for name in names):
            raise TypeError("Pipeline step names must be non-empty strings.")
        return tuple(dict.fromkeys(names))

    def _validate_step_names(self, names: tuple[str, ...]) -> None:
        available = {name for name, _ in self.steps}
        missing = [name for name in names if name not in available]
        if missing:
            raise ValueError(
                f"Unknown pipeline step name(s): {missing}. "
                f"Available steps: {[name for name, _ in self.steps]}."
            )


__all__ = ["Pipeline"]
