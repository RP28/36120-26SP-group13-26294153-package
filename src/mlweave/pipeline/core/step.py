from __future__ import annotations

from time import perf_counter
from typing import Any, Callable, Iterable

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

from mlweave.pipeline.core.specs import (
    OutputValidationSpec,
    SnapshotField,
    ValidationSpec,
)


class MLWeavePipelineStep(TransformerMixin, BaseEstimator):
    """Single sklearn-compatible estimator used by mlweave pipeline steps.

    The runtime is deliberately light: decorators compile to immutable tuples,
    validation is short-circuiting, expensive ``both`` validations run only
    once in ``fit_transform``, and input snapshots are captured only when an
    attached output contract requires them.
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
    ) -> None:
        # Keep sklearn constructor parameters unchanged so clone()/get_params()
        # remain reliable. Derived runtime data is stored only in private attrs.
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

        self._snapshot_fields = self._collect_snapshot_fields(output_validators)

    def fit(self, X, y=None):
        """Fit the step; stateless steps intentionally learn nothing."""
        self._validate(X, stage="fit")
        self._fit_core(X, y)
        return self

    def transform(self, X):
        """Apply the wrapped transform function and output contracts."""
        if self.stateful:
            check_is_fitted(self, "_mlweave_is_fitted_")

        self._validate(X, stage="transform")
        return self._transform_core(X)

    def fit_transform(self, X, y=None, **fit_params):
        """Fit then transform while avoiding duplicate expensive validation.

        sklearn's default ``TransformerMixin.fit_transform`` calls ``fit`` and
        then ``transform``. A validator configured for ``stage='both'`` would
        therefore scan the same training data twice. For large pandas frames
        this override runs those validators once, while still running fit-only
        checks before fitting and transform-only checks before transformation.
        """
        if fit_params:
            # mlweave does not currently consume sklearn metadata-routing fit
            # parameters. Raise explicitly rather than silently dropping them.
            unexpected = ", ".join(sorted(fit_params))
            raise TypeError(
                "MLWeavePipelineStep.fit_transform does not accept extra fit "
                f"parameters yet: {unexpected}."
            )

        self._validate_fit_transform_pre_fit(X)
        self._fit_core(X, y)
        self._validate_transform_only(X)
        return self._transform_core(X)

    def _fit_core(self, X, y=None) -> None:
        started = perf_counter() if self.tracking else None

        if self.stateful:
            state = self.fit_func(
                X,
                y,
                *self.call_args,
                **(self.call_kwargs if self.call_kwargs is not None else {}),
            )

            if state is not None:
                if not isinstance(state, dict):
                    raise TypeError(
                        "A stateful pipeline step's fit function must return "
                        "None or a dict of learned attributes."
                    )

                for name, value in state.items():
                    learned_name = name if name.endswith("_") else f"{name}_"
                    setattr(self, learned_name, value)

            self._mlweave_is_fitted_ = True

        if self.tracking:
            self._record_event(
                stage="fit",
                input_shape=getattr(X, "shape", None),
                output_shape=None,
                duration_seconds=perf_counter() - started,
            )

    def _transform_core(self, X):
        input_snapshot = self._snapshot_input(X, self._snapshot_fields)
        input_shape = getattr(X, "shape", None) if self.tracking else None
        started = perf_counter() if self.tracking else None
        kwargs = self.call_kwargs if self.call_kwargs is not None else {}

        if self.stateful:
            result = self.transform_func(
                X,
                self,
                *self.call_args,
                **kwargs,
            )
        else:
            result = self.transform_func(
                X,
                *self.call_args,
                **kwargs,
            )

        self._validate_output(input_snapshot, result)

        if self.tracking:
            self._record_event(
                stage="transform",
                input_shape=input_shape,
                output_shape=getattr(result, "shape", None),
                duration_seconds=perf_counter() - started,
            )

        return result

    def _validate(self, X, stage: str) -> None:
        for validation in self.validators:
            if validation.stage not in (stage, "both"):
                continue
            self._run_validation(validation, X)

    def _validate_fit_transform_pre_fit(self, X) -> None:
        # Run fit and both validators once before fit. ``both`` is intentionally
        # not repeated again before transform in this combined lifecycle.
        for validation in self.validators:
            if validation.stage in ("fit", "both"):
                self._run_validation(validation, X)

    def _validate_transform_only(self, X) -> None:
        for validation in self.validators:
            if validation.stage == "transform":
                self._run_validation(validation, X)

    @staticmethod
    def _run_validation(validation: ValidationSpec, X) -> None:
        validation.validator(
            X,
            *validation.args,
            **validation.kwargs,
        )

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

    def _record_event(
        self,
        *,
        stage: str,
        input_shape,
        output_shape,
        duration_seconds: float,
    ) -> None:
        if not hasattr(self, "history_"):
            self.history_ = []

        self.history_.append(
            {
                "stage": stage,
                "input_shape": input_shape,
                "output_shape": output_shape,
                "duration_seconds": duration_seconds,
            }
        )
