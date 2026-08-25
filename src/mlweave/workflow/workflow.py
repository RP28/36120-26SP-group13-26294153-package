from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import PredefinedSplit

from mlweave.exceptions import MLWeaveConfigurationError, MLWeaveValidationError
from mlweave.workflow.core.context import WorkflowContext
from mlweave.workflow.core.inference import MLWeaveInferenceStep


class MLWorkflow:
    """Orchestrate preprocessing, model search/selection, and business inference.

    ``preprocessing`` is an MLWeave/sklearn-compatible pipeline that must
    produce multiplexed partitions during ``fit_transform``. Partition zero is
    used for search training and partition one is used as the validation fold.
    Remaining partitions are retained for inference/evaluation context but are
    never exposed to model search.

    ``model_search`` is an sklearn search object such as ``GridSearchCV`` or
    ``RandomizedSearchCV``. Its ``cv`` value is replaced at fit time with a
    ``PredefinedSplit`` built from the preprocessing train/validation boundary.
    The search must refit a winning estimator, typically with an
    ``@model_selection`` policy supplied through ``refit=...``.

    ``inference`` is an optional ``@inference_step`` runtime component. It sees
    a ``WorkflowContext`` containing fitted workflow state, allowing one
    function to create predictions, plots, reports, files, or other business
    outputs.
    """

    def __init__(
        self,
        preprocessing,
        model_search,
        inference: MLWeaveInferenceStep | None = None,
        partition_names: tuple[str, ...] = ("train", "validation", "test"),
    ) -> None:
        self.preprocessing = preprocessing
        self.model_search = model_search
        self.inference = inference
        self.partition_names = partition_names

    def fit(
        self,
        data,
        y=None,
        *,
        preprocessing_params: dict[str, Any] | None = None,
        search_params: dict[str, Any] | None = None,
    ) -> MLWorkflow:
        """Fit preprocessing and search/refit the selected model."""
        preprocessing_params = dict(preprocessing_params or {})
        search_params = dict(search_params or {})

        self.preprocessing_ = clone(self.preprocessing)
        X_parts = self.preprocessing_.fit_transform(
            data,
            y,
            **preprocessing_params,
        )

        if not isinstance(X_parts, tuple) or len(X_parts) < 2:
            raise MLWeaveValidationError(
                "MLWorkflow preprocessing must produce at least train and "
                "validation partitions. Add an MLWeave @split_step or pass a "
                "preprocessing pipeline that returns multiplexed output."
            )

        y_parts = getattr(self.preprocessing_, "multiplex_y_", None)
        if not isinstance(y_parts, tuple) or len(y_parts) != len(X_parts):
            raise MLWeaveValidationError(
                "MLWorkflow could not obtain matching target partitions from "
                "preprocessing.multiplex_y_."
            )

        self.X_parts_ = tuple(X_parts)
        self.y_parts_ = tuple(y_parts)
        self.partition_names_ = self._resolve_partition_names(len(X_parts))
        self.training_data_ = data

        X_search = self._concat_parts(X_parts[0], X_parts[1])
        y_search = self._concat_parts(y_parts[0], y_parts[1])

        validation_fold = np.concatenate(
            [
                np.full(len(X_parts[0]), -1, dtype=int),
                np.zeros(len(X_parts[1]), dtype=int),
            ]
        )
        predefined_split = PredefinedSplit(test_fold=validation_fold)

        self.model_search_ = clone(self.model_search)
        search_parameters = self.model_search_.get_params(deep=False)

        if "cv" not in search_parameters:
            raise MLWeaveConfigurationError(
                "MLWorkflow model_search must expose an sklearn-compatible "
                "'cv' parameter."
            )

        if search_parameters.get("refit", True) is False:
            raise MLWeaveConfigurationError(
                "MLWorkflow requires model_search.refit to produce a final "
                "estimator. Use refit=True, a metric name, or an "
                "@model_selection(...) policy."
            )

        self.model_search_.set_params(cv=predefined_split)
        self.model_search_.fit(X_search, y_search, **search_params)

        if not hasattr(self.model_search_, "best_estimator_"):
            raise MLWeaveValidationError(
                "The fitted model search did not expose best_estimator_. "
                "Ensure the configured search refits a selected candidate."
            )

        self.model_ = self.model_search_.best_estimator_
        self.inference_ = clone(self.inference) if self.inference is not None else None
        return self

    def infer(self, data=None):
        """Run the configured business inference stage.

        ``data`` is optional. When supplied, it is transformed with the fitted
        preprocessing pipeline and exposed as ``context.X_inference`` while the
        original object is retained as ``context.inference_data``.
        """
        if not hasattr(self, "model_"):
            raise MLWeaveValidationError(
                "MLWorkflow must be fitted before inference."
            )

        if self.inference_ is None:
            raise MLWeaveConfigurationError(
                "MLWorkflow has no inference step. Configure one with "
                "inference=@inference_step(...)."
            )

        X_inference = None
        if data is not None:
            X_inference = self.preprocessing_.transform(data)
            if isinstance(X_inference, tuple):
                raise MLWeaveValidationError(
                    "Inference preprocessing returned multiplexed data. A "
                    "fit-time @split_step should be pass-through during "
                    "transform/inference."
                )

        inference_index = self._resolve_inference_index(data, X_inference)

        context = WorkflowContext(
            model=self.model_,
            search=self.model_search_,
            preprocessing=self.preprocessing_,
            training_data=self.training_data_,
            X_parts=self.X_parts_,
            y_parts=self.y_parts_,
            partition_names=self.partition_names_,
            inference_data=data,
            X_inference=X_inference,
            inference_index=inference_index,
        )

        self.context_ = context
        self.inference_result_ = self.inference_.run(context)
        return self.inference_result_

    def run(
        self,
        data,
        y=None,
        *,
        inference_data=None,
        preprocessing_params: dict[str, Any] | None = None,
        search_params: dict[str, Any] | None = None,
    ):
        """Fit the full workflow and immediately execute inference."""
        self.fit(
            data,
            y,
            preprocessing_params=preprocessing_params,
            search_params=search_params,
        )

        if self.inference_ is None:
            return self

        return self.infer(inference_data)

    def _resolve_partition_names(self, partition_count: int) -> tuple[str, ...]:
        if len(self.partition_names) == partition_count:
            names = tuple(self.partition_names)
        elif self.partition_names == ("train", "validation", "test"):
            # Keep convenient conventional names where possible while still
            # supporting workflows with more or fewer than three partitions.
            defaults = ["train", "validation", "test"]
            names = tuple(
                defaults[index] if index < len(defaults) else f"partition_{index}"
                for index in range(partition_count)
            )
        else:
            raise MLWeaveConfigurationError(
                "partition_names must contain exactly one name per fitted "
                f"partition; got {len(self.partition_names)} names for "
                f"{partition_count} partitions."
            )

        if len(set(names)) != len(names):
            raise MLWeaveConfigurationError(
                "partition_names must contain unique values."
            )

        if partition_count >= 2 and names[:2] != ("train", "validation"):
            raise MLWeaveConfigurationError(
                "The first two MLWorkflow partitions must be named 'train' "
                "and 'validation' because they define the search boundary."
            )

        return names


    @staticmethod
    def _resolve_inference_index(data, X_inference):
        if X_inference is None:
            return None

        transformed_index = getattr(X_inference, "index", None)
        if transformed_index is not None:
            return transformed_index

        raw_index = getattr(data, "index", None)
        if raw_index is not None and len(data) == len(X_inference):
            return raw_index

        return None

    @staticmethod
    def _concat_parts(left, right):
        if isinstance(left, (pd.DataFrame, pd.Series)) and isinstance(
            right, type(left)
        ):
            return pd.concat([left, right], axis=0)

        try:
            return np.concatenate([left, right], axis=0)
        except Exception as exc:  # pragma: no cover - defensive error context
            raise MLWeaveValidationError(
                "MLWorkflow could not concatenate train and validation "
                "partitions for model search."
            ) from exc
