from __future__ import annotations

from collections.abc import Mapping
from numbers import Integral
from typing import Any, Callable
from sklearn.base import BaseEstimator
from mlweave.exceptions import MLWeaveValidationError

class MLWeaveModelSelectionPolicy(BaseEstimator):
    """Sklearn-compatible callable used as a search ``refit`` policy.
    The wrapped function receives sklearn's ``cv_results_`` mapping followed by
    any arguments configured when the decorated builder is called. It must
    return the integer row index of the candidate that sklearn should refit.
    """
    def __init__(
        self,
        selection_func: Callable[..., Any],
        call_args: tuple[Any, ...] = (),
        call_kwargs: dict[str, Any] | None = None,
    ) -> None:
        self.selection_func = selection_func
        self.call_args = call_args
        self.call_kwargs = call_kwargs

    def __call__(self, cv_results: Mapping[str, Any]) -> int:
        if not isinstance(cv_results, Mapping):
            raise TypeError(
                "A model-selection refit policy expects sklearn cv_results_ "
                "as a mapping."
            )
        if "params" not in cv_results:
            raise MLWeaveValidationError(
                "The supplied cv_results_ mapping does not contain 'params'."
            )
        selected_index = self.selection_func(
            cv_results,
            *self.call_args,
            **dict(self.call_kwargs or {}),
        )
        if isinstance(selected_index, bool) or not isinstance(
            selected_index, Integral
        ):
            raise MLWeaveValidationError(
                f"Model-selection function '{self.selection_func.__name__}' "
                "must return an integer candidate index."
            )
        selected_index = int(selected_index)
        candidate_count = len(cv_results["params"])
        if not 0 <= selected_index < candidate_count:
            raise MLWeaveValidationError(
                f"Model-selection function '{self.selection_func.__name__}' "
                f"returned index {selected_index}, but the search contains "
                f"{candidate_count} candidates."
            )
        self.selected_index_ = selected_index
        return selected_index
