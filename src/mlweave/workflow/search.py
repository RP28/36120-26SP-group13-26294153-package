from __future__ import annotations

from typing import Any, Callable

import numpy as np
from sklearn.base import clone


class FunctionalSearchCV:
    """Adapt an external search function to MLWorkflow's search contract.

    ``optimizer(estimator, X_train, y_train, X_validation, y_validation)``
    returns an iterable of {"params": mapping, "scores": mapping}.
    ``selector(candidates)`` returns the integer index of one candidate.

    All metrics are assumed to be on a maximize-is-better scale. The selector
    is application-defined and may implement other conventions.
    """
    def __init__(self, estimator, optimizer: Callable, selector: Callable,
                 cv=None, refit=True):
        self.estimator = estimator
        self.optimizer = optimizer
        self.selector = selector
        self.cv = cv
        self.refit = refit

    def get_params(self, deep=True):
        params = dict(estimator=self.estimator, optimizer=self.optimizer,
                      selector=self.selector, cv=self.cv, refit=self.refit)
        if deep:
            for name, value in self.estimator.get_params(deep=True).items():
                params[f"estimator__{name}"] = value
        return params

    def set_params(self, **params):
        estimator_params = {}
        for name, value in params.items():
            if name.startswith("estimator__"):
                estimator_params[name.split("__", 1)[1]] = value
            elif name in ("estimator", "optimizer", "selector", "cv", "refit"):
                setattr(self, name, value)
            else:
                raise ValueError(f"Unknown parameter: {name}")
        if estimator_params:
            self.estimator.set_params(**estimator_params)
        return self

    @staticmethod
    def _rows(data, indices):
        if hasattr(data, "iloc"):
            return data.iloc[indices]
        return np.asarray(data)[indices]

    def fit(self, X, y, **fit_params):
        if self.cv is None or not hasattr(self.cv, "split"):
            raise ValueError("A CV splitter with exactly one fold is required.")
        folds = list(self.cv.split(X, y))
        if len(folds) != 1:
            raise ValueError("FunctionalSearchCV requires exactly one fold.")
        train_idx, val_idx = folds[0]
        candidates = list(self.optimizer(
            clone(self.estimator),
            self._rows(X, train_idx), self._rows(y, train_idx),
            self._rows(X, val_idx), self._rows(y, val_idx),
        ))
        if not candidates:
            raise ValueError("The optimizer returned no candidates.")
        names = tuple(candidates[0]["scores"].keys())
        if not names:
            raise ValueError("Candidate scores cannot be empty.")
        for candidate in candidates:
            if set(candidate["scores"]) != set(names):
                raise ValueError("All candidates must contain the same scores.")
            if not all(np.isfinite(float(v)) for v in candidate["scores"].values()):
                raise ValueError("Candidate scores must be finite.")
            
        index = self.selector(candidates)
        if not isinstance(index, (int, np.integer)) or not 0 <= index < len(candidates):
            raise ValueError("Selector must return a valid candidate index.")
        self.candidates_ = candidates
        self.best_index_ = int(index)
        self.best_params_ = dict(candidates[index]["params"])
        self.best_score_ = float(next(iter(candidates[index]["scores"].values())))
        self.cv_results_ = {"params": [dict(c["params"]) for c in candidates]}
        for name in names:
            self.cv_results_[f"mean_test_{name}"] = np.array([float(c["scores"][name]) for c in candidates])
        self.best_estimator_ = clone(self.estimator).set_params(**self.best_params_)
        if self.refit:
            self.best_estimator_.fit(X, y, **fit_params)
        return self