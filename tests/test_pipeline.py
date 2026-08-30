from __future__ import annotations

import importlib
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from scipy import sparse
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.exceptions import NotFittedError
from sklearn.preprocessing import StandardScaler

import mlweave as mlweave_package
import mlweave.mlweave as mlweave_cli
import mlweave.pipeline.pipeline as pipeline_mod
from mlweave.exceptions import MLWeaveConfigurationError, MLWeaveValidationError
from mlweave.mlweave import main
import mlweave.pipeline.decorators.contracts as contract_mod
from mlweave.pipeline.core.builder import PendingPipelineStep, PipelineStepBuilder
from mlweave.pipeline.core.multiplex import (
    is_multiplexed,
    map_partitions,
    partition_count,
    partition_mapping,
    partition_parameter,
    row_count,
    shape_of,
    training_value,
    validate_multiplex,
)
from mlweave.pipeline.core.split import MLWeaveSplitStep
from mlweave.pipeline.core.step import MLWeavePipelineStep
from mlweave.pipeline.core.specs import PipelineStepSpec, WrappedStepSpec
from mlweave.pipeline.core.wrapped import MLWeaveWrappedStep, wrap_step
from mlweave.pipeline.decorators.base import PipelineMetadataDecorator
from mlweave.pipeline.decorators.conditions import when_column_absent, when_column_present
from mlweave.pipeline.decorators.contracts import (
    allowed_values,
    no_missing_input,
    no_missing_output,
    preserve_columns,
    preserve_rows,
    requires_columns,
)
from mlweave.pipeline.decorators.metadata import description, tag
from mlweave.pipeline.decorators.step import pipeline_step, split_step, stateful_pipeline_step
from mlweave.pipeline.decorators.tracking import track
from mlweave.pipeline.decorators.validation import validate
from mlweave.pipeline.pipeline import Pipeline

class AddOneTransformer(TransformerMixin, BaseEstimator):
    def fit(self, X, y=None, **params):
        self.fit_params_ = params
        self.n_features_in_ = X.shape[1]
        self.feature_names_in_ = np.asarray(getattr(X, "columns", []), dtype=object)
        return self

    def transform(self, X, **params):
        out = X.copy()
        out["a"] = out["a"] + params.get("increment", 1)
        return out

    def inverse_transform(self, X, **params):
        out = X.copy()
        out["a"] = out["a"] - params.get("increment", 1)
        return out

    def get_feature_names_out(self, input_features=None):
        return np.asarray(input_features if input_features is not None else self.feature_names_in_, dtype=object)

class RecordingClassifier(ClassifierMixin, BaseEstimator):
    def fit(self, X, y, **params):
        self.X_fit_ = X.copy()
        self.y_fit_ = np.asarray(y)
        self.fit_params_ = params
        self.classes_ = np.unique(y)
        return self

    def predict(self, X, **params):
        self.predict_params_ = params
        return np.full(len(X), self.classes_[-1])

    def predict_proba(self, X, **params):
        probs = np.linspace(0.25, 0.75, len(X))
        return np.column_stack([1 - probs, probs])

    def predict_log_proba(self, X, **params):
        return np.log(self.predict_proba(X, **params))

    def decision_function(self, X, **params):
        return np.arange(len(X), dtype=float)

    def score(self, X, y, sample_weight=None, **params):
        return float(len(X))

class ScoreSamplesEstimator(TransformerMixin, BaseEstimator):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X

    def score_samples(self, X):
        return np.arange(len(X), dtype=float)

class FitPredictClassifier(RecordingClassifier):
    def fit_predict(self, X, y, **params):
        self.fit(X, y, **params)
        return np.full(len(X), self.classes_[0])

class TransformOnlyEstimator(TransformerMixin, BaseEstimator):
    def fit(self, X, y=None, **params):
        self.fit_params_ = params
        return self

    def transform(self, X, **params):
        self.transform_params_ = params
        return X.copy()

class BareTransformEstimator(BaseEstimator):
    def fit(self, X, y=None, **params):
        self.fit_params_ = params
        return self

    def transform(self, X, **params):
        self.transform_params_ = params
        return X.copy()

class NoInverseTransformer(TransformerMixin, BaseEstimator):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X

class NoFeatureNamesTransformer(TransformerMixin, BaseEstimator):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X

def test_multiplex_helpers_validate_shapes_and_partition_params(sample_frame):
    X = (sample_frame.iloc[:2], sample_frame.iloc[2:4])
    y = (sample_frame["target"].iloc[:2], sample_frame["target"].iloc[2:4])
    assert is_multiplexed(X)
    assert partition_count(X) == 2
    assert partition_count(sample_frame) == 1
    assert row_count(sample_frame) == 6
    assert row_count(object()) is None
    assert row_count(SimpleNamespace(shape=("bad",), __len__=lambda: 4)) is None
    assert shape_of(X) == ((2, 4), (2, 4))
    assert training_value(X) is X[0]
    assert training_value("plain") == "plain"
    assert map_partitions(len, X) == (2, 2)
    assert partition_parameter(("left", "right"), 1, 2) == "right"
    assert partition_parameter(("shared", "tuple"), 0, 3) == ("shared", "tuple")
    assert partition_mapping({"alpha": (1, 2), "beta": 3}, 1, 2) == {"alpha": 2, "beta": 3}
    validate_multiplex(X, y, require_multiple=True)

    with pytest.raises(MLWeaveValidationError, match="must be a tuple"):
        validate_multiplex(sample_frame)
    with pytest.raises(MLWeaveValidationError, match="at least 2"):
        validate_multiplex((sample_frame.iloc[:2],))
    with pytest.raises(MLWeaveValidationError, match="None"):
        validate_multiplex((sample_frame.iloc[:2], None))
    with pytest.raises(MLWeaveValidationError, match="y must also be a tuple"):
        validate_multiplex(X, sample_frame["target"])
    with pytest.raises(MLWeaveValidationError, match="same number"):
        validate_multiplex(X, y[:1])
    with pytest.raises(MLWeaveValidationError, match="row counts differ"):
        validate_multiplex(X, (y[0].iloc[:1], y[1]))
    with pytest.raises(MLWeaveValidationError, match="cannot be empty"):
        training_value(())

    class UnknownRows:
        def __len__(self):
            raise TypeError

    validate_multiplex((UnknownRows(), UnknownRows()), (UnknownRows(), UnknownRows()), require_multiple=True)

def test_main_prints_package_version(capsys):
    main()
    assert capsys.readouterr().out.startswith("mlweave ")

def test_version_fallbacks_when_package_metadata_is_missing(monkeypatch, capsys):
    def missing_version(package_name):
        raise mlweave_cli.PackageNotFoundError(package_name)

    monkeypatch.setattr(mlweave_cli, "version", missing_version)
    mlweave_cli.main()
    assert capsys.readouterr().out == "mlweave unknown\n"

    import importlib.metadata

    monkeypatch.setattr(importlib.metadata, "version", missing_version)
    reloaded = importlib.reload(mlweave_package)
    assert reloaded.__version__ == "unknown"

def test_lazy_pipeline_step_collects_metadata_contracts_and_tracking(sample_frame, capsys):
    @description("Select and double columns")
    @tag("feature")
    @pipeline_step
    @requires_columns(["a", "b"])
    @preserve_rows
    @track
    def select_double(frame, columns):
        return frame.loc[:, columns].mul(2)

    builder = select_double
    assert isinstance(builder, PipelineStepBuilder)
    step = builder(["a", "b"])
    assert isinstance(step, MLWeavePipelineStep)
    assert step.description == "Select and double columns"
    assert step.tags == ("feature",)
    result = step.fit_transform(sample_frame)
    pd.testing.assert_frame_equal(result, sample_frame.loc[:, ["a", "b"]].mul(2))
    assert step.get_feature_names_out().tolist() == ["a", "b"]
    assert "[mlweave.track] select_double | fit_transform" in capsys.readouterr().out

def test_pending_pipeline_step_and_finalizer_errors():
    @tag("metadata-only")
    def unfinished(frame):
        return frame

    assert isinstance(unfinished, PendingPipelineStep)
    assert "missing pipeline finalizer" in repr(unfinished)
    with pytest.raises(MLWeaveConfigurationError, match="missing @pipeline_step"):
        unfinished(pd.DataFrame({"a": [1]}))

    unfinalized = PipelineStepBuilder(PipelineStepSpec(lambda X: X))
    assert "unfinalized" in repr(unfinalized)
    with pytest.raises(MLWeaveConfigurationError, match="not finalized"):
        unfinalized()

    with pytest.raises(MLWeaveConfigurationError, match="cannot also be declared"):
        split_step(pipeline_step(lambda X: X))

    @stateful_pipeline_step
    def missing_fit(frame, fitted):
        return frame

    with pytest.raises(MLWeaveConfigurationError, match="has no fit function"):
        missing_fit()

    with pytest.raises(MLWeaveConfigurationError, match="stateful pipeline step"):
        pipeline_step(lambda X: X).fit(lambda X, y: None)

    @stateful_pipeline_step
    def duplicate_fit(frame, fitted):
        return frame

    @duplicate_fit.fit
    def fit_once(frame, y):
        return {}

    with pytest.raises(MLWeaveConfigurationError, match="already registered"):
        duplicate_fit.fit(lambda X, y: {})

def test_pipeline_decorator_base_and_metadata_guards():
    with pytest.raises(MLWeaveConfigurationError, match="Call wrap_step"):
        tag("raw-estimator")(AddOneTransformer())
    with pytest.raises(TypeError, match="can only be applied"):
        tag("bad")(object())
    with pytest.raises(ValueError, match="non-empty"):
        description(" ")
    with pytest.raises(ValueError, match="non-empty"):
        tag("")

    @description("one")
    def described(frame):
        return frame

    with pytest.raises(MLWeaveConfigurationError, match="already has a description"):
        description("two")(described)

    @pipeline_step
    def finalized(frame):
        return frame

    assert pipeline_step(finalized) is finalized

    class MetadataProbe(PipelineMetadataDecorator):
        def configure(self, spec):
            pass

    with pytest.raises(MLWeaveConfigurationError, match="lost its wrapper"):
        MetadataProbe()._preserve_state(lambda X: X, WrappedStepSpec("wrapped"))

def test_validate_decorator_records_stage_args_and_rejects_invalid_stage(sample_frame):
    calls = []

    def validator(frame, expected_column, *, marker):
        calls.append((expected_column, marker, tuple(frame.columns)))

    @pipeline_step
    @validate(validator, "a", marker="fit-only", stage="fit")
    def identity(frame):
        return frame

    result = identity().fit_transform(sample_frame)
    pd.testing.assert_frame_equal(result, sample_frame)
    assert calls == [("a", "fit-only", tuple(sample_frame.columns))]

    with pytest.raises(ValueError, match="stage"):
        validate(validator, stage="predict")

def test_stateful_step_fits_training_partition_and_reuses_state(sample_frame):
    @stateful_pipeline_step
    def center(frame, fitted, column):
        out = frame.copy()
        out[column] = out[column] - fitted.mean_
        return out

    @center.fit
    def fit_center(frame, y, column, sample_weight=None):
        return {"mean": np.average(frame[column], weights=sample_weight)}

    step = center("a")
    X_parts = (sample_frame.iloc[:3].copy(), sample_frame.iloc[3:].copy())
    y_parts = (sample_frame["target"].iloc[:3], sample_frame["target"].iloc[3:])
    result = step.fit_transform(X_parts, y_parts, sample_weight=(np.array([1, 1, 2]), np.array([99, 99, 99])))
    assert step.mean_ == pytest.approx(2.25)
    assert result[0]["a"].tolist() == pytest.approx([-1.25, -0.25, 0.75])
    assert result[1]["a"].tolist() == pytest.approx([1.75, 2.75, 3.75])

    with pytest.raises(NotFittedError):
        center("a").transform(sample_frame)

    @stateful_pipeline_step
    def bad_state(frame, fitted):
        return frame

    @bad_state.fit
    def fit_bad_state(frame, y):
        return 1

    with pytest.raises(TypeError, match="must return None or a dict"):
        bad_state().fit(sample_frame)

    @stateful_pipeline_step
    def fit_returns_none(frame, fitted):
        return frame.assign(a=frame["a"] + 1)

    @fit_returns_none.fit
    def fit_none(frame, y):
        return None

    assert fit_returns_none().fit_transform(sample_frame)["a"].iloc[0] == 2.0

    with pytest.raises(TypeError, match="Stateless"):
        pipeline_step(lambda X: X)().fit(sample_frame, sample_weight=np.ones(len(sample_frame)))

    @stateful_pipeline_step
    def conflicting_fit(frame, fitted, weight=None):
        return frame

    @conflicting_fit.fit
    def fit_conflicting(frame, y, weight=None):
        return {}

    with pytest.raises(TypeError, match="conflict"):
        conflicting_fit(weight=1).fit(sample_frame, weight=2)

def test_pipeline_step_fit_transform_runtime_branches(sample_frame, capsys):
    X_parts = (sample_frame.iloc[:3].copy(), sample_frame.iloc[3:].copy())
    y_parts = (sample_frame["target"].iloc[:3], sample_frame["target"].iloc[3:])

    @pipeline_step
    @track
    def increment(frame):
        return frame.assign(a=frame["a"] + 1)

    step = increment()
    assert step.fit(X_parts, y_parts) is step
    result = step.transform(X_parts)
    assert result[0]["a"].tolist() == [2.0, 3.0, 4.0]
    assert "increment | fit | input" in capsys.readouterr().out
    one_part = step.fit_transform((sample_frame.copy(),), (sample_frame["target"],))
    assert len(one_part) == 1

    @stateful_pipeline_step
    @when_column_present("missing")
    @track
    def skipped_stateful(frame, fitted):
        return frame.assign(a=0)

    @skipped_stateful.fit
    def fit_skipped_stateful(frame, y):
        return {"seen": True}

    skipped = skipped_stateful()
    assert skipped.fit(sample_frame) is skipped
    assert skipped.transform(sample_frame) is sample_frame
    assert "skipped" in capsys.readouterr().out

    @pipeline_step
    @validate(lambda frame: (_ for _ in ()).throw(AssertionError("fit validator skipped")), stage="fit")
    def transform_only_validated(frame):
        return frame

    transform_only_validated().transform(sample_frame)

    @pipeline_step
    @validate(lambda frame: (_ for _ in ()).throw(AssertionError("transform validator skipped")), stage="transform")
    def fit_only_validated(frame):
        return frame

    fit_only_validated().fit(sample_frame)

    calls = []

    @pipeline_step
    @validate(lambda frame: calls.append("transform"), stage="transform")
    def transform_validated(frame):
        return frame

    transform_validated().fit_transform(sample_frame)
    assert calls == ["transform"]
    transform_validated().transform(sample_frame)
    assert calls == ["transform", "transform"]

    @pipeline_step
    @preserve_columns
    def to_numpy(frame):
        return frame.to_numpy()

    fitted = to_numpy().fit(sample_frame)
    assert fitted.get_feature_names_out().tolist() == list(sample_frame.columns)
    assert to_numpy().get_feature_names_out(["x", "y"]).tolist() == ["x", "y"]
    snapshot = MLWeavePipelineStep._snapshot_input(object(), frozenset({"row_count", "column_count", "columns"}))
    assert snapshot == {"row_count": None, "column_count": None, "columns": None}

def test_contract_decorators_raise_for_invalid_inputs(sample_frame):
    @pipeline_step
    @requires_columns(["a", "missing"])
    def identity(frame):
        return frame

    with pytest.raises(MLWeaveValidationError, match="Required columns"):
        identity().fit_transform(sample_frame)

    @pipeline_step
    @no_missing_input
    def passthrough(frame):
        return frame

    with pytest.raises(MLWeaveValidationError, match="input contains missing"):
        passthrough().fit_transform(pd.DataFrame({"a": [1.0, np.nan]}))

    @pipeline_step
    @allowed_values("cat", ["x", "y"])
    def values(frame):
        return frame

    with pytest.raises(MLWeaveValidationError, match="outside the allowed set"):
        values().fit_transform(sample_frame)

    @pipeline_step
    @preserve_rows
    def drop_row(frame):
        return frame.iloc[:1]

    with pytest.raises(MLWeaveValidationError, match="changed the number of rows"):
        drop_row().fit_transform(sample_frame)

    @pipeline_step
    @preserve_columns
    def rename_column(frame):
        return frame.rename(columns={"a": "renamed"})

    with pytest.raises(MLWeaveValidationError, match="changed the columns"):
        rename_column().fit_transform(sample_frame)

    @pipeline_step
    @no_missing_output
    def make_missing(frame):
        out = frame.copy()
        out.loc[out.index[0], "a"] = np.nan
        return out

    with pytest.raises(MLWeaveValidationError, match="output contains missing"):
        make_missing().fit_transform(sample_frame)

def test_contract_decorators_cover_positive_and_fallback_paths(sample_frame):
    @pipeline_step
    @allowed_values("cat", ["x", "y", "z"])
    @preserve_rows
    @preserve_columns
    def reorder_columns(frame):
        return frame[["target", "cat", "b", "a"]]

    result = reorder_columns().fit_transform(sample_frame)
    assert result.columns.tolist() == ["target", "cat", "b", "a"]

    assert no_missing_input.validation_spec().validator(np.array(["ok"], dtype=object)) is None
    assert no_missing_input.validation_spec().validator(np.array(["2020-01-01"], dtype="datetime64[D]")) is None
    with pytest.raises(MLWeaveValidationError, match="missing"):
        no_missing_input.validation_spec().validator(np.array(["NaT"], dtype="datetime64[D]"))

    requires_validator = requires_columns(["a"]).validation_spec().validator
    with pytest.raises(MLWeaveValidationError, match="columns"):
        requires_validator(np.array([[1, 2]]), ("a",))
    with pytest.raises(ValueError, match="at least one"):
        requires_columns([])

    allowed_validator = allowed_values("cat", ["x"]).validation_spec().validator
    with pytest.raises(MLWeaveValidationError, match="columns"):
        allowed_validator(np.array([["x"]]), "cat", ("x",))
    with pytest.raises(MLWeaveValidationError, match="missing"):
        allowed_validator(pd.DataFrame({"other": ["x"]}), "cat", ("x",))
    with pytest.raises(ValueError, match="at least one"):
        allowed_values("cat", [])

    class TinyTable:
        columns = ("cat",)

        def __getitem__(self, key):
            assert key == "cat"
            return ["x", "bad", "bad"]

    with pytest.raises(MLWeaveValidationError, match="outside the allowed set"):
        allowed_validator(TinyTable(), "cat", ("x",))

    preserve_rows_validator = preserve_rows.validation_spec().validator
    with pytest.raises(MLWeaveValidationError, match="could not determine"):
        preserve_rows_validator({"row_count": None}, object())

    preserve_columns_validator = preserve_columns.validation_spec().validator
    preserve_columns_validator({"columns": None, "column_count": 2}, np.ones((2, 2)))
    with pytest.raises(MLWeaveValidationError, match="could not determine"):
        preserve_columns_validator({"columns": None, "column_count": None}, object())
    with pytest.raises(MLWeaveValidationError, match="number of columns"):
        preserve_columns_validator({"columns": None, "column_count": 2}, np.ones((2, 3)))

    assert contract_mod._column_count(SimpleNamespace(columns=("a", "b"))) == 2

    class MissingMask:
        def any(self):
            return np.asarray([False, True])

    class PandasLikeWithoutToNumpy:
        shape = (1, 2)

        @property
        def iloc(self):
            return self

        def __getitem__(self, item):
            return self

        def isna(self):
            return MissingMask()

    assert contract_mod._pandas_like_contains_missing(PandasLikeWithoutToNumpy()) is True
    with pytest.raises(MLWeaveValidationError):
        no_missing_input.validation_spec().validator(np.array([np.nan], dtype=object))

    class PlainInvalidSeries:
        def __init__(self, values):
            self._values = values

        def __iter__(self):
            return iter(self._values)

    class PlainInvalidTable:
        columns = ("cat",)

        def __getitem__(self, key):
            assert key == "cat"
            return PlainInvalidSeries([f"bad_{index}" for index in range(12)])

    with pytest.raises(MLWeaveValidationError, match="showing up to"):
        allowed_validator(PlainInvalidTable(), "cat", ("ok",))

    class SliceableSeries:
        def __init__(self, values):
            self._values = list(values)

        @property
        def iloc(self):
            return self

        def __getitem__(self, item):
            if isinstance(item, slice):
                return SliceableSeries(self._values[item])
            if isinstance(item, (list, np.ndarray)):
                return SliceableSeries([value for value, keep in zip(self._values, item) if keep])
            return self._values[item]

        def __len__(self):
            return len(self._values)

        def isin(self, allowed):
            return np.asarray([value in allowed for value in self._values])

    assert contract_mod._invalid_values_from_pandas_series(SliceableSeries(["ok"]), ("ok",)) == []
    assert contract_mod._invalid_values_from_pandas_series(SliceableSeries(["bad"]), ("ok",)) == ["bad"]
    assert no_missing_input.validation_spec().validator(pd.DataFrame({"a": [1.0]})) is None

def test_contract_helpers_cover_numpy_sparse_and_object_paths():
    assert no_missing_input.validation_spec().validator(np.array([1.0, 2.0])) is None
    with pytest.raises(MLWeaveValidationError):
        no_missing_input.validation_spec().validator(np.array([1.0, np.nan]))
    assert no_missing_input.validation_spec().validator(np.array(["nan"], dtype=np.str_)) is None
    with pytest.raises(MLWeaveValidationError):
        no_missing_input.validation_spec().validator(np.array([None], dtype=object))
    with pytest.raises(MLWeaveValidationError):
        no_missing_input.validation_spec().validator(sparse.csr_matrix([[0.0, np.nan]]))

def test_column_conditions_can_skip_or_execute(sample_frame, capsys):
    @pipeline_step
    @when_column_present("extra")
    @track
    def add_when_present(frame):
        out = frame.copy()
        out["added"] = 1
        return out

    skipped = add_when_present().fit_transform(sample_frame)
    assert skipped is sample_frame
    assert "skipped" in capsys.readouterr().out

    executed = add_when_present().fit_transform(sample_frame.assign(extra=1))
    assert "added" in executed

    @pipeline_step
    @when_column_absent("extra")
    def add_when_absent(frame):
        return frame.assign(absent=1)

    assert "absent" in add_when_absent().fit_transform(sample_frame)
    assert "absent" not in add_when_absent().fit_transform(sample_frame.assign(extra=1))

    with pytest.raises(MLWeaveConfigurationError, match="already has a column"):
        when_column_absent("b")(when_column_present("a")(lambda X: X))
    same_condition = when_column_present("a")(when_column_present("a")(lambda X: X))
    assert isinstance(same_condition, PendingPipelineStep)
    with pytest.raises(MLWeaveValidationError, match="columns"):
        add_when_present().fit_transform(np.array([[1]]))

    invalid = add_when_present()
    invalid.column_condition = "sometimes"
    with pytest.raises(MLWeaveValidationError, match="Unsupported"):
        invalid.fit_transform(sample_frame)

def test_split_step_direct_use_and_validation(sample_frame, capsys):
    @track
    @split_step
    @requires_columns(["a"])
    def first_half(frame, y, cut=3):
        return (frame.iloc[:cut], frame.iloc[cut:]), (y.iloc[:cut], y.iloc[cut:])

    step = first_half(cut=2)
    assert step.fit(sample_frame) is step
    X_parts = step.fit_transform(sample_frame, sample_frame["target"])
    assert isinstance(step, MLWeaveSplitStep)
    assert len(X_parts) == 2
    assert step.split_y_[0].tolist() == [0, 1]
    assert step.transform(sample_frame) is sample_frame
    assert step.inverse_transform(sample_frame) is sample_frame
    assert step.set_output(transform="pandas") is step
    assert step.get_feature_names_out().tolist() == list(sample_frame.columns)
    assert step.get_feature_names_out(["x", "y"]).tolist() == list(sample_frame.columns)
    assert "first_half | split" in capsys.readouterr().out

    @split_step
    def bad_return(frame, y):
        return frame

    with pytest.raises(MLWeaveValidationError, match="must return a tuple"):
        bad_return().fit_transform(sample_frame)

    @split_step
    def missing_y(frame, y):
        return frame.iloc[:2], frame.iloc[2:]

    with pytest.raises(MLWeaveValidationError, match="received y"):
        missing_y().fit_transform(sample_frame, sample_frame["target"])

    with pytest.raises(TypeError, match="conflict"):
        first_half(cut=2).split(sample_frame, sample_frame["target"], cut=3)

    @split_step
    @when_column_present("missing")
    @track
    def conditional_split(frame, y):
        return (frame.iloc[:1], frame.iloc[1:])

    conditional = conditional_split()
    assert conditional.split(sample_frame, None) == (sample_frame, None)
    assert "skipped" in capsys.readouterr().out

    invalid_condition = conditional_split()
    invalid_condition.column_condition = "bad"
    with pytest.raises(MLWeaveValidationError, match="Unsupported"):
        invalid_condition.split(sample_frame, None)

    with pytest.raises(MLWeaveValidationError, match="columns"):
        conditional.split(np.ones((2, 2)), None)

    no_names = MLWeaveSplitStep(lambda X, y: (X[:1], X[1:]))
    no_names.feature_names_in_ = np.asarray(["a", "b"], dtype=object)
    assert no_names.get_feature_names_out().tolist() == ["a", "b"]
    del no_names.feature_names_in_
    assert no_names.get_feature_names_out(["a", "b"]).tolist() == ["a", "b"]

    snapshot = MLWeaveSplitStep._snapshot_input(object(), frozenset({"row_count", "column_count", "columns"}))
    assert snapshot == {"row_count": None, "column_count": None, "columns": None}
    assert MLWeaveSplitStep._snapshot_input(sample_frame, frozenset({"row_count", "column_count"})) == {
        "row_count": 6,
        "column_count": 4,
    }

    @split_step
    @when_column_absent("missing")
    @preserve_rows
    def absent_split(frame, y):
        return frame.copy(), frame.copy()

    assert len(absent_split().fit_transform(sample_frame)) == 2

def test_wrapped_step_supports_decorators_multiplex_and_inverse(sample_frame, capsys):
    wrapped = wrap_step(AddOneTransformer())
    wrapped = requires_columns(["a"])(wrapped)
    wrapped = preserve_rows(wrapped)
    wrapped = preserve_columns(wrapped)
    wrapped = description("wrapped add")(wrapped)
    wrapped = tag("wrapped")(wrapped)
    wrapped = track(wrapped)

    result = wrapped.fit_transform(sample_frame[["a", "b"]])
    assert result["a"].tolist() == [2.0, 3.0, 4.0, 5.0, 6.0, 7.0]
    assert wrapped.description == "wrapped add"
    assert wrapped.tags == ("wrapped",)
    assert wrapped.n_features_in_ == 2
    assert wrapped.feature_names_in_.tolist() == ["a", "b"]
    assert wrapped.get_feature_names_out(["a", "b"]).tolist() == ["a", "b"]
    restored = wrapped.inverse_transform(result)
    pd.testing.assert_frame_equal(restored, sample_frame[["a", "b"]])
    assert "AddOneTransformer | fit_transform" in capsys.readouterr().out

    X_parts = (sample_frame.iloc[:3][["a", "b"]], sample_frame.iloc[3:][["a", "b"]])
    part_result = wrapped.fit_transform(X_parts)
    assert part_result[0]["a"].tolist() == [2.0, 3.0, 4.0]
    assert part_result[1]["a"].tolist() == [5.0, 6.0, 7.0]
    wrapped.fit(X_parts, sample_weight=(np.ones(3), np.ones(3) * 2))
    assert wrapped.estimator_.fit_params_["sample_weight"].tolist() == [1.0, 1.0, 1.0]
    transformed = wrapped.transform(X_parts, increment=(10, 20))
    assert transformed[0]["a"].tolist() == [11.0, 12.0, 13.0]
    assert transformed[1]["a"].tolist() == [24.0, 25.0, 26.0]
    inverse_parts = wrapped.inverse_transform(transformed, increment=(10, 20))
    pd.testing.assert_frame_equal(inverse_parts[0], X_parts[0])

    skipped = when_column_absent("a")(wrap_step(AddOneTransformer()))
    assert skipped.fit_transform(sample_frame[["a"]]).equals(sample_frame[["a"]])
    assert skipped.fit(sample_frame[["a"]]) is skipped
    assert skipped.transform(sample_frame[["a"]]).equals(sample_frame[["a"]])

    present = when_column_present("a")(wrap_step(AddOneTransformer()))
    assert present.fit(sample_frame[["a"]]) is present

def test_wrapped_step_errors_and_fallbacks(sample_frame):
    bare = wrap_step(BareTransformEstimator())
    assert bare.fit_transform(sample_frame[["a"]]).equals(sample_frame[["a"]])

    skipped = track(when_column_absent("a")(wrap_step(AddOneTransformer())))
    skipped.transform(sample_frame[["a"]])

    no_inverse = wrap_step(NoInverseTransformer()).fit(sample_frame[["a"]])
    with pytest.raises(AttributeError, match="inverse_transform"):
        no_inverse.inverse_transform(sample_frame[["a"]])

    no_features = wrap_step(NoFeatureNamesTransformer()).fit(sample_frame[["a"]])
    with pytest.raises(AttributeError, match="get_feature_names_out"):
        no_features.get_feature_names_out()

    invalid_condition = wrap_step(AddOneTransformer())
    invalid_condition.spec.column_condition = "bad"
    with pytest.raises(MLWeaveValidationError, match="Unsupported"):
        invalid_condition.fit(sample_frame[["a"]])
    with pytest.raises(MLWeaveValidationError, match="columns"):
        when_column_present("a")(wrap_step(AddOneTransformer())).fit(np.ones((2, 1)))

    wrapped = preserve_rows(wrap_step(AddOneTransformer()))
    snapshot = wrapped._snapshot_input(object(), frozenset({"row_count", "column_count", "columns"}))
    assert snapshot == {"row_count": None, "column_count": None, "columns": None}

def test_wrap_step_rejects_invalid_estimators_and_missing_methods():
    with pytest.raises(TypeError, match="BaseEstimator"):
        wrap_step(object())

    class NoTransform(BaseEstimator):
        def fit(self, X, y=None):
            return self

    with pytest.raises(MLWeaveConfigurationError, match="transform"):
        wrap_step(NoTransform())

    with pytest.raises(MLWeaveConfigurationError, match="wrap_step"):
        pipeline_step(wrap_step(AddOneTransformer()))

    class NoFit(BaseEstimator):
        def fit(self, X, y=None):
            return self

        fit = None

        def transform(self, X):
            return X

    with pytest.raises(MLWeaveConfigurationError, match="fit"):
        wrap_step(NoFit())

def test_pipeline_extended_fit_predict_score_and_helpers(sample_frame):
    @split_step
    def split_train_eval(frame, y):
        return (frame.iloc[:3][["a", "b"]], frame.iloc[3:][["a", "b"]]), (y.iloc[:3], y.iloc[3:])

    @pipeline_step
    @description("double a")
    @tag("feature")
    def double_a(frame):
        out = frame.copy()
        out["a"] = out["a"] * 2
        return out

    pipe = Pipeline([
        ("split", split_train_eval()),
        ("features", double_a()),
        ("model", RecordingClassifier()),
    ])
    fitted = pipe.fit(sample_frame, sample_frame["target"], model__sample_weight=np.ones(6))
    assert fitted is pipe
    assert pipe.named_steps["model"].X_fit_["a"].tolist() == [2.0, 4.0, 6.0]
    assert hasattr(pipe, "multiplex_X_")
    predictions = pipe.predict(pipe.multiplex_X_)
    assert tuple(len(part) for part in predictions) == (3, 3)
    probabilities = pipe.predict_proba(pipe.multiplex_X_)
    assert probabilities[0].shape == (3, 2)
    scores = pipe.score(pipe.multiplex_X_, pipe.multiplex_y_, sample_weight=(np.ones(3), np.ones(3) * 2))
    assert scores == (3.0, 3.0)
    assert pipe.describe()[1]["description"] == "double a"
    assert pipe.describe()[1]["tags"] == ("feature",)
    excluded = pipe.exclude_steps("features")
    assert excluded.steps[1] == ("features", "passthrough")
    assert pipe.excluding(["features"]).steps[1] == ("features", "passthrough")
    pipe.clear_multiplex_data()
    assert not hasattr(pipe, "multiplex_X_")
    assert pipe.clear_multiplex_data() is pipe
    assert pipe.describe()[0]["type"] == "MLWeaveSplitStep"

    with pytest.raises(ValueError, match="Unknown pipeline"):
        pipe.exclude_steps("missing")
    with pytest.raises(TypeError, match="non-empty"):
        pipe.exclude_steps("")

def test_pipeline_sklearn_fallbacks_and_passthrough_paths(sample_frame):
    X = sample_frame[["a", "b"]]
    y = sample_frame["target"]

    pipe = Pipeline([("scale", StandardScaler()), ("model", FitPredictClassifier())])
    assert pipe.fit(X, y) is pipe
    assert pipe.fit_predict(X, y).shape == (len(X),)
    assert pipe._mlweave_multiplex_fitted_ is False

    transform_pipe = Pipeline([("add", wrap_step(AddOneTransformer())), ("last", TransformOnlyEstimator())])
    transformed = transform_pipe.fit_transform(X, y)
    assert transformed["a"].tolist() == [2.0, 3.0, 4.0, 5.0, 6.0, 7.0]

    passthrough = Pipeline([("split", Pipeline([("inner", "passthrough")])), ("final", "passthrough")])
    assert passthrough.describe()[1]["type"] == "passthrough"

    passthrough_final = Pipeline([("add", wrap_step(AddOneTransformer())), ("final", "passthrough")])
    passthrough_result = passthrough_final.fit_transform(X, y)
    assert passthrough_result["a"].tolist() == transformed["a"].tolist()

    @split_step
    def split_once(frame, target):
        return (frame.iloc[:3], frame.iloc[3:]), (target.iloc[:3], target.iloc[3:])

    with pytest.raises(AttributeError, match="fit_predict"):
        Pipeline([("split", split_once()), ("final", "passthrough")]).fit_predict(X, y)

    split_passthrough = Pipeline([("split", split_once()), ("final", "passthrough")])
    passthrough_parts = split_passthrough.fit_transform(X, y)
    assert tuple(len(part) for part in passthrough_parts) == (3, 3)

    pre_split = Pipeline([
        ("pre", wrap_step(AddOneTransformer())),
        ("split", split_once()),
        ("model", RecordingClassifier()),
    ]).fit(X, y)
    assert pre_split.named_steps["model"].X_fit_["a"].tolist() == [2.0, 3.0, 4.0]

    names = Pipeline._normalise_step_names(((name for name in ["add", "add", "final"]),))
    assert names == ("add", "final")
    assert Pipeline._method_params(SimpleNamespace(transform={"copy": False}), "transform") == {"copy": False}
    params = SimpleNamespace(transform={"alpha": (1, 2)}, fit={"beta": (3, 4)})
    assert Pipeline._partition_step_params(params, 1, 2) == {
        "transform": {"alpha": 2},
        "fit": {"beta": 4},
    }
    cloned = Pipeline._clone_for_pipeline(StandardScaler(), SimpleNamespace(location="/tmp/mlweave-cache"))
    assert cloned is not None

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(pipeline_mod, "_print_elapsed_time", None)
    try:
        context = Pipeline([("model", RecordingClassifier())])._elapsed_context(0)
        assert context.__class__.__name__ == "nullcontext"
    finally:
        monkeypatch.undo()

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(pipeline_mod, "_HAS_LEGACY_XT", False)
    monkeypatch.setattr(
        pipeline_mod.SklearnPipeline,
        "inverse_transform",
        lambda self, target, **params: target,
    )
    try:
        assert Pipeline([("model", RecordingClassifier())])._call_super_inverse("target", None, {}) == "target"
        assert Pipeline([("add", wrap_step(AddOneTransformer()))]).fit(X, y).inverse_transform() is None
    finally:
        monkeypatch.undo()

def test_pipeline_rejects_invalid_split_configurations_and_external_multiplex(sample_frame):
    @split_step
    def split_once(frame, y):
        return (frame.iloc[:3], frame.iloc[3:]), (y.iloc[:3], y.iloc[3:])

    with pytest.raises(MLWeaveConfigurationError, match="one active @split_step"):
        Pipeline([
            ("one", split_once()),
            ("two", split_once()),
            ("model", RecordingClassifier()),
        ]).fit(sample_frame, sample_frame["target"])

    with pytest.raises(MLWeaveConfigurationError, match="final pipeline step"):
        Pipeline([("split", split_once())]).fit(sample_frame, sample_frame["target"])

    X_parts = (sample_frame.iloc[:3][["a"]], sample_frame.iloc[3:][["a"]])
    with pytest.raises(MLWeaveValidationError, match="requires both X and y"):
        Pipeline([
            ("split", split_once()),
            ("model", RecordingClassifier()),
        ]).fit(X_parts, sample_frame["target"])

    with pytest.raises(MLWeaveConfigurationError, match="already multiplexed"):
        Pipeline([
            ("noop", "passthrough"),
            ("split", split_once()),
            ("model", RecordingClassifier()),
        ]).fit(X_parts, (sample_frame["target"].iloc[:3], sample_frame["target"].iloc[3:]))

def test_pipeline_runtime_methods_map_over_multiplexed_input(sample_frame):
    X_parts = (sample_frame.iloc[:3][["a", "b"]], sample_frame.iloc[3:][["a", "b"]])
    y_parts = (sample_frame["target"].iloc[:3], sample_frame["target"].iloc[3:])
    pipe = Pipeline([("model", RecordingClassifier())]).fit(X_parts, y_parts)
    assert pipe.predict(X_parts)[0].shape == (3,)
    assert pipe.predict_log_proba(X_parts)[0].shape == (3, 2)
    assert pipe.decision_function(X_parts)[1].tolist() == [0.0, 1.0, 2.0]
    assert pipe.score(sample_frame[["a", "b"]], sample_frame["target"]) == 6.0
    with pytest.raises(MLWeaveValidationError, match="requires y"):
        pipe.score(X_parts, sample_frame["target"])

    scoring_pipe = Pipeline([("scores", ScoreSamplesEstimator())]).fit(X_parts, y_parts)
    assert scoring_pipe.score_samples(X_parts)[0].tolist() == [0.0, 1.0, 2.0]
    assert scoring_pipe.score_samples(sample_frame[["a", "b"]]).tolist() == list(range(len(sample_frame)))

    inverse_pipe = Pipeline([("add", wrap_step(AddOneTransformer()))]).fit(X_parts, y_parts)
    transformed = inverse_pipe.transform(X_parts)
    restored = inverse_pipe.inverse_transform(transformed)
    pd.testing.assert_frame_equal(restored[0], X_parts[0])
    single = inverse_pipe.inverse_transform(transformed[0])
    pd.testing.assert_frame_equal(single, X_parts[0])

def test_get_feature_names_errors(sample_frame):
    @pipeline_step
    def to_numpy(frame):
        return frame.to_numpy()

    step = to_numpy().fit(sample_frame)
    with pytest.raises(AttributeError, match="Output feature names"):
        step.get_feature_names_out()

    split = MLWeaveSplitStep(lambda X, y: (X.iloc[:1], X.iloc[1:]))
    with pytest.raises(AttributeError, match="input_features"):
        split.get_feature_names_out()

    wrapped = wrap_step(StandardScaler()).fit(sample_frame[["a", "b"]])
    assert wrapped.get_feature_names_out(["a", "b"]).tolist() == ["a", "b"]

def test_pipeline_extended_fit_predict_and_final_transform_paths(sample_frame):
    @split_step
    def split_once(frame, y):
        return (frame.iloc[:3][["a", "b"]], frame.iloc[3:][["a", "b"]]), (y.iloc[:3], y.iloc[3:])

    predictions = Pipeline([
        ("split", split_once()),
        ("model", FitPredictClassifier()),
    ]).fit_predict(sample_frame, sample_frame["target"])
    assert predictions.tolist() == [0, 0, 0]

    transformed = Pipeline([
        ("split", split_once()),
        ("final", BareTransformEstimator()),
    ]).fit_transform(sample_frame, sample_frame["target"])
    assert tuple(len(part) for part in transformed) == (3, 3)

    pipe = Pipeline([("split", split_once()), ("final", "passthrough")])
    with pytest.raises(AttributeError, match="fit_predict"):
        pipe._fit_extended(sample_frame, sample_frame["target"], params={}, caller="fit_predict")

    split_final = Pipeline([("final", split_once())])
    with pytest.raises(MLWeaveConfigurationError, match="final pipeline step"):
        split_final._fit_final_extended(sample_frame, sample_frame["target"], routed_params={"final": {}}, caller="fit")

    direct = Pipeline([("final", BareTransformEstimator())])
    direct.steps = list(direct.steps)
    direct_result = direct._fit_final_extended(
        sample_frame[["a", "b"]],
        sample_frame["target"],
        routed_params={"final": {}},
        caller="fit_transform",
    )
    assert direct_result.equals(sample_frame[["a", "b"]])

    direct_fit = Pipeline([("final", RecordingClassifier())])
    assert direct_fit._fit_extended(
        sample_frame[["a", "b"]],
        sample_frame["target"],
        params={},
        caller="fit",
    ) is direct_fit
    assert not hasattr(direct_fit, "multiplex_X_")

    assert Pipeline._method_params(None, "fit") == {}
    assert Pipeline._partition_step_params(None, 0, 1) == {}
