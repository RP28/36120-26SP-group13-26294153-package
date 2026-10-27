from __future__ import annotations

import inspect
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from joblib import Memory
from sklearn import config_context
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler

from mlweave.exceptions import MLWeaveConfigurationError, MLWeaveValidationError
from mlweave.pipeline.core.builder import PipelineStepBuilder
from mlweave.pipeline.core.step import MLWeavePipelineStep
from mlweave.pipeline.decorators.conditions import (
    when_column_absent,
    when_column_present,
)
from mlweave.pipeline.decorators.contracts import (
    preserve_columns,
    preserve_rows,
    requires_columns,
)
from mlweave.pipeline.decorators.metadata import description, tag
from mlweave.pipeline.decorators.step import (
    inference_step,
    pipeline_step,
    runtime_inputs,
    split_step,
    stateful_pipeline_step,
)
from mlweave.pipeline.decorators.tracking import track
from mlweave.pipeline.decorators.validation import validate
from mlweave.pipeline.pipeline import Pipeline
from mlweave.pipeline.wrapping import wrap_step


class RecordingEstimator(TransformerMixin, BaseEstimator):
    def fit(self, X, y=None, **params):
        self.X_fit_ = X.copy()
        self.fit_params_ = params
        self.fit_count_ = getattr(self, "fit_count_", 0) + 1
        return self

    def predict(self, X, marker=None):
        self.marker_ = marker
        return np.asarray(X)[:, 0]

    def predict_proba(self, X, marker=None):
        return self.predict(X, marker)

    def predict_log_proba(self, X, marker=None):
        return self.predict(X, marker)

    def decision_function(self, X, marker=None):
        return self.predict(X, marker)

    def transform(self, X):
        return X

    def score_samples(self, X):
        return self.predict(X)

    def score(self, X, y=None, sample_weight=None):
        self.weights_ = sample_weight
        return float(np.asarray(X)[:, 0].sum())

    def fit_predict(self, X, y=None, **params):
        self.fit(X, y, **params)
        return self.predict(X)


def _add_one(data):
    return data + 1

add_one = pipeline_step(_add_one)


@inference_step
@runtime_inputs("offset", "optional")
def add_runtime(data, factor=1, *, offset, optional=0):
    return data * factor + offset + optional


@split_step
def split_halves(data, y):
    cut = len(data) // 2
    return (data.iloc[:cut], data.iloc[cut:]), (y.iloc[:cut], y.iloc[cut:])


@pytest.mark.parametrize("reverse", [False, True])
def test_runtime_decorator_order_and_constructor_arguments(sample_frame, reverse):
    def enrich(frame, column, market_data, sentiment_data=None):
        result = frame.copy()
        result[column] += market_data
        if sentiment_data is not None:
            result[column] += sentiment_data
        return result

    declare = runtime_inputs("market_data", "sentiment_data")
    builder = declare(inference_step(enrich)) if reverse else inference_step(declare(enrich))
    assert isinstance(builder, PipelineStepBuilder)
    step = builder("a")
    assert step.inference_only and not step.stateful
    pipe = Pipeline([("runtime", step), ("model", RecordingEstimator())]).fit(sample_frame)
    assert not hasattr(step, "n_features_in_")
    np.testing.assert_array_equal(pipe.predict(sample_frame, market_data=10), sample_frame.a + 10)
    np.testing.assert_array_equal(
        pipe.predict(sample_frame, market_data=10, sentiment_data=2), sample_frame.a + 12
    )
    with pytest.raises(MLWeaveValidationError, match="market_data"):
        pipe.predict(sample_frame)


@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("names,match", [
    (("data",), "first"),
    (("missing",), "signature"),
    (("market", "market"), "unique"),
    (("X",), "public inference"),
    (("y",), "public inference"),
    (("sample_weight",), "public inference"),
    (("self",), "public inference"),
    (("args",), "keyword-capable"),
    (("kwargs",), "keyword-capable"),
    (("",), "non-empty"),
    ((1,), "non-empty"),
])
def test_runtime_signature_validation_happens_at_instantiation(reverse, names, match):
    def transform(data, market=None, X=None, y=None, sample_weight=None, self=None, *args, **kwargs):
        return data

    config = runtime_inputs(*names)
    builder = config(inference_step(transform)) if reverse else inference_step(config(transform))
    with pytest.raises(MLWeaveConfigurationError, match=match):
        builder()


@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("finalizer", [pipeline_step, stateful_pipeline_step, split_step])
def test_runtime_inputs_require_inference_mode(reverse, finalizer):
    def transform(data, extra=None):
        return data

    config = runtime_inputs("extra")
    builder = config(finalizer(transform)) if reverse else finalizer(config(transform))
    with pytest.raises(MLWeaveConfigurationError, match="only valid with @inference_step"):
        builder()


@pytest.mark.parametrize("finalizer", [pipeline_step, stateful_pipeline_step, split_step])
def test_incompatible_finalizers_keep_decorator_time_checks(finalizer):
    with pytest.raises(MLWeaveConfigurationError, match="cannot also be declared"):
        inference_step(finalizer(lambda data: data))
    with pytest.raises(MLWeaveConfigurationError, match="cannot also be declared"):
        finalizer(inference_step(lambda data: data))


def test_runtime_configuration_errors():
    @runtime_inputs("extra")
    def pending(data, extra):
        return data

    with pytest.raises(MLWeaveConfigurationError, match="only valid with @inference_step"):
        pending()
    with pytest.raises(MLWeaveConfigurationError, match="fixed constructor"):
        add_runtime(offset=1)

    @inference_step
    @runtime_inputs("extra")
    def positional(data, extra, /):
        return data

    with pytest.raises(MLWeaveConfigurationError, match="keyword-capable"):
        positional()
    with pytest.raises(MLWeaveConfigurationError, match="fixed constructor"):
        inference_step(runtime_inputs("extra")(lambda data, extra: data))(1)
    with pytest.raises(MLWeaveConfigurationError, match="first positional"):
        inference_step(lambda *, data: data)()
    with pytest.raises(MLWeaveConfigurationError, match="configuration"):
        inference_step(lambda data: data)(unexpected=1)
    with pytest.raises(MLWeaveConfigurationError, match="stateless"):
        MLWeavePipelineStep(lambda data: data, inference_only=True, stateful=True)
    with pytest.raises(MLWeaveConfigurationError, match="stateless"):
        MLWeavePipelineStep(lambda data: data, inference_only=True, fit_func=lambda X, y: {})
    with pytest.raises(MLWeaveConfigurationError, match="only valid"):
        runtime_inputs("extra")(wrap_step(StandardScaler()))
    with pytest.raises(MLWeaveConfigurationError, match="only.*functions"):
        inference_step(wrap_step(StandardScaler()))


@pytest.mark.parametrize("method", ["fit", "fit_transform", "fit_predict"])
@pytest.mark.parametrize("mode", ["ordinary", "external", "split"])
def test_training_bypasses_every_inference_step(sample_frame, method, mode, capsys):
    def fail(data):
        raise AssertionError("inference transform or validation ran during training")

    @track
    @requires_columns(["missing"])
    @validate(fail, stage="fit")
    @validate(fail, stage="transform")
    @inference_step
    @runtime_inputs("required")
    def inference_only(data, required):
        return fail(data)

    X, y = sample_frame[["a", "b"]], sample_frame.target
    steps = [("before", inference_only()), ("ordinary", add_one()), ("after", inference_only())]
    if mode == "split":
        steps.insert(1, ("split", split_halves()))
    pipe = Pipeline(steps + [("model", RecordingEstimator())])
    if mode == "external":
        X, y = (X.iloc[:3], X.iloc[3:]), (y.iloc[:3], y.iloc[3:])
    result = getattr(pipe, method)(X, y)
    expected = sample_frame[["a", "b"]] + 1
    if mode != "ordinary":
        expected = expected.iloc[:3]
    pd.testing.assert_frame_equal(pipe.named_steps.model.X_fit_, expected)
    if method == "fit_predict":
        np.testing.assert_array_equal(result, expected.a)
    elif method == "fit_transform":
        pd.testing.assert_frame_equal(result[0] if mode != "ordinary" else result, expected)
    assert capsys.readouterr().out == ""
    assert pipe.named_steps.before.inference_only
    assert pipe.named_steps.after.inference_only


@pytest.mark.parametrize("method", ["fit", "fit_transform"])
@pytest.mark.parametrize("multiplex", [False, True])
def test_final_inference_transform_is_skipped_in_training(sample_frame, method, multiplex):
    X, y = sample_frame[["a"]], sample_frame.target
    pipe = Pipeline([("ordinary", add_one()), ("runtime", add_runtime())])
    if multiplex:
        X, y = (X.iloc[:3], X.iloc[3:]), (y.iloc[:3], y.iloc[3:])
    result = getattr(pipe, method)(X, y)
    if method == "fit_transform":
        pd.testing.assert_frame_equal(result[0] if multiplex else result, (X[0] if multiplex else X) + 1)
    transformed = pipe.transform(X, offset=5)
    pd.testing.assert_frame_equal(transformed[0] if multiplex else transformed, (X[0] if multiplex else X) + 6)


@pytest.mark.parametrize("method", [
    "predict", "predict_proba", "predict_log_proba", "decision_function", "score_samples", "transform", "score"
])
@pytest.mark.parametrize("multiplex", [False, True])
def test_all_supported_inference_methods_and_shared_inputs(sample_frame, method, multiplex):
    X, y = sample_frame[["a"]], sample_frame.target
    pipe = Pipeline([
        ("first", add_runtime(2)), ("ordinary", add_one()),
        ("second", add_runtime()), ("model", RecordingEstimator())
    ])
    if multiplex:
        X, y = (X.iloc[:3], X.iloc[3:]), (y.iloc[:3], y.iloc[3:])
    pipe.fit(X, y)
    offsets = (10, 20) if multiplex else 10
    params = {"offset": offsets}
    marker = object()
    if method in ("predict", "predict_proba", "predict_log_proba", "decision_function"):
        params["marker"] = marker
    weights = (np.ones(3), np.ones(3) * 2) if multiplex else np.ones(6)
    if method == "score":
        params.update(y=y, sample_weight=weights)
    output = getattr(pipe, method)(X, **params)
    parts = output if multiplex else (output,)
    for index, result in enumerate(parts):
        expected = (X[index] if multiplex else X) * 2 + 1 + 2 * (offsets[index] if multiplex else offsets)
        if method == "transform":
            pd.testing.assert_frame_equal(result, expected)
        elif method == "score":
            assert result == expected.a.sum()
        else:
            np.testing.assert_array_equal(result, expected.a)
    if "marker" in params:
        assert pipe.named_steps.model.marker_ is marker
    if method == "score":
        np.testing.assert_array_equal(pipe.named_steps.model.weights_, weights[-1] if multiplex else weights)


def test_runtime_values_are_only_forwarded_to_declaring_steps(sample_frame):
    received = []

    @inference_step
    @runtime_inputs("left")
    def left(data, left, **kwargs):
        received.append((left, kwargs))
        return data

    @inference_step
    @runtime_inputs("right")
    def right(data, right, **kwargs):
        received.append((right, kwargs))
        return data

    @inference_step
    def plain(data):
        return data

    pipe = Pipeline([("left", left()), ("plain", plain()), ("right", right()), ("model", RecordingEstimator())])
    pipe.fit(sample_frame).predict(sample_frame, left=1, right=2, marker=3)
    assert received == [(1, {}), (2, {})]
    assert pipe.named_steps.model.marker_ == 3
    with pytest.raises(TypeError, match="unexpected keyword"):
        pipe.predict(sample_frame, left=1, right=2, undeclared=4)
    with pytest.raises(ValueError, match="metadata"):
        pipe.transform(sample_frame, left=1, right=2, undeclared=4)
    with pytest.raises(TypeError, match="declared runtime"):
        pipe.score_samples(sample_frame, left=1, right=2, undeclared=4)


@pytest.mark.parametrize("condition,column,executes", [
    (when_column_present, "a", True), (when_column_present, "missing", False),
    (when_column_absent, "missing", True), (when_column_absent, "a", False),
])
def test_conditions_tracking_validations_and_contracts(sample_frame, capsys, condition, column, executes):
    calls = []

    @condition(column)
    @track
    @preserve_rows
    @preserve_columns
    @requires_columns(["a"])
    @validate(lambda data: calls.append("transform"), stage="transform")
    @validate(lambda data: calls.append("fit"), stage="fit")
    @description("runtime update")
    @tag("inference")
    @inference_step
    def update(data):
        return data + 1

    X = sample_frame[["a"]]
    pipe = Pipeline([("runtime", update()), ("model", RecordingEstimator())]).fit(X)
    assert calls == [] and capsys.readouterr().out == ""
    result = pipe.predict(X)
    np.testing.assert_array_equal(result, X.a + int(executes))
    assert calls == (["transform"] if executes else [])
    output = capsys.readouterr().out
    assert "transform" in output and ("skipped" in output) == (not executes)
    assert pipe.describe()[0]["description"] == "runtime update"
    assert not pipe.get_tracking_disabled_pipeline().named_steps.runtime.tracking


def test_inference_contract_failures_remain_visible(sample_frame):
    @inference_step
    @preserve_rows
    def remove_row(data):
        return data.iloc[1:]

    pipe = Pipeline([("runtime", remove_row()), ("model", RecordingEstimator())]).fit(sample_frame)
    with pytest.raises(MLWeaveValidationError, match="rows"):
        pipe.predict(sample_frame)

    @inference_step
    @requires_columns(["missing"])
    def invalid_input(data):
        return data

    pipe.steps[0] = ("runtime", invalid_input())
    with pytest.raises(MLWeaveValidationError, match="missing"):
        pipe.predict(sample_frame)


@pytest.mark.parametrize("multiplex", [False, True])
def test_post_fit_insert_replace_remove_and_invalid_structure(sample_frame, multiplex):
    X, y = sample_frame[["a"]], sample_frame.target
    pipe = Pipeline([("scale", StandardScaler()), ("model", RecordingEstimator())])
    if multiplex:
        X, y = (X.iloc[:3], X.iloc[3:]), (y.iloc[:3], y.iloc[3:])
    pipe.fit(X, y)
    scaler, model = pipe.named_steps.scale, pipe.named_steps.model
    baseline = pipe.predict(X)
    pipe.steps.insert(-1, ("runtime", add_runtime()))
    inserted = pipe.predict(X, offset=5)
    np.testing.assert_array_equal(inserted[0] if multiplex else inserted, (baseline[0] if multiplex else baseline) + 5)
    pipe.steps[1] = ("runtime", add_runtime(factor=2))
    replaced = pipe.predict(X, offset=0)
    np.testing.assert_array_equal(replaced[0] if multiplex else replaced, (baseline[0] if multiplex else baseline) * 2)
    del pipe.steps[1]
    np.testing.assert_array_equal(pipe.predict(X)[0] if multiplex else pipe.predict(X), baseline[0] if multiplex else baseline)
    assert pipe.named_steps.scale is scaler and pipe.named_steps.model is model
    assert model.fit_count_ == 1
    pipe.steps.insert(-1, ("scale", add_runtime()))
    with pytest.raises(ValueError, match="unique"):
        pipe.predict(X, offset=1)
    pipe.steps[1] = ("bad__name", add_runtime())
    with pytest.raises(ValueError, match="__"):
        pipe.predict(X, offset=1)
    pipe.steps.insert(-1, ("runtime", add_runtime()))
    pipe.steps[1] = ("bad", object())
    with pytest.raises(TypeError, match="intermediate steps"):
        pipe.predict(X)


def test_mutation_preserves_unfitted_errors_and_downstream_schema_errors(sample_frame):
    X, y = sample_frame[["a", "b"]], sample_frame.target
    pipe = Pipeline([("scale", StandardScaler()), ("model", LinearRegression())])
    pipe.steps.insert(-1, ("runtime", add_runtime()))
    with pytest.raises(NotFittedError):
        pipe.predict(X, offset=1)
    pipe.fit(X, y)

    @inference_step
    def drop_feature(data):
        return data[:, :1]

    pipe.steps[1] = ("drop", drop_feature())
    with pytest.raises(ValueError, match="features"):
        pipe.predict(X)
    pipe.steps.insert(0, ("invalid_model_position", LinearRegression()))
    with pytest.raises(TypeError, match="intermediate steps"):
        pipe.predict(X)


@pytest.mark.parametrize("multiplex", [False, True])
def test_stateful_wrapped_clone_and_memory_compatibility(sample_frame, tmp_path, multiplex):
    @stateful_pipeline_step
    def center(data, self):
        return data - self.mean_

    @center.fit
    def fit_center(data, y):
        return {"mean": data.mean()}

    X, y = sample_frame[["a"]], sample_frame.target
    pipe = Pipeline([
        ("center", center()), ("wrapped", wrap_step(StandardScaler())),
        ("runtime", add_runtime(factor=2)), ("model", RecordingEstimator()),
    ])
    if multiplex:
        X, y = (X.iloc[:3], X.iloc[3:]), (y.iloc[:3], y.iloc[3:])
    pipe.fit(X, y)
    np.testing.assert_array_equal(pipe.named_steps.center.mean_, (X[0] if multiplex else X).mean())
    output = pipe.predict(X, offset=10)
    assert len(output) == len(X)
    cloned = clone(pipe)
    assert cloned.named_steps.runtime.runtime_inputs == ("offset", "optional")
    assert cloned.named_steps.runtime.inference_only
    assert cloned.named_steps.runtime.call_kwargs == {"factor": 2}
    with pytest.raises(NotFittedError):
        cloned.predict(X, offset=1)

    # Top-level functions are pickleable by joblib's optional memory cache.
    cached = Pipeline([
        ("ordinary", add_one()), ("runtime", add_runtime(2)), ("model", RecordingEstimator())
    ], memory=Memory(tmp_path, verbose=0))
    original = cached.named_steps.ordinary
    cached.fit(X, y)
    assert cached.named_steps.ordinary is not original
    for _ in range(2):
        result = cached.predict(X, offset=3)
        expected = ((X[0] if multiplex else X) + 1) * 2 + 3
        np.testing.assert_array_equal(result[0] if multiplex else result, expected.a)
    cached.fit(X, y)
    assert cached.named_steps.runtime.inference_only


def test_metadata_routing_and_sklearn_model_selection(sample_frame):
    X, y = sample_frame[["a"]], sample_frame.target
    weights = np.arange(1, len(X) + 1)
    with config_context(enable_metadata_routing=True):
        scaler = StandardScaler().set_fit_request(sample_weight=True).set_transform_request(copy=True)
        model = LinearRegression().set_fit_request(sample_weight=True).set_score_request(sample_weight=True)
        pipe = Pipeline([("scale", scaler), ("runtime", add_runtime()), ("model", model)])
        pipe.fit(X, y, sample_weight=weights)
        prediction = pipe.predict(X, offset=2, copy=True)
        expected = model.predict(scaler.transform(X) + 2)
        np.testing.assert_array_equal(prediction, expected)
        assert pipe.score(X, y, offset=2, sample_weight=weights, copy=True) == model.score(
            scaler.transform(X) + 2, y, sample_weight=weights
        )
        pd_output = pipe[:-1].transform(X, offset=2, copy=True)
        np.testing.assert_array_equal(pd_output, scaler.transform(X) + 2)

    @inference_step
    def identity(data):
        return data

    search = GridSearchCV(Pipeline([
        ("scale", StandardScaler()), ("runtime", identity()), ("model", LogisticRegression())
    ]), {"model__C": [0.1, 1]}, cv=2).fit(X, y)
    assert search.predict(X).shape == (len(X),)


def test_sklearn_transform_input_bypasses_inference_during_fit(sample_frame):
    X, y = sample_frame[["a"]], sample_frame.target

    class MetadataEstimator(BaseEstimator):
        def fit(self, X, y=None, validation=None):
            self.validation_ = validation
            self.fitted_ = True
            return self

    with config_context(enable_metadata_routing=True):
        pipe = Pipeline([
            ("ordinary", add_one()), ("runtime", add_runtime()),
            ("model", MetadataEstimator().set_fit_request(validation=True)),
        ], transform_input=["validation"])
        pipe.fit(X, y, validation=X)
        pd.testing.assert_frame_equal(pipe.named_steps.model.validation_, X + 1)


def test_training_failure_restores_inference_steps(sample_frame):
    class FailingEstimator(BaseEstimator):
        def fit(self, X, y=None):
            raise ValueError("training failed")

    step = add_runtime()
    pipe = Pipeline([("runtime", step), ("model", FailingEstimator())])
    with pytest.raises(ValueError, match="training failed"):
        pipe.fit(sample_frame)
    assert pipe.steps[0] == ("runtime", step)


def test_unavailable_methods_and_pipelines_without_inference(sample_frame):
    X, y = sample_frame[["a"]], sample_frame.target
    pipe = Pipeline([("runtime", add_runtime()), ("model", LinearRegression())]).fit(X, y)
    for method in ("predict_proba", "predict_log_proba", "decision_function", "score_samples", "fit_predict", "transform"):
        assert not hasattr(pipe, method)
    plain = Pipeline([("ordinary", add_one()), ("model", RecordingEstimator())]).fit(X, y)
    np.testing.assert_array_equal(plain.predict(X, marker=3), X.a + 1)
    assert plain.named_steps.model.marker_ == 3
    pd.testing.assert_frame_equal(Pipeline([("identity", "passthrough")]).transform(X), X)
    with pytest.raises(TypeError, match="runtime transform"):
        add_one().transform(X, unexpected=1)


def test_runtime_inputs_with_positional_only_data_and_shared_multiplex_values(sample_frame):
    @inference_step
    @runtime_inputs("extra")
    def positional_data(frame, /, *, extra=1):
        return frame + extra

    X = sample_frame[["a"]]
    pipe = Pipeline([("runtime", positional_data()), ("model", RecordingEstimator())]).fit(X)
    np.testing.assert_array_equal(pipe.predict(X), X.a + 1)
    with pytest.raises(MLWeaveConfigurationError, match="unique"):
        inference_step(runtime_inputs("extra")(runtime_inputs("extra")(lambda frame, extra: frame)))()
    assert list(inspect.signature(positional_data).parameters) == ["frame", "extra"]

    split_pipe = Pipeline([
        ("split", split_halves()), ("runtime", positional_data()), ("model", RecordingEstimator())
    ]).fit(X, sample_frame.target)
    X_parts = (X.iloc[:3], X.iloc[3:])
    result = split_pipe.predict(X_parts, extra=10)
    np.testing.assert_array_equal(result[0], X_parts[0].a + 10)
    np.testing.assert_array_equal(result[1], X_parts[1].a + 10)


@pytest.mark.parametrize("multiplex", [False, True])
def test_runtime_results_are_never_cached(sample_frame, tmp_path, multiplex):
    calls = []

    @inference_step
    def count_calls(data):
        calls.append(1)
        return data

    X, y = sample_frame[["a"]], sample_frame.target
    pipe = Pipeline([
        ("scale", StandardScaler()), ("runtime", count_calls()), ("model", RecordingEstimator())
    ], memory=str(tmp_path))
    if multiplex:
        X, y = (X.iloc[:3], X.iloc[3:]), (y.iloc[:3], y.iloc[3:])
    pipe.fit(X, y)
    assert calls == []
    pipe.predict(X)
    pipe.predict(X)
    assert len(calls) == (4 if multiplex else 2)


@pytest.mark.parametrize("path,section", [
    ("README.md", "## Inference Steps and Runtime Inputs"),
    ("docs/inference.md", None),
])
def test_documented_inference_examples_are_executable(path, section):
    text = (Path(__file__).resolve().parents[1] / path).read_text()
    if section:
        text = text.split(section, 1)[1].split("\n## ", 1)[0]
    namespace = {}
    for block in text.split("```python\n")[1:]:
        code = block.split("```", 1)[0]
        exec(compile(code, path, "exec"), namespace)  # noqa: S102 -- trusted repository examples
