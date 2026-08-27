from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import GridSearchCV

from mlweave.exceptions import MLWeaveConfigurationError, MLWeaveValidationError
from mlweave.pipeline.decorators.step import pipeline_step, split_step
from mlweave.pipeline.pipeline import Pipeline
from mlweave.workflow.core.builder import WorkflowStepBuilder
from mlweave.workflow.core.context import WorkflowContext
from mlweave.workflow.core.specs import WorkflowStepSpec
from mlweave.workflow.decorators.base import BaseWorkflowDecorator
from mlweave.workflow.decorators.step import inference_step, model_selection
from mlweave.workflow.workflow import MLWorkflow

class ThresholdClassifier(ClassifierMixin, BaseEstimator):
    def __init__(self, threshold: float = 0.0) -> None:
        self.threshold = threshold

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        self.mean_ = float(np.mean(X["a"]))
        return self

    def predict(self, X):
        return (X["a"].to_numpy() >= self.threshold).astype(int)

    def score(self, X, y):
        return float(np.mean(self.predict(X) == np.asarray(y)))

def workflow_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "a": [0.0, 0.2, 0.8, 1.1, 1.4, 1.8],
            "target": [0, 0, 1, 1, 1, 1],
        },
        index=[100, 101, 102, 103, 104, 105],
    )

@split_step
def workflow_split(frame, y):
    del y
    X = frame[["a"]]
    target = frame["target"]
    return (X.iloc[:4], X.iloc[4:5], X.iloc[5:]), (target.iloc[:4], target.iloc[4:5], target.iloc[5:])

@pipeline_step
def workflow_identity(frame):
    return frame

class SearchWithoutBestIndex(BaseEstimator):
    def __init__(self, estimator=None, cv=None, refit=True):
        self.estimator = estimator
        self.cv = cv
        self.refit = refit

    def fit(self, X, y, **params):
        self.cv_results_ = {"params": [{}]}
        return self

class SearchWithOddMetrics(BaseEstimator):
    def __init__(self, estimator=None, cv=None, refit=True):
        self.estimator = ThresholdClassifier() if estimator is None else estimator
        self.cv = cv
        self.refit = refit

    def fit(self, X, y, **params):
        self.best_index_ = 0
        self.best_params_ = {"threshold": 0.5}
        self.cv_results_ = {
            "params": [{"threshold": 0.5}],
            "mean_train_score": [0.75],
            "mean_train_bad": [object()],
            "mean_test_score": [0.5],
            "mean_test_bad": ["bad"],
        }
        self.best_estimator_ = self.estimator.set_params(**self.best_params_).fit(X, y)
        return self

def test_workflow_run_builds_context_and_executes_inference():
    @inference_step
    def summarize(context, label):
        return {
            "label": label,
            "train_rows": len(context.X_train),
            "validation_rows": len(context.X_validation),
            "test_rows": len(context.X_test),
            "inference_index": tuple(context.inference_index),
            "training_metrics": context.training_metrics,
            "validation_metrics": context.validation_metrics,
            "predictions": context.model.predict(context.X_inference).tolist(),
        }

    preprocessing = Pipeline([
        ("split", workflow_split()),
        ("identity", workflow_identity()),
    ])
    search = GridSearchCV(
        ThresholdClassifier(),
        param_grid={"threshold": [0.5, 1.0]},
        cv=None,
        refit=True,
        return_train_score=True,
    )
    workflow = MLWorkflow(
        preprocessing=preprocessing,
        model_search=search,
        inference=summarize("scouting"),
    )
    inference_data = pd.DataFrame({"a": [0.1, 2.0]}, index=[200, 201])
    result = workflow.run(workflow_frame(), inference_data=inference_data)
    assert result["label"] == "scouting"
    assert result["train_rows"] == 4
    assert result["validation_rows"] == 1
    assert result["test_rows"] == 1
    assert result["inference_index"] == (200, 201)
    assert set(result["training_metrics"]) == {"score"}
    assert set(result["validation_metrics"]) == {"score"}
    assert workflow.context_.partition("test")[0].index.tolist() == [105]
    assert workflow.context_.y_train.tolist() == [0, 0, 1, 1]
    assert workflow.context_.y_validation.tolist() == [1]
    assert workflow.context_.y_test.tolist() == [1]

def test_workflow_replacing_and_infer_errors():
    workflow = MLWorkflow("pre", "search")
    replaced = workflow.replacing(preprocessing="new", partition_names=("train", "validation"))
    assert replaced.preprocessing == "new"
    assert replaced.model_search == "search"
    assert replaced.partition_names == ("train", "validation")

    with pytest.raises(MLWeaveValidationError, match="fitted"):
        workflow.infer()

    fitted_without_inference = MLWorkflow.__new__(MLWorkflow)
    fitted_without_inference.model_ = object()
    fitted_without_inference.inference_ = None
    with pytest.raises(MLWeaveConfigurationError, match="no inference step"):
        fitted_without_inference.infer()

    context = WorkflowContext(
        model=None,
        search=None,
        preprocessing=None,
        training_data=None,
        X_parts=(pd.DataFrame({"a": [1]}),),
        y_parts=(pd.Series([1]),),
        partition_names=("train",),
        best_index=0,
        best_params={},
        training_metrics={},
        validation_metrics={},
    )
    with pytest.raises(MLWeaveValidationError, match="Unknown workflow partition"):
        context.partition("missing")

    class TupleTransformPreprocessing:
        def transform(self, data):
            return (data,)

    @inference_step
    def read(context):
        return context.best_index

    fitted = MLWorkflow.__new__(MLWorkflow)
    fitted.model_ = object()
    fitted.model_search_ = object()
    fitted.preprocessing_ = TupleTransformPreprocessing()
    fitted.training_data_ = None
    fitted.X_parts_ = (pd.DataFrame({"a": [1]}), pd.DataFrame({"a": [2]}))
    fitted.y_parts_ = (pd.Series([0]), pd.Series([1]))
    fitted.partition_names_ = ("train", "validation")
    fitted.best_index_ = 0
    fitted.best_params_ = {}
    fitted.training_metrics_ = {}
    fitted.validation_metrics_ = {}
    fitted.inference_ = read()
    with pytest.raises(MLWeaveValidationError, match="multiplexed data"):
        fitted.infer(pd.DataFrame({"a": [1]}))

def test_workflow_validation_errors(sample_frame):
    @pipeline_step
    def no_split(frame):
        return frame[["a"]]

    search = GridSearchCV(ThresholdClassifier(), {"threshold": [0.0]}, cv=2)
    with pytest.raises(MLWeaveValidationError, match="must produce at least"):
        MLWorkflow(Pipeline([("no_split", no_split())]), search).fit(sample_frame)

    class NoCvSearch(BaseEstimator):
        def fit(self, X, y):
            return self

    valid_preprocessing = Pipeline([
        ("split", workflow_split()),
        ("identity", workflow_identity()),
    ])

    with pytest.raises(MLWeaveConfigurationError, match="'cv' parameter"):
        MLWorkflow(valid_preprocessing, NoCvSearch()).fit(workflow_frame())

    bad_refit = GridSearchCV(ThresholdClassifier(), {"threshold": [0.0]}, cv=2, refit=False)
    with pytest.raises(MLWeaveConfigurationError, match="requires model_search.refit"):
        MLWorkflow(valid_preprocessing, bad_refit).fit(workflow_frame())

    with pytest.raises(MLWeaveConfigurationError, match="unique"):
        MLWorkflow(
            valid_preprocessing,
            search,
            partition_names=("train", "train", "test"),
        ).fit(workflow_frame())

    with pytest.raises(MLWeaveConfigurationError, match="first two"):
        MLWorkflow(
            valid_preprocessing,
            search,
            partition_names=("train", "test", "validation"),
        ).fit(workflow_frame())

    class MissingTargets(BaseEstimator):
        def fit_transform(self, data, y=None):
            return (data.iloc[:2][["a"]], data.iloc[2:][["a"]])

    with pytest.raises(MLWeaveValidationError, match="matching target"):
        MLWorkflow(MissingTargets(), search).fit(workflow_frame())

    with pytest.raises(MLWeaveValidationError, match="best_index"):
        MLWorkflow(
            valid_preprocessing,
            SearchWithoutBestIndex(estimator=ThresholdClassifier(), cv=2),
        ).fit(workflow_frame())

    with pytest.raises(MLWeaveConfigurationError, match="exactly one name"):
        MLWorkflow(
            valid_preprocessing,
            search,
            partition_names=("train", "validation"),
        ).fit(workflow_frame())

def test_workflow_run_without_inference_and_helper_branches():
    preprocessing = Pipeline([
        ("split", workflow_split()),
        ("identity", workflow_identity()),
    ])
    workflow = MLWorkflow(
        preprocessing=preprocessing,
        model_search=SearchWithOddMetrics(cv=2),
    )
    result = workflow.run(workflow_frame())
    assert result is workflow
    assert workflow.training_metrics_ == {"score": 0.75}
    assert workflow.validation_metrics_ == {"score": 0.5}
    assert MLWorkflow._resolve_inference_index(pd.DataFrame({"a": [1]}, index=[9]), np.array([[1]]))[0] == 9
    assert MLWorkflow._resolve_inference_index(None, None) is None
    assert MLWorkflow._resolve_inference_index(pd.DataFrame({"a": [1, 2]}), np.array([[1]])) is None
    assert MLWorkflow._concat_parts(np.array([[1]]), np.array([[2]])).tolist() == [[1], [2]]
    with pytest.raises(MLWeaveValidationError, match="could not concatenate"):
        MLWorkflow._concat_parts(object(), object())

    @split_step
    def split_four(frame, y):
        X = frame[["a"]]
        target = frame["target"]
        return (
            X.iloc[:2],
            X.iloc[2:4],
            X.iloc[4:5],
            X.iloc[5:],
        ), (
            target.iloc[:2],
            target.iloc[2:4],
            target.iloc[4:5],
            target.iloc[5:],
        )

    four = MLWorkflow(
        Pipeline([("split", split_four()), ("identity", workflow_identity())]),
        SearchWithOddMetrics(cv=2),
    ).fit(workflow_frame())
    assert four.partition_names_ == ("train", "validation", "test", "partition_3")

def test_workflow_builder_and_decorator_guards():
    builder = WorkflowStepBuilder(WorkflowStepSpec(lambda results: 0))
    assert "unfinalized" in repr(builder)
    with pytest.raises(MLWeaveConfigurationError, match="not a finalized"):
        builder()

    @model_selection
    def choose_first(results):
        return 0

    assert model_selection(choose_first) is choose_first

    with pytest.raises(MLWeaveConfigurationError, match="cannot also be declared"):
        inference_step(choose_first)

    class Probe(BaseWorkflowDecorator):
        pass

    with pytest.raises(TypeError, match="workflow decorators"):
        Probe()._get_spec(object())
    assert WorkflowStepSpec(lambda: None).display_name == "<lambda>"

def test_model_selection_policy_validates_return_values():
    @model_selection
    def choose(cv_results, offset=0):
        return offset

    policy = choose(offset=1)
    assert policy({"params": [{}, {}]}) == 1
    assert policy.selected_index_ == 1

    with pytest.raises(TypeError, match="mapping"):
        policy([])
    with pytest.raises(MLWeaveValidationError, match="does not contain"):
        policy({})

    @model_selection
    def choose_bool(cv_results):
        return True

    with pytest.raises(MLWeaveValidationError, match="integer"):
        choose_bool()({"params": [{}]})

    @model_selection
    def choose_out_of_range(cv_results):
        return 3

    with pytest.raises(MLWeaveValidationError, match="returned index"):
        choose_out_of_range()({"params": [{}]})

def test_inference_step_requires_context():
    @inference_step
    def read_context(context):
        return context.best_index

    with pytest.raises(TypeError, match="WorkflowContext"):
        read_context().run(object())
