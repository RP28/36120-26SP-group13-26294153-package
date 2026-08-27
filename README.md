# mlweave

MLWeave adds lightweight decorators around sklearn pipelines, model-selection workflows, and repeatable plots.

It keeps user code as plain Python functions, then materializes those functions lazily into sklearn-compatible components when you call the decorated builder.

## Installation

```bash
pip install mlweave
```

## Pipeline Quick Start

```python
from sklearn.linear_model import LogisticRegression

from mlweave.pipeline.pipeline import Pipeline
from mlweave.pipeline.decorators.step import pipeline_step, split_step
from mlweave.pipeline.decorators.contracts import preserve_rows, requires_columns

@split_step
def train_validation_split(X, y, validation_fraction=0.2):
    cut = int(len(X) * (1 - validation_fraction))
    return (X.iloc[:cut], X.iloc[cut:]), (y.iloc[:cut], y.iloc[cut:])

@pipeline_step
@requires_columns(["age", "income"])
@preserve_rows
def select_features(X, columns):
    return X.loc[:, columns]

pipe = Pipeline([
    ("split", train_validation_split(validation_fraction=0.25)),
    ("features", select_features(["age", "income"])),
    ("model", LogisticRegression()),
])

pipe.fit(X, y)
```

## Lazy Initialization

Pipeline, workflow, and plot decorators do not immediately create runtime objects. They collect configuration in a small builder.

```python
@pipeline_step
def scale_column(X, column, factor=1.0):
    X = X.copy()
    X[column] = X[column] * factor
    return X

step = scale_column("price", factor=100)
```

`scale_column` is a builder. Calling it creates an sklearn-compatible transformer with the call arguments stored on the component. This makes decorated functions reusable, configurable, and safe to clone inside sklearn pipelines.

Stateful pipeline steps use the same pattern, plus a registered fit function:

```python
from mlweave.pipeline.decorators.step import stateful_pipeline_step

@stateful_pipeline_step
def standardize(X, self, columns):
    X = X.copy()
    X.loc[:, columns] = (X.loc[:, columns] - self.mean_) / self.std_
    return X

@standardize.fit
def fit_standardize(X, y, columns):
    return {
        "mean": X.loc[:, columns].mean(),
        "std": X.loc[:, columns].std(),
    }

step = standardize(["age", "income"])
```

Returned fit-state keys are stored as learned sklearn attributes. Keys without a trailing underscore are converted to names like `mean_`.

## Pipeline Decorators

Finalizers:

| Decorator | Use |
| --- | --- |
| `@pipeline_step` | Declare a stateless transform function. |
| `@stateful_pipeline_step` | Declare a transform function that receives `self` and has a `.fit` function. |
| `@split_step` | Declare a fit-time split boundary that returns dataset partitions. |

Validation and contracts:

| Decorator | Use |
| --- | --- |
| `@validate(func, *args, stage="both", **kwargs)` | Run a custom validator before `fit`, `transform`, or both. |
| `@requires_columns(columns)` | Require named DataFrame columns. |
| `@no_missing_input` | Reject missing values before execution. |
| `@allowed_values(column, values)` | Restrict one column to an allowed value set. |
| `@preserve_rows` | Ensure transform output keeps the input row count. |
| `@preserve_columns` | Ensure transform output keeps the same column labels. |
| `@no_missing_output` | Reject missing values after transform. |

Metadata, tracking, and conditions:

| Decorator | Use |
| --- | --- |
| `@description(text)` | Attach a human-readable step description. |
| `@tag(value)` | Attach a tag used by `Pipeline.describe()`. |
| `@track` | Print lightweight timing and shape information. |
| `@when_column_present(column)` | Run the component only when a column exists. |
| `@when_column_absent(column)` | Run the component only when a column is missing. |

Decorator order is flexible for configuration decorators, but every function pipeline step must be finalized with `@pipeline_step`, `@stateful_pipeline_step`, or `@split_step` before use.

## Multiplexed Splits

`@split_step` creates ordered partitions during `fit`/`fit_transform`. Partition `0` is always training data; later partitions are validation, test, or other non-training data.

A split function returns either:

```python
(X_train, X_validation, X_test)
```

or, when targets are present:

```python
((X_train, X_validation, X_test), (y_train, y_validation, y_test))
```

After a split, intermediate transformers fit on partition `0` and transform every partition with the same fitted state. Final estimators fit only on partition `0`. Runtime methods such as `predict`, `transform`, and `score` map over partition tuples when the pipeline was fitted in multiplex mode.

## Wrapping Existing Sklearn Steps

Use `wrap_step()` when you want MLWeave decorators on an existing sklearn transformer.

```python
from sklearn.preprocessing import StandardScaler

from mlweave.pipeline.wrapping import wrap_step
from mlweave.pipeline.decorators.contracts import preserve_rows
from mlweave.pipeline.decorators.metadata import tag

scaler = preserve_rows(tag("numeric")(wrap_step(StandardScaler())))
```

Finalizer decorators are only for functions. Wrapped sklearn steps are already concrete components.

## Pipeline Helpers

`Pipeline` subclasses `sklearn.pipeline.Pipeline` and adds a few helpers:

```python
pipe.describe()
pipe.exclude_steps("features")
pipe.excluding(["features", "scaler"])
pipe.clear_multiplex_data()
```

`describe()` returns step metadata, including descriptions and tags. `exclude_steps()` and `excluding()` return cloned pipelines with selected steps set to `passthrough`.

## Workflow Decorators

MLWeave workflows combine preprocessing, sklearn model search, selection, and inference.

```python
from sklearn.model_selection import GridSearchCV

from mlweave.workflow.workflow import MLWorkflow
from mlweave.workflow.decorators.step import inference_step, model_selection

@model_selection
def choose_best(cv_results):
    return int(cv_results["rank_test_score"].argmin())

@inference_step
def predict_submission(context):
    X = context.X_inference if context.X_inference is not None else context.X_test
    return context.model.predict(X)

search = GridSearchCV(
    estimator=model,
    param_grid={"C": [0.1, 1.0, 10.0]},
    refit=choose_best(),
    return_train_score=True,
)

workflow = MLWorkflow(
    preprocessing=pipe,
    model_search=search,
    inference=predict_submission(),
)

workflow.fit(data, y)
predictions = workflow.infer(inference_data)
```

Workflow decorators:

| Decorator | Use |
| --- | --- |
| `@model_selection` | Build an sklearn `refit` policy from a function that returns the selected candidate index. |
| `@inference_step` | Build a workflow inference step that receives a `WorkflowContext`. |

The workflow expects preprocessing to produce at least train and validation partitions. It uses those first two partitions to build a `PredefinedSplit` for the configured sklearn search.

## Visualization Decorators

Repeatable plot recipes are also lazy. Decorators describe the recipe, and calling the decorated function materializes it.

```python
from mlweave.visualization.repeatable import RepeatablePlots
from mlweave.visualization.decorators.recipe import plot_recipe
from mlweave.visualization.decorators.types import numerical, categorical
from mlweave.visualization.decorators.plots import histogram, boxplot, countplot

@plot_recipe
@numerical
@histogram(bins=30)
@boxplot()
def numeric_summary():
    """Default numeric summary plots."""

@plot_recipe
@categorical
@countplot(top_n=20, rotate_xticks=45)
def category_summary():
    """Default categorical summary plots."""

plots = RepeatablePlots(df, max_plot_rows=10_000)
plots.use(numeric_summary())
plots.use(category_summary())

fig = plots.render("age")
```

Visualization decorators:

| Decorator | Use |
| --- | --- |
| `@plot_recipe` | Finalize a reusable plot recipe. |
| `@numerical` | Mark the recipe as targeting numerical columns. |
| `@categorical` | Mark the recipe as targeting categorical columns. |
| `@histogram(*args, **kwargs)` | Add a seaborn histogram plot. |
| `@scatterplot(second_column, **kwargs)` | Add a scatter plot. |
| `@boxplot(*args, **kwargs)` | Add a box plot. |
| `@countplot(*args, **kwargs)` | Add a count plot. |
| `@barplot(second_column, **kwargs)` | Add a bar plot. |

One-shot plots can be passed to `render()` without registering them:

```python
from mlweave.visualization.core.specs import ColTransform
from mlweave.visualization.decorators.plots import histogram

fig = plots.render("income", temporary=histogram(bins=50, transform=ColTransform.LOG1P))
```

Use `ColTransform.LOG`, `ColTransform.LOG1P`, `ColTransform.SIGN_LOG1P`, or a callable for numerical transforms.

## Errors

MLWeave raises package-specific exceptions for configuration and validation failures:

```python
from mlweave.exceptions import (
    MLWeaveConfigurationError,
    MLWeaveError,
    MLWeaveValidationError,
)
```

## License

`mlweave` was created by rp. It is licensed under the terms of the MIT license.
