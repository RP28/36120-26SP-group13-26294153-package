from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from mlweave.exceptions import MLWeaveConfigurationError
from mlweave.visualization.core.recipe import PendingPlotRecipe, PlotRecipeBuilder, materialize_recipe
from mlweave.visualization.core.specs import ColTransform, ColType, PlotRecipe, PlotRecipeSpec, PlotSpec, PlotType
from mlweave.visualization.decorators.plots import AddPlotDecorator, barplot, boxplot, countplot, histogram, scatterplot
from mlweave.visualization.decorators.recipe import plot_recipe
from mlweave.visualization.decorators.types import categorical, numerical
from mlweave.visualization.repeatable import RepeatablePlots

def plot_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "num": [1.0, 2.0, 3.0, 4.0, 5.0],
            "other": [5.0, 4.0, 3.0, 2.0, 1.0],
            "cat": pd.Categorical(["a", "b", "a", "c", "a"]),
            "group": ["g1", "g1", "g2", "g2", "g2"],
        }
    )

def test_plot_recipe_builders_materialize_in_decorator_order():
    @plot_recipe
    @numerical
    @histogram(bins=5)
    @boxplot()
    def numeric_recipe():
        """Numeric docs."""

    assert isinstance(numeric_recipe, PlotRecipeBuilder)
    assert "PlotRecipeBuilder" in repr(numeric_recipe)
    recipe = numeric_recipe()
    assert numeric_recipe.build().name == "numeric_recipe"
    assert plot_recipe(numeric_recipe) is numeric_recipe
    assert recipe.name == "numeric_recipe"
    assert recipe.description == "Numeric docs."
    assert recipe.col_type == ColType.NUMERICAL
    assert [plot.plot_type for plot in recipe.plots] == [PlotType.HISTOGRAM, PlotType.BOX]
    assert materialize_recipe(numeric_recipe).plots[0].kwargs["bins"] == 5
    assert materialize_recipe(recipe) is recipe

    with pytest.raises(TypeError, match="Expected"):
        materialize_recipe(object())

def test_plot_recipe_errors_for_missing_finalizer_or_type_conflict():
    @numerical
    def pending():
        pass

    assert isinstance(pending, PendingPlotRecipe)
    assert "missing @plot_recipe" in repr(pending)
    with pytest.raises(MLWeaveConfigurationError, match="missing @plot_recipe"):
        pending()

    unfinalized = PlotRecipeBuilder(PlotRecipeSpec(lambda: None))
    with pytest.raises(MLWeaveConfigurationError, match="not finalized"):
        unfinalized()

    @plot_recipe
    def missing_type():
        pass

    with pytest.raises(MLWeaveConfigurationError, match="@numerical or @categorical"):
        missing_type()

    with pytest.raises(MLWeaveConfigurationError, match="cannot be both"):
        categorical(numerical(lambda: None))

    with pytest.raises(TypeError, match="plot_type"):
        AddPlotDecorator("histogram")
    with pytest.raises(TypeError, match="visualization decorators"):
        numerical(object())

def test_repeatable_plots_render_numeric_and_categorical_recipes():
    df = plot_frame()

    @plot_recipe
    @numerical
    @histogram(bins=3)
    @scatterplot("other", hue="group")
    def numeric_recipe():
        pass

    numeric_plots = RepeatablePlots(df, max_plot_rows=4).use(numeric_recipe)
    fig = numeric_plots.render("num", transform=ColTransform.LOG1P)
    assert len(fig.axes) == 2
    assert "Histogram of num" in fig.axes[0].get_title()
    plt.close(fig)

    @plot_recipe
    @categorical
    @countplot(top_n=2, rotate_xticks=30)
    @barplot("other", estimator="mean", errorbar=None)
    def categorical_recipe():
        pass

    categorical_plots = RepeatablePlots(df).use(categorical_recipe())
    fig = categorical_plots.render("cat")
    assert len(fig.axes) == 2
    assert "Count Plot of cat" in fig.axes[0].get_title()
    plt.close(fig)

    @plot_recipe
    @numerical
    @histogram()
    def single_recipe():
        pass

    fig = RepeatablePlots(df).use(single_recipe).render("num")
    assert len(fig.axes) == 1
    plt.close(fig)

def test_repeatable_plots_temporary_plots_and_transform_paths():
    df = plot_frame()

    @plot_recipe
    @numerical
    @histogram()
    def numeric_recipe():
        pass

    renderer = RepeatablePlots(df)
    renderer.use(numeric_recipe)
    fig = renderer.render(
        "num",
        temporary=[
            histogram(transform=lambda series: series + 1),
            boxplot(max_rows=3, random_state=7),
        ],
    )
    assert len(fig.axes) == 3
    plt.close(fig)
    fig = renderer.render("num", temporary=histogram())
    assert len(fig.axes) == 2
    plt.close(fig)
    assert np.allclose(RepeatablePlots._apply_transform(df["num"], ColTransform.SIGN_LOG1P), np.sign(df["num"]) * np.log1p(np.abs(df["num"])))
    assert np.allclose(RepeatablePlots._apply_transform(df["num"], ColTransform.LOG), np.log(df["num"]))

    class CallableTransform:
        def __call__(self, series):
            return series.sort_index(ascending=False)

    conflict_df = df.assign(num_transformed=0)
    fig = RepeatablePlots(conflict_df).use(numeric_recipe).render("num", transform=CallableTransform())
    assert "transformed(num)" in fig.axes[0].get_xlabel()
    plt.close(fig)

    fig = RepeatablePlots(df).use(numeric_recipe).render("num", transform=lambda series: series.to_numpy())
    assert "<lambda>(num)" in fig.axes[0].get_xlabel()
    plt.close(fig)

def test_repeatable_plots_validation_errors():
    df = plot_frame()
    with pytest.raises(TypeError, match="DataFrame"):
        RepeatablePlots([1, 2, 3])
    with pytest.raises(ValueError, match="greater than 0"):
        RepeatablePlots(df, max_plot_rows=0)
    with pytest.raises(ValueError, match="does not exist"):
        RepeatablePlots(df).render("missing")
    with pytest.raises(ValueError, match="No plots registered"):
        RepeatablePlots(df).render("num")
    with pytest.raises(TypeError, match="temporary"):
        RepeatablePlots._materialize_temporary("bad")
    with pytest.raises(TypeError, match="temporary"):
        RepeatablePlots._materialize_temporary(object())
    with pytest.raises(TypeError, match="temporary entries"):
        RepeatablePlots._materialize_temporary([object()])
    with pytest.raises(ValueError, match="Log transform"):
        RepeatablePlots._apply_transform(pd.Series([0.0]), ColTransform.LOG)
    with pytest.raises(ValueError, match="Log1p transform"):
        RepeatablePlots._apply_transform(pd.Series([-1.0]), ColTransform.LOG1P)
    with pytest.raises(ValueError, match="Unsupported transform"):
        RepeatablePlots._apply_transform(pd.Series([1.0]), "bad")
    with pytest.raises(ValueError, match="Unsupported transform"):
        RepeatablePlots._get_transform_label("bad")
    with pytest.raises(ValueError, match="unsupported dtype"):
        RepeatablePlots(df.assign(date=pd.date_range("2024-01-01", periods=len(df)))).render("date")
    assert RepeatablePlots(df)._narrow_frame(df, "num", ["other", "cat", "group"]) is df

def test_repeatable_plots_render_errors_for_mismatched_plot_types():
    df = plot_frame()

    @plot_recipe
    @categorical
    @boxplot()
    def bad_categorical_box():
        pass

    with pytest.raises(ValueError, match="requires a numerical second column"):
        RepeatablePlots(df).use(bad_categorical_box).render("cat")

    @plot_recipe
    @numerical
    @countplot()
    def bad_numeric_count():
        pass

    with pytest.raises(ValueError, match="COUNT plot"):
        RepeatablePlots(df).use(bad_numeric_count).render("num")

    @plot_recipe
    @categorical
    @barplot()
    def bad_bar():
        pass

    with pytest.raises(ValueError, match="second numerical column"):
        RepeatablePlots(df).use(bad_bar).render("cat")

    @plot_recipe
    @categorical
    @countplot(transform=ColTransform.LOG1P)
    def bad_transform():
        pass

    with pytest.raises(ValueError, match="Transformations"):
        RepeatablePlots(df).use(bad_transform).render("cat")

    @plot_recipe
    @categorical
    @boxplot("other")
    def categorical_box():
        pass

    fig = RepeatablePlots(df).use(categorical_box).render("cat")
    assert "Box Plot of other by cat" in fig.axes[0].get_title()
    plt.close(fig)

    @plot_recipe
    @numerical
    @scatterplot()
    def bad_scatter():
        pass

    with pytest.raises(ValueError, match="second column"):
        RepeatablePlots(df).use(bad_scatter).render("num")

    @plot_recipe
    @numerical
    @scatterplot("missing")
    def missing_second():
        pass

    with pytest.raises(ValueError, match="does not exist"):
        RepeatablePlots(df).use(missing_second).render("num")

    @plot_recipe
    @categorical
    @countplot(top_n=0)
    def bad_top_n():
        pass

    with pytest.raises(ValueError, match="top_n"):
        RepeatablePlots(df).use(bad_top_n).render("cat")

    @plot_recipe
    @numerical
    @histogram(max_rows=0)
    def bad_max_rows():
        pass

    with pytest.raises(ValueError, match="max_rows"):
        RepeatablePlots(df).use(bad_max_rows).render("num")

    @plot_recipe
    @numerical
    @barplot("other")
    def bad_numeric_bar():
        pass

    with pytest.raises(ValueError, match="BAR plot"):
        RepeatablePlots(df).use(bad_numeric_bar).render("num")

    fig, ax = plt.subplots()
    with pytest.raises(ValueError, match="Unsupported PlotType"):
        RepeatablePlots(df)._render_one(
            plot=PlotSpec("bad"),
            plot_df=df,
            plot_col_name="num",
            original_col_name="num",
            display_col_name="num",
            col_type=ColType.NUMERICAL,
            second_col=None,
            title_suffix="",
            ax=ax,
            plot_kwargs={},
        )
    plt.close(fig)

def test_plot_specs_clone_independently():
    spec = PlotSpec(PlotType.HISTOGRAM, kwargs={"bins": 10})
    clone = spec.clone()
    clone.kwargs["bins"] = 5
    assert spec.kwargs["bins"] == 10
    recipe = PlotRecipe("r", ColType.NUMERICAL, (spec,))
    assert recipe.name == "r"
