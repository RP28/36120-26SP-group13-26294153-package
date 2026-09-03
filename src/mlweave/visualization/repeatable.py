from __future__ import annotations

from collections.abc import Iterable
from typing import Any, Callable
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from mlweave.visualization.core.recipe import PlotRecipeBuilder, materialize_recipe
from mlweave.visualization.core.specs import ColTransform, ColType, PlotRecipe, PlotSpec, PlotType
from mlweave.visualization.decorators.plots import AddPlotDecorator

_DATA_COLUMN_KWARGS = frozenset({"hue", "style", "size", "weights", "units"})

class RepeatablePlots:
    """Render reusable plot recipes against DataFrame columns.
    The renderer owns runtime state (the DataFrame and figure creation), while
    decorators only describe reusable recipe configuration. One-shot plots are
    supplied directly to ``render`` and are never stored.
    Parameters
    ----------
    df:
        DataFrame containing columns to visualize.
    max_plot_rows:
        Optional global row cap used for plotting. ``None`` keeps exact input
        row counts. Supplying a cap can make interactive plotting of very large
        pandas frames much faster, at the cost of sampling.
    random_state:
        Random seed used when ``max_plot_rows`` (or per-plot ``max_rows``) is
        active.
    """
    def __init__(
        self,
        df: pd.DataFrame,
        *,
        max_plot_rows: int | None = None,
        random_state: int | None = 0,
    ) -> None:
        if not isinstance(df, pd.DataFrame):
            raise TypeError("RepeatablePlots currently expects a pandas DataFrame.")
        if max_plot_rows is not None and max_plot_rows <= 0:
            raise ValueError("max_plot_rows must be greater than 0 or None.")
        self._df = df
        self._plots = {
            ColType.NUMERICAL: [],
            ColType.CATEGORICAL: [],
        }
        self._column_type_cache: dict[Any, tuple[Any, ColType]] = {}
        self._max_plot_rows = max_plot_rows
        self._random_state = random_state

    def use(self, recipe: PlotRecipe | PlotRecipeBuilder) -> "RepeatablePlots":
        """Register every plot from a reusable recipe."""
        materialized = materialize_recipe(recipe)
        for plot in materialized.plots:
            self._plots[materialized.col_type].append(plot.clone())
        return self

    def render(
        self,
        col_name: Any,
        transform: ColTransform | Callable | None = None,
        *,
        temporary: AddPlotDecorator | Iterable[AddPlotDecorator] | None = None,
    ):
        """Render reusable recipe plots plus optional one-shot plots.
        Parameters
        ----------
        col_name:
            DataFrame column to visualize.
        transform:
            Optional default numerical transformation for this render call.
        temporary:
            One plot decorator or an iterable of plot decorators to include
            only in this call. These specifications are materialized locally
            and are never stored on the renderer or reusable recipe.
        """
        if col_name not in self._df.columns:
            raise ValueError(f"Column {col_name!r} does not exist in the DataFrame.")
        col_type = self._classify_column(col_name)
        temporary_plots = self._materialize_temporary(temporary)
        plots = tuple(self._plots[col_type]) + temporary_plots
        if not plots:
            raise ValueError(f"No plots registered for {col_type.value} columns.")
        n_plots = len(plots)
        n_cols = 1 if n_plots == 1 else 2
        n_rows = (n_plots + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 5 * n_rows))
        axes_array = np.atleast_1d(axes).ravel()
        transform_cache: dict[tuple[Any, Any], tuple[pd.Series, str, str]] = {}
        category_cache: dict[tuple[Any, int], pd.Index] = {}
        for i, plot in enumerate(plots):
            ax = axes_array[i]
            plot_kwargs = dict(plot.kwargs)
            top_n = plot_kwargs.pop("top_n", None)
            rotate_xticks = plot_kwargs.pop("rotate_xticks", None)
            plot_transform = plot_kwargs.pop("transform", transform)
            max_rows = plot_kwargs.pop("max_rows", self._max_plot_rows)
            random_state = plot_kwargs.pop("random_state", self._random_state)
            if max_rows is not None and max_rows <= 0:
                raise ValueError("max_rows must be greater than 0 or None.")
            if col_type == ColType.CATEGORICAL and plot_transform is not None:
                raise ValueError("Transformations can only be applied to numerical columns.")
            second_col = self._second_column(plot, col_type)
            referenced_columns = self._referenced_columns(plot_kwargs)
            if second_col is not None:
                referenced_columns.add(second_col)
            plot_df, plot_col_name, transform_label = self._prepare_plot_data(
                col_name=col_name,
                transform=plot_transform,
                required_columns=referenced_columns,
                transform_cache=transform_cache,
            )
            if col_type == ColType.CATEGORICAL and top_n is not None:
                plot_df = self._filter_top_categories(
                    plot_df,
                    plot_col_name,
                    int(top_n),
                    required_columns=referenced_columns,
                    cache=category_cache,
                )
            elif max_rows is not None and len(plot_df) > max_rows:
                plot_df = self._narrow_frame(plot_df, plot_col_name, referenced_columns)
            if max_rows is not None and len(plot_df) > max_rows:
                plot_df = plot_df.sample(n=max_rows, random_state=random_state)
            title_suffix = f" ({transform_label})" if transform_label else ""
            display_col_name = (
                f"{transform_label}({col_name})" if transform_label else str(col_name)
            )
            self._render_one(
                plot=plot,
                plot_df=plot_df,
                plot_col_name=plot_col_name,
                original_col_name=col_name,
                display_col_name=display_col_name,
                col_type=col_type,
                second_col=second_col,
                title_suffix=title_suffix,
                ax=ax,
                plot_kwargs=plot_kwargs,
            )
            if rotate_xticks is not None:
                ax.tick_params(axis="x", labelrotation=rotate_xticks)
        for j in range(n_plots, len(axes_array)):
            fig.delaxes(axes_array[j])
        fig.tight_layout()
        return fig

    @staticmethod
    def _materialize_temporary(
        temporary: AddPlotDecorator | Iterable[AddPlotDecorator] | None,
    ) -> tuple[PlotSpec, ...]:
        """Build one-shot plot specs without mutating renderer state."""
        if temporary is None:
            return ()
        if isinstance(temporary, AddPlotDecorator):
            items = (temporary,)
        else:
            if isinstance(temporary, (str, bytes)):
                raise TypeError(
                    "temporary must be a plot decorator or an iterable of plot decorators."
                )
            try:
                items = tuple(temporary)
            except TypeError as exc:
                raise TypeError(
                    "temporary must be a plot decorator or an iterable of plot decorators."
                ) from exc
        materialized: list[PlotSpec] = []
        for item in items:
            if not isinstance(item, AddPlotDecorator):
                raise TypeError(
                    "temporary entries must come from histogram(), boxplot(), "
                    "scatterplot(), regplot(), countplot(), or barplot()."
                )
            materialized.append(item.build())
        return tuple(materialized)

    def _render_one(
        self,
        *,
        plot: PlotSpec,
        plot_df: pd.DataFrame,
        plot_col_name: Any,
        original_col_name: Any,
        display_col_name: str,
        col_type: ColType,
        second_col: Any | None,
        title_suffix: str,
        ax,
        plot_kwargs: dict[str, Any],
    ) -> None:
        if "hue" not in plot_kwargs and "color" not in plot_kwargs:
            plot_kwargs["color"] = ax._get_lines.get_next_color()
        match plot.plot_type:
            case PlotType.HISTOGRAM:
                sns.histplot(
                    data=plot_df,
                    x=plot_col_name,
                    ax=ax,
                    *plot.args,
                    **plot_kwargs,
                )
                ax.set_title(f"Histogram of {original_col_name}{title_suffix}")
            case PlotType.SCATTER:
                if second_col is None:
                    raise ValueError("Scatter plot requires a second column name as an argument.")
                sns.scatterplot(
                    data=plot_df,
                    x=plot_col_name,
                    y=second_col,
                    ax=ax,
                    **plot_kwargs,
                )
                ax.set_title(
                    f"Scatter Plot of {original_col_name}{title_suffix} vs {second_col}"
                )
            case PlotType.REG:
                if second_col is None:
                    raise ValueError(
                        "Regression plot requires a second column name as an argument."
                    )
                sns.regplot(
                    data=plot_df,
                    x=plot_col_name,
                    y=second_col,
                    ax=ax,
                    **plot_kwargs,
                )
                ax.set_title(
                    f"Regression Plot of {original_col_name}{title_suffix} vs {second_col}"
                )
            case PlotType.BOX:
                if col_type == ColType.NUMERICAL:
                    sns.boxplot(
                        data=plot_df,
                        x=plot_col_name,
                        ax=ax,
                        *plot.args,
                        **plot_kwargs,
                    )
                    ax.set_title(f"Box Plot of {original_col_name}{title_suffix}")
                else:
                    if second_col is None:
                        raise ValueError(
                            "For categorical columns, BOX requires a numerical second column."
                        )
                    sns.boxplot(
                        data=plot_df,
                        x=plot_col_name,
                        y=second_col,
                        ax=ax,
                        **plot_kwargs,
                    )
                    ax.set_title(f"Box Plot of {second_col} by {original_col_name}")
            case PlotType.COUNT:
                if col_type != ColType.CATEGORICAL:
                    raise ValueError("COUNT plot is intended for categorical columns.")
                if "order" not in plot_kwargs:
                    plot_kwargs["order"] = plot_df[plot_col_name].value_counts().index
                sns.countplot(
                    data=plot_df,
                    x=plot_col_name,
                    ax=ax,
                    *plot.args,
                    **plot_kwargs,
                )
                ax.set_title(f"Count Plot of {original_col_name}")
            case PlotType.BAR:
                if col_type != ColType.CATEGORICAL:
                    raise ValueError("BAR plot is intended for categorical columns.")
                if second_col is None:
                    raise ValueError("BAR plot requires a second numerical column.")
                sns.barplot(
                    data=plot_df,
                    x=plot_col_name,
                    y=second_col,
                    ax=ax,
                    **plot_kwargs,
                )
                ax.set_title(f"{second_col} by {original_col_name}")
            case _:
                raise ValueError(f"Unsupported PlotType: {plot.plot_type}")
        if col_type == ColType.NUMERICAL:
            ax.set_xlabel(display_col_name)

    def _prepare_plot_data(
        self,
        *,
        col_name: Any,
        transform: ColTransform | Callable | None,
        required_columns: set[Any],
        transform_cache: dict[tuple[Any, Any], tuple[pd.Series, str, str]],
    ) -> tuple[pd.DataFrame, Any, str | None]:
        if transform is None:
            return self._df, col_name, None
        transform_key = self._transform_key(transform)
        cache_key = (col_name, transform_key)
        cached = transform_cache.get(cache_key)
        if cached is None:
            transform_label = self._get_transform_label(transform)
            transformed_col_name = self._unique_transformed_name(col_name, transform_label)
            transformed = self._apply_transform(self._df[col_name], transform)
            if not isinstance(transformed, pd.Series):
                transformed = pd.Series(transformed, index=self._df.index)
            elif not transformed.index.equals(self._df.index):
                transformed = transformed.reindex(self._df.index)
            cached = (transformed, transformed_col_name, transform_label)
            transform_cache[cache_key] = cached
        transformed, transformed_col_name, transform_label = cached
        data: dict[Any, pd.Series] = {transformed_col_name: transformed}
        for column in required_columns:
            self._validate_second_column(column)
            if column != col_name:
                data[column] = self._df[column]
        return pd.DataFrame(data, index=self._df.index, copy=False), transformed_col_name, transform_label

    def _filter_top_categories(
        self,
        df: pd.DataFrame,
        col_name: Any,
        top_n: int,
        *,
        required_columns: set[Any],
        cache: dict[tuple[Any, int], pd.Index],
    ) -> pd.DataFrame:
        if top_n <= 0:
            raise ValueError("top_n must be greater than 0.")
        key = (col_name, top_n)
        top_categories = cache.get(key)
        if top_categories is None:
            top_categories = df[col_name].value_counts().nlargest(top_n).index
            cache[key] = top_categories
        narrow = self._narrow_frame(df, col_name, required_columns)
        filtered = narrow.loc[narrow[col_name].isin(top_categories)]
        if isinstance(filtered[col_name].dtype, pd.CategoricalDtype):
            filtered = filtered.copy()
            filtered[col_name] = filtered[col_name].cat.remove_unused_categories()
        return filtered

    def _narrow_frame(
        self,
        df: pd.DataFrame,
        primary_col: Any,
        required_columns: Iterable[Any],
    ) -> pd.DataFrame:
        columns = [primary_col]
        seen = {primary_col}
        for column in required_columns:
            if column not in seen:
                self._validate_second_column(column)
                columns.append(column)
                seen.add(column)
        if len(columns) == len(df.columns) and all(a == b for a, b in zip(columns, df.columns)):
            return df
        return df.loc[:, columns]

    def _classify_column(self, col_name: Any) -> ColType:
        dtype = self._df[col_name].dtype
        cached = self._column_type_cache.get(col_name)
        if cached is not None and cached[0] == dtype:
            return cached[1]
        is_categorical = (
            isinstance(dtype, pd.CategoricalDtype)
            or pd.api.types.is_object_dtype(dtype)
            or pd.api.types.is_string_dtype(dtype)
            or pd.api.types.is_bool_dtype(dtype)
        )
        if is_categorical:
            col_type = ColType.CATEGORICAL
        elif pd.api.types.is_numeric_dtype(dtype):
            col_type = ColType.NUMERICAL
        else:
            raise ValueError(f"Column {col_name!r} has unsupported dtype: {dtype}")
        self._column_type_cache[col_name] = (dtype, col_type)
        return col_type

    def _second_column(self, plot: PlotSpec, col_type: ColType) -> Any | None:
        needs_second = (
            plot.plot_type in {PlotType.SCATTER, PlotType.REG, PlotType.BAR}
            or (plot.plot_type == PlotType.BOX and col_type == ColType.CATEGORICAL)
        )
        if not needs_second:
            return None
        if not plot.args:
            return None
        second_col = plot.args[0]
        self._validate_second_column(second_col)
        return second_col

    def _referenced_columns(self, kwargs: dict[str, Any]) -> set[Any]:
        columns: set[Any] = set()
        for key in _DATA_COLUMN_KWARGS:
            value = kwargs.get(key)
            if value is not None and value in self._df.columns:
                columns.add(value)
        return columns

    def _validate_second_column(self, second_col: Any) -> None:
        if second_col not in self._df.columns:
            raise ValueError(f"Column {second_col!r} does not exist in the DataFrame.")

    @staticmethod
    def _transform_key(transform: ColTransform | Callable) -> Any:
        if isinstance(transform, ColTransform):
            return transform
        return id(transform)

    def _unique_transformed_name(self, col_name: Any, label: str) -> str:
        base = f"{col_name}_{label}"
        candidate = base
        counter = 1
        while candidate in self._df.columns:
            candidate = f"{base}_{counter}"
            counter += 1
        return candidate

    @staticmethod
    def _apply_transform(series: pd.Series, transform: ColTransform | Callable):
        if callable(transform):
            return transform(series)
        match transform:
            case ColTransform.LOG:
                if (series <= 0).any():
                    raise ValueError("Log transform requires all values to be greater than 0.")
                return np.log(series)
            case ColTransform.LOG1P:
                if (series < 0).any():
                    raise ValueError("Log1p transform requires all values to be >= 0.")
                return np.log1p(series)
            case ColTransform.SIGN_LOG1P:
                return np.sign(series) * np.log1p(np.abs(series))
            case _:
                raise ValueError(f"Unsupported transform: {transform}")

    @staticmethod
    def _get_transform_label(transform: ColTransform | Callable) -> str:
        if callable(transform):
            return getattr(transform, "__name__", "transformed")
        if isinstance(transform, ColTransform):
            return transform.value
        raise ValueError(f"Unsupported transform: {transform}")
