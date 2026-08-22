from __future__ import annotations

from typing import Any

from mlweave.visualization.core.specs import ColType, PlotRecipeSpec, PlotSpec, PlotType
from mlweave.visualization.decorators.base import PlotRecipeConfigurationDecorator


class AddPlotDecorator(PlotRecipeConfigurationDecorator):
    """Add one built-in plot specification to a recipe.

    One parameterized decorator class is sufficient for all built-in plot
    types; separate subclasses would add hierarchy without adding behavior.
    """

    def __init__(
        self,
        plot_type: PlotType,
        col_type: ColType,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        if not isinstance(plot_type, PlotType):
            raise TypeError("plot_type must be a PlotType value.")
        if not isinstance(col_type, ColType):
            raise TypeError("col_type must be a ColType value.")
        self._plot_spec = PlotSpec(plot_type, col_type, tuple(args), dict(kwargs))

    def configure(self, spec: PlotRecipeSpec) -> None:
        # Python applies decorators bottom-up. Inserting at the front makes the
        # final render order match the visual top-to-bottom decorator order.
        spec.plots.insert(0, self.build())

    def build(self) -> PlotSpec:
        """Materialize this declarative plot configuration.

        The same object can therefore be used either as a decorator inside a
        ``@plot_recipe`` definition or as a one-shot plot passed directly to
        ``RepeatablePlots.render(..., temporary=[...])``.
        """
        return self._plot_spec.clone()


def histogram(col_type: ColType, *args: Any, **kwargs: Any) -> AddPlotDecorator:
    """Add a histogram to a plot recipe."""
    return AddPlotDecorator(PlotType.HISTOGRAM, col_type, *args, **kwargs)


def scatterplot(col_type: ColType, *args: Any, **kwargs: Any) -> AddPlotDecorator:
    """Add a scatter plot to a plot recipe."""
    return AddPlotDecorator(PlotType.SCATTER, col_type, *args, **kwargs)


def boxplot(col_type: ColType, *args: Any, **kwargs: Any) -> AddPlotDecorator:
    """Add a box plot to a plot recipe."""
    return AddPlotDecorator(PlotType.BOX, col_type, *args, **kwargs)


def countplot(col_type: ColType = ColType.CATEGORICAL, *args: Any, **kwargs: Any) -> AddPlotDecorator:
    """Add a count plot to a plot recipe."""
    return AddPlotDecorator(PlotType.COUNT, col_type, *args, **kwargs)


def barplot(col_type: ColType = ColType.CATEGORICAL, *args: Any, **kwargs: Any) -> AddPlotDecorator:
    """Add a bar plot to a plot recipe."""
    return AddPlotDecorator(PlotType.BAR, col_type, *args, **kwargs)
