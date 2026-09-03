from __future__ import annotations

from dataclasses import dataclass, field
import enum
from typing import Any, Callable

class PlotType(enum.Enum):
    """Built-in plot types supported by :class:`RepeatablePlots`."""
    HISTOGRAM = "histogram"
    SCATTER = "scatter"
    REG = "reg"
    BOX = "box"
    COUNT = "count"
    BAR = "bar"

class ColType(enum.Enum):
    """Column groups that a plot recipe can target."""
    NUMERICAL = "numerical"
    CATEGORICAL = "categorical"

class ColTransform(enum.Enum):
    """Built-in numerical transforms available while rendering."""
    LOG = "log"
    LOG1P = "log1p"
    SIGN_LOG1P = "sign_log1p"

@dataclass(slots=True)

class PlotSpec:
    """Declarative configuration for one plot in a recipe."""
    plot_type: PlotType
    args: tuple[Any, ...] = ()
    kwargs: dict[str, Any] = field(default_factory=dict)

    def clone(self) -> "PlotSpec":
        """Return an independent lightweight copy of this specification."""
        return PlotSpec(
            plot_type=self.plot_type,
            args=self.args,
            kwargs=dict(self.kwargs),
        )

@dataclass(slots=True)

class PlotRecipeSpec:
    """Mutable specification accumulated by visualization decorators."""
    recipe_func: Callable[..., Any]
    plots: list[PlotSpec] = field(default_factory=list)
    col_type: ColType | None = None
    finalized: bool = False

@dataclass(slots=True, frozen=True)

class PlotRecipe:
    """Reusable, materialized collection of plot specifications."""
    name: str
    col_type: ColType
    plots: tuple[PlotSpec, ...]
    description: str | None = None
