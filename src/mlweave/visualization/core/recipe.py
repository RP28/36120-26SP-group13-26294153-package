from __future__ import annotations

from functools import update_wrapper
from typing import Any
from mlweave.exceptions import MLWeaveConfigurationError
from mlweave.visualization.core.specs import PlotRecipe, PlotRecipeSpec

class PendingPlotRecipe:
    """Hold plot configuration until ``@plot_recipe`` finalizes it."""
    def __init__(self, spec: PlotRecipeSpec) -> None:
        self.spec = spec
        update_wrapper(self, spec.recipe_func)

    def __call__(self, *args: Any, **kwargs: Any):
        raise MLWeaveConfigurationError(
            f"'{self.spec.recipe_func.__name__}' uses mlweave visualization "
            "decorators but is missing @plot_recipe."
        )

    def __repr__(self) -> str:
        return (
            f"<PendingPlotRecipe {self.spec.recipe_func.__name__!r}: "
            "missing @plot_recipe>"
        )

class PlotRecipeBuilder:
    """Lazy builder for a reusable :class:`PlotRecipe`."""
    def __init__(self, spec: PlotRecipeSpec) -> None:
        self.spec = spec
        update_wrapper(self, spec.recipe_func)

    def __call__(self) -> PlotRecipe:
        if not self.spec.finalized:
            raise MLWeaveConfigurationError(
                f"'{self.spec.recipe_func.__name__}' is not finalized. "
                "Add @plot_recipe."
            )
        if self.spec.col_type is None:
            raise MLWeaveConfigurationError(
                f"'{self.spec.recipe_func.__name__}' must declare either "
                "@numerical or @categorical."
            )
        return PlotRecipe(
            name=self.spec.recipe_func.__name__,
            description=self.spec.recipe_func.__doc__,
            col_type=self.spec.col_type,
            plots=tuple(plot.clone() for plot in self.spec.plots),
        )

    def build(self) -> PlotRecipe:
        """Materialize the recipe explicitly."""
        return self()

    def __repr__(self) -> str:
        return f"<PlotRecipeBuilder {self.spec.recipe_func.__name__!r}>"

def materialize_recipe(recipe: PlotRecipe | PlotRecipeBuilder) -> PlotRecipe:
    """Return a concrete recipe from either accepted public representation."""
    if isinstance(recipe, PlotRecipe):
        return recipe
    if isinstance(recipe, PlotRecipeBuilder):
        return recipe()
    raise TypeError("Expected a PlotRecipe or @plot_recipe decorated builder.")
