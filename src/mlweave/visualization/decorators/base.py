from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any
from mlweave.visualization.core.recipe import PendingPlotRecipe, PlotRecipeBuilder
from mlweave.visualization.core.specs import PlotRecipeSpec

class BasePlotRecipeDecorator(ABC):
    """Shared plumbing for decorators that construct plot recipes."""
    def _get_spec(self, obj: Any) -> PlotRecipeSpec:
        if isinstance(obj, (PendingPlotRecipe, PlotRecipeBuilder)):
            return obj.spec
        if callable(obj):
            return PlotRecipeSpec(recipe_func=obj)
        raise TypeError(
            "mlweave visualization decorators can only be applied to a callable "
            "or an mlweave plot-recipe specification."
        )

    @staticmethod
    def _preserve_state(obj: Any, spec: PlotRecipeSpec):
        if isinstance(obj, (PendingPlotRecipe, PlotRecipeBuilder)):
            return obj
        return PendingPlotRecipe(spec)

class PlotRecipeConfigurationDecorator(BasePlotRecipeDecorator):
    """Base for decorators that add declarative plot configuration."""
    def __call__(self, obj: Any):
        spec = self._get_spec(obj)
        self.configure(spec)
        return self._preserve_state(obj, spec)

    @abstractmethod
    def configure(self, spec: PlotRecipeSpec) -> None:
        """Mutate the shared recipe specification."""

class PlotRecipeFinalizingDecorator(BasePlotRecipeDecorator):
    """Base for the decorator that declares a plot recipe."""
    def __call__(self, obj: Any) -> PlotRecipeBuilder:
        spec = self._get_spec(obj)
        spec.finalized = True
        if isinstance(obj, PlotRecipeBuilder):
            return obj
        return PlotRecipeBuilder(spec)
