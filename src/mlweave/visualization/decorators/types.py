from __future__ import annotations

from mlweave.exceptions import MLWeaveConfigurationError
from mlweave.visualization.core.specs import ColType, PlotRecipeSpec
from mlweave.visualization.decorators.base import PlotRecipeConfigurationDecorator


class RecipeColumnTypeDecorator(PlotRecipeConfigurationDecorator):
    """Base decorator that assigns one column type to a whole plot recipe."""

    col_type: ColType

    def configure(self, spec: PlotRecipeSpec) -> None:
        if spec.col_type is not None and spec.col_type != self.col_type:
            raise MLWeaveConfigurationError(
                f"'{spec.recipe_func.__name__}' cannot be both "
                "@numerical and @categorical."
            )
        spec.col_type = self.col_type


class NumericalDecorator(RecipeColumnTypeDecorator):
    """Declare that every plot in the recipe targets numerical columns."""

    col_type = ColType.NUMERICAL


class CategoricalDecorator(RecipeColumnTypeDecorator):
    """Declare that every plot in the recipe targets categorical columns."""

    col_type = ColType.CATEGORICAL


numerical = NumericalDecorator()
categorical = CategoricalDecorator()
