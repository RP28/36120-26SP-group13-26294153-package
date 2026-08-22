# MLWeave visualization

`mlweave.visualization` uses decorators to define reusable plot recipes while
`RepeatablePlots` owns DataFrame-specific rendering state.

- `@plot_recipe` finalizes a reusable recipe lazily.
- `@histogram`, `@boxplot`, `@scatterplot`, `@countplot`, and `@barplot`
  add declarative plot specifications.
- `RepeatablePlots.use(...)` registers reusable recipes.
- `RepeatablePlots.render(..., temporary=...)` accepts one-shot plot
  declarations that exist only for that render call.

There is intentionally no imperative `add_plot()` / `add_temp_plot()` API.
Reusable configuration belongs in recipes, while one-off configuration is
passed directly to `render()`.

## Reusable recipe

```python
from mlweave.visualization.core.specs import ColTransform, ColType
from mlweave.visualization.decorators.plots import boxplot, histogram, scatterplot
from mlweave.visualization.decorators.recipe import plot_recipe
from mlweave.visualization.repeatable import RepeatablePlots


@plot_recipe
@histogram(ColType.NUMERICAL, bins="auto", kde=True, hue="label")
@boxplot(ColType.NUMERICAL)
def numerical_overview():
    """Reusable numerical EDA views."""


plots = RepeatablePlots(df)
plots.use(numerical_overview)
plots.render("FTM", transform=ColTransform.LOG1P)
```

The position of `@plot_recipe` in the decorator stack does not matter.

## One-shot temporary plots

The built-in plot decorators are also lightweight plot declarations, so the
same objects can be supplied directly to `render()`:

```python
fig = plots.render(
    "FTM",
    temporary=[
        scatterplot(ColType.NUMERICAL, "Min_per", hue="label", max_rows=50_000),
        boxplot(ColType.NUMERICAL),
    ],
)
```

Those plots are materialized locally for that call. They are not stored on the
renderer, do not mutate the reusable recipe, and do not need cleanup after a
successful or failed render. A one-shot plot can also be used when no reusable
recipe has been registered.

## Large-frame behavior

The renderer avoids copying the complete DataFrame merely to transform one
column. A transformed plot materializes only the transformed Series plus other
columns actually referenced by that plot (for example `y`, `hue`, `style`,
`size`, `weights`, or `units`). Identical transforms are cached for one render
cycle and then released.

For plots where drawing every row is unnecessary, use `max_rows=` on an
individual plot decorator or `max_plot_rows=` on `RepeatablePlots`. Sampling is
opt-in so exact plotting remains the default.
