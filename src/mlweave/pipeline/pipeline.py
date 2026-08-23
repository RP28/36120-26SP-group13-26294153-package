from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from sklearn.base import clone
from sklearn.pipeline import Pipeline as SklearnPipeline


class Pipeline(SklearnPipeline):
    """An sklearn ``Pipeline`` with small MLWeave conveniences.

    This class intentionally subclasses sklearn's ``Pipeline`` without
    overriding its constructor or execution methods. Therefore 
    behaviour provided by the installed sklearn Pipeline remain sklearn's own
    implementation.
    """

    def exclude_steps(self, *step_names: str | Iterable[str]) -> Pipeline:
        """Return a cloned pipeline with selected steps set to ``passthrough``.

        This method does not mutate the current pipeline. Exclusion is expressed
        using sklearn's native ``passthrough`` mechanism, so the returned object
        continues to behave like an sklearn Pipeline.

        Parameters
        ----------
        *step_names:
            Top-level pipeline step names. Either pass names separately or pass
            one iterable of names.

        Returns
        -------
        Pipeline
            An unfitted clone with the requested steps replaced by
            ``"passthrough"``.
        """
        names = self._normalise_step_names(step_names)
        self._validate_step_names(names)

        result = clone(self)
        if names:
            result.set_params(**{name: "passthrough" for name in names})
        return result

    def excluding(self, *step_names: str | Iterable[str]) -> Pipeline:
        """Alias for :meth:`exclude_steps`."""
        return self.exclude_steps(*step_names)

    def describe(self) -> list[dict[str, Any]]:
        """Return lightweight information about the configured pipeline steps."""
        description: list[dict[str, Any]] = []

        for index, (name, step) in enumerate(self.steps):
            if step is None or (isinstance(step, str) and step == "passthrough"):
                description.append(
                    {
                        "index": index,
                        "name": name,
                        "type": "passthrough",
                        "description": None,
                        "tags": (),
                    })
                continue

            description.append(
                {
                    "index": index,
                    "name": name,
                    "type": step.__class__.__name__,
                    "description": getattr(step, "description", None),
                    "tags": tuple(getattr(step, "tags", ())),
                })
        return description

    @staticmethod
    def _normalise_step_names(
        step_names: tuple[str | Iterable[str], ...],
    ) -> tuple[str, ...]:
        if len(step_names) == 1 and not isinstance(step_names[0], str):
            candidate = step_names[0]
            if isinstance(candidate, Iterable):
                names = tuple(candidate)
            else:
                names = (candidate,)
        else:
            names = tuple(step_names)

        if any(not isinstance(name, str) or not name for name in names):
            raise TypeError("Pipeline step names must be non-empty strings.")

        return tuple(dict.fromkeys(names))

    def _validate_step_names(self, names: tuple[str, ...]) -> None:
        available = {name for name, _ in self.steps}
        missing = [name for name in names if name not in available]
        if missing:
            raise ValueError(
                f"Unknown pipeline step name(s): {missing}. "
                f"Available steps: {[name for name, _ in self.steps]}."
            )


__all__ = ["Pipeline"]
