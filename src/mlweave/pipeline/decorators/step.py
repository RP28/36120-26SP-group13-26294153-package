from mlweave.exceptions import MLWeaveConfigurationError
from mlweave.pipeline.core.specs import WrappedStepSpec
from mlweave.pipeline.decorators.base import (
    PipelineConfigSpec,
    PipelineConfigurationDecorator,
    PipelineFinalizingDecorator,
)

class PipelineStepDecorator(PipelineFinalizingDecorator):
    """Declare a stateless pipeline step whose ``fit`` is a no-op."""
    mode = "stateless"

class StatefulPipelineStepDecorator(PipelineFinalizingDecorator):
    """Declare a pipeline step with user-defined fitting logic."""
    mode = "stateful"

class InferenceStepDecorator(PipelineFinalizingDecorator):
    """Declare a stateless transform that MLWeave Pipeline skips during fitting.

    Pipeline data is the first positional argument, regardless of its name.
    Use ``@runtime_inputs`` for additional inference keywords and call the
    resulting builder to create an sklearn-compatible transformer.
    """
    mode = "inference"

class RuntimeInputsDecorator(PipelineConfigurationDecorator):
    """Declare keywords dispatched only to this function's inference step.

    Works above or below ``@inference_step``. Signature and compatibility
    checks happen when the builder is called. Defaults make inputs optional;
    inputs without defaults must be supplied to the pipeline inference call.
    """
    def __init__(self, *names: str) -> None:
        self.names = names

    def configure(self, spec: PipelineConfigSpec) -> None:
        if isinstance(spec, WrappedStepSpec):
            raise MLWeaveConfigurationError(
                "@runtime_inputs is only valid with a function @inference_step."
            )
        spec.runtime_inputs = (spec.runtime_inputs or ()) + self.names

class SplitStepDecorator(PipelineFinalizingDecorator):
    """Declare a fit-time dataset split boundary.

    The decorated function receives ``(X, y, *args, **kwargs)`` and must return
    either ``(X_train, X_eval, ...)`` or
    ``((X_train, X_eval, ...), (y_train, y_eval, ...))``. Partition zero is
    always treated as training data by MLWeave's multiplexing runtime.
    """

    mode = "split"

pipeline_step = PipelineStepDecorator()
stateful_pipeline_step = StatefulPipelineStepDecorator()
inference_step = InferenceStepDecorator()
runtime_inputs = RuntimeInputsDecorator
split_step = SplitStepDecorator()
