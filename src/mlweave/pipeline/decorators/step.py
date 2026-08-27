from mlweave.pipeline.decorators.base import PipelineFinalizingDecorator

class PipelineStepDecorator(PipelineFinalizingDecorator):
    """Declare a stateless pipeline step whose ``fit`` is a no-op."""
    mode = "stateless"

class StatefulPipelineStepDecorator(PipelineFinalizingDecorator):
    """Declare a pipeline step with user-defined fitting logic."""
    mode = "stateful"

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
split_step = SplitStepDecorator()
