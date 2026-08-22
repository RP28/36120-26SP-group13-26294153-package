from mlweave.pipeline.decorators.base import PipelineFinalizingDecorator


class PipelineStepDecorator(PipelineFinalizingDecorator):
    """Declare a stateless pipeline step whose ``fit`` is a no-op."""

    mode = "stateless"


class StatefulPipelineStepDecorator(PipelineFinalizingDecorator):
    """Declare a pipeline step with user-defined fitting logic."""

    mode = "stateful"


pipeline_step = PipelineStepDecorator()
stateful_pipeline_step = StatefulPipelineStepDecorator()
