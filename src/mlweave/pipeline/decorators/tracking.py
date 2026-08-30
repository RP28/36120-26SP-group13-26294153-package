from mlweave.pipeline.decorators.base import PipelineConfigSpec, PipelineConfigurationDecorator

class TrackDecorator(PipelineConfigurationDecorator):
    """Print lightweight execution information for a pipeline component."""
    def configure(self, spec: PipelineConfigSpec) -> None:
        spec.tracking = True

track = TrackDecorator()
