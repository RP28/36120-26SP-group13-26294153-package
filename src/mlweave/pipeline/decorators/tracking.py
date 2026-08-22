from mlweave.pipeline.decorators.base import PipelineConfigSpec, PipelineConfigurationDecorator


class TrackDecorator(PipelineConfigurationDecorator):
    """Enable lightweight fit/transform execution history."""

    def configure(self, spec: PipelineConfigSpec) -> None:
        spec.tracking = True


track = TrackDecorator()
