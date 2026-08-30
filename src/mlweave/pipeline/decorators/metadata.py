from __future__ import annotations

from mlweave.exceptions import MLWeaveConfigurationError
from mlweave.pipeline.decorators.base import PipelineConfigSpec, PipelineMetadataDecorator

class DescriptionDecorator(PipelineMetadataDecorator):
    """Attach a human-readable description to a pipeline component."""
    def __init__(self, text: str) -> None:
        if not isinstance(text, str) or not text.strip():
            raise ValueError("@description requires a non-empty string.")
        self.text = text.strip()

    def configure(self, spec: PipelineConfigSpec) -> None:
        if spec.description is not None and spec.description != self.text:
            raise MLWeaveConfigurationError(
                f"'{spec.display_name}' already has a description."
            )
        spec.description = self.text

class TagDecorator(PipelineMetadataDecorator):
    """Attach an arbitrary tag to a pipeline component."""
    def __init__(self, value: str) -> None:
        if not isinstance(value, str) or not value.strip():
            raise ValueError("@tag requires a non-empty string.")
        self.value = value.strip()

    def configure(self, spec: PipelineConfigSpec) -> None:
        spec.tags.add(self.value)

description = DescriptionDecorator
tag = TagDecorator
