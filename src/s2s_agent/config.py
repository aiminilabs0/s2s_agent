"""Provider-neutral configuration for live speech-to-speech sessions."""

from __future__ import annotations

from dataclasses import dataclass

from .errors import ConfigurationError

DEFAULT_INPUT_SAMPLE_RATE = 16_000
OUTPUT_SAMPLE_RATE = 24_000
CHANNELS = 1
SAMPLE_WIDTH_BYTES = 2


@dataclass(frozen=True, slots=True)
class LiveConfig:
    """Common options for a real-time speech-to-speech session.

    Providers choose their own default model and voice when either value is
    omitted. Input audio is mono, little-endian, signed 16-bit PCM.
    """

    model: str | None = None
    voice: str | None = None
    instructions: str | None = None
    input_sample_rate: int = DEFAULT_INPUT_SAMPLE_RATE

    def __post_init__(self) -> None:
        if self.model is not None and not self.model.strip():
            raise ConfigurationError("model must not be empty")
        if self.voice is not None and not self.voice.strip():
            raise ConfigurationError("voice must not be empty")
        if self.instructions is not None and not self.instructions.strip():
            raise ConfigurationError("instructions must not be blank")
        if self.input_sample_rate <= 0:
            raise ConfigurationError("input_sample_rate must be greater than zero")

    @property
    def input_mime_type(self) -> str:
        """MIME type sent with each raw PCM input chunk."""

        return f"audio/pcm;rate={self.input_sample_rate}"
