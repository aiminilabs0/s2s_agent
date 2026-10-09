"""Provider-neutral events emitted by a live speech session."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypeAlias

from .config import CHANNELS, OUTPUT_SAMPLE_RATE, SAMPLE_WIDTH_BYTES


@dataclass(frozen=True, slots=True)
class AudioEvent:
    """A chunk of raw, mono, little-endian signed 16-bit PCM model audio."""

    data: bytes
    sample_rate: int = OUTPUT_SAMPLE_RATE
    channels: int = CHANNELS
    sample_width: int = SAMPLE_WIDTH_BYTES


@dataclass(frozen=True, slots=True)
class TurnCompleteEvent:
    """The provider has completed its current response turn."""


@dataclass(frozen=True, slots=True)
class InterruptedEvent:
    """The user interrupted the provider's current response."""


SessionEvent: TypeAlias = AudioEvent | TurnCompleteEvent | InterruptedEvent
