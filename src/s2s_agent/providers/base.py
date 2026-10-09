"""Provider-neutral interfaces for real-time speech-to-speech backends."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import AbstractAsyncContextManager
from typing import Protocol

from ..config import LiveConfig
from ..events import SessionEvent


class S2SConnection(Protocol):
    """An open connection to a real-time speech-to-speech provider."""

    async def send_audio(self, data: bytes, *, sample_rate: int) -> None:
        """Send one chunk of mono, signed 16-bit PCM audio."""

    async def end_audio(self) -> None:
        """Signal that the current input audio stream has ended."""

    def events(self) -> AsyncIterator[SessionEvent]:
        """Yield normalized events until the provider connection closes."""


class S2SProvider(Protocol):
    """Factory for provider-specific speech-to-speech connections."""

    name: str

    def connect(self, config: LiveConfig) -> AbstractAsyncContextManager[S2SConnection]:
        """Create an async context manager for one live connection."""
