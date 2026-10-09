"""Built-in and extensible speech-to-speech providers."""

from .base import S2SConnection, S2SProvider
from .gemini import (
    DEFAULT_API_VERSION,
    DEFAULT_MODEL,
    DEFAULT_VOICE,
    GeminiProvider,
)

__all__ = [
    "DEFAULT_API_VERSION",
    "DEFAULT_MODEL",
    "DEFAULT_VOICE",
    "GeminiProvider",
    "S2SConnection",
    "S2SProvider",
]
