"""A small async SDK for real-time speech-to-speech agents."""

from .agent import create_agent
from .config import (
    CHANNELS,
    DEFAULT_INPUT_SAMPLE_RATE,
    OUTPUT_SAMPLE_RATE,
    SAMPLE_WIDTH_BYTES,
    LiveConfig,
)
from .errors import (
    ConfigurationError,
    LiveConnectionError,
    S2SAgentError,
    SessionStateError,
)
from .events import AudioEvent, InterruptedEvent, SessionEvent, TurnCompleteEvent
from .providers import (
    DEFAULT_API_VERSION,
    DEFAULT_MODEL,
    DEFAULT_VOICE,
    GeminiProvider,
    S2SConnection,
    S2SProvider,
)
from .session import AgentTask, LiveSession, SessionState

__all__ = [
    "CHANNELS",
    "DEFAULT_API_VERSION",
    "DEFAULT_INPUT_SAMPLE_RATE",
    "DEFAULT_MODEL",
    "DEFAULT_VOICE",
    "OUTPUT_SAMPLE_RATE",
    "SAMPLE_WIDTH_BYTES",
    "AgentTask",
    "AudioEvent",
    "ConfigurationError",
    "GeminiProvider",
    "InterruptedEvent",
    "LiveConfig",
    "LiveConnectionError",
    "LiveSession",
    "S2SAgentError",
    "S2SConnection",
    "S2SProvider",
    "SessionEvent",
    "SessionState",
    "SessionStateError",
    "TurnCompleteEvent",
    "create_agent",
]

__version__ = "0.1.0"
