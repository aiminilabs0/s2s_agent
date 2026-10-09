"""Exceptions raised by s2s-agent."""


class S2SAgentError(Exception):
    """Base class for all SDK-specific errors."""


class ConfigurationError(S2SAgentError, ValueError):
    """The SDK configuration is incomplete or invalid."""


class SessionStateError(S2SAgentError, RuntimeError):
    """An operation is not valid for the session's current state."""


class LiveConnectionError(S2SAgentError):
    """A live connection could not be opened or closed cleanly."""
