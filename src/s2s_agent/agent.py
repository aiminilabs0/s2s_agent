"""High-level helpers for creating voice agents."""

from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING

from .config import DEFAULT_INPUT_SAMPLE_RATE, LiveConfig
from .providers.base import S2SProvider
from .session import AgentTask, LiveSession

if TYPE_CHECKING:
    from google import genai


def create_agent(
    *,
    instructions: str | None = None,
    voice: str | None = None,
    model: str | None = None,
    input_sample_rate: int = DEFAULT_INPUT_SAMPLE_RATE,
    provider: S2SProvider | None = None,
    api_key: str | None = None,
    client: genai.Client | None = None,
    tasks: Iterable[AgentTask] = (),
) -> LiveSession:
    """Create a configured voice agent.

    The returned agent can be started with :meth:`LiveSession.run`, used from
    an event loop with :meth:`LiveSession.run_async`, or controlled through the
    lower-level session methods.
    """

    config = LiveConfig(
        model=model,
        voice=voice,
        instructions=instructions,
        input_sample_rate=input_sample_rate,
    )
    return LiveSession(
        config=config,
        provider=provider,
        api_key=api_key,
        client=client,
        tasks=tasks,
    )
