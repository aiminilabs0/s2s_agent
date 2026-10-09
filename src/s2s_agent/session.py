"""Provider-neutral live session lifecycle and audio streaming."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Awaitable, Callable, Iterable
from contextlib import AbstractAsyncContextManager
from enum import StrEnum
from types import TracebackType
from typing import TYPE_CHECKING, TypeAlias

from .audio import DEFAULT_INPUT_BLOCK_FRAMES, run_default_audio
from .config import SAMPLE_WIDTH_BYTES, LiveConfig
from .errors import ConfigurationError, LiveConnectionError, SessionStateError
from .events import SessionEvent
from .providers.base import S2SConnection, S2SProvider
from .providers.gemini import GeminiProvider

if TYPE_CHECKING:
    from google import genai

AgentTask: TypeAlias = Callable[["LiveSession"], Awaitable[None]]


class SessionState(StrEnum):
    """Lifecycle states for a :class:`LiveSession`."""

    NEW = "new"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    CLOSED = "closed"


class LiveSession:
    """A provider-neutral bidirectional audio session.

    Use the session as an async context manager. Sending microphone audio and
    consuming :meth:`events` can happen concurrently, but only one event
    consumer may be active at a time.
    """

    def __init__(
        self,
        *,
        config: LiveConfig | None = None,
        provider: S2SProvider | None = None,
        api_key: str | None = None,
        client: genai.Client | None = None,
        tasks: Iterable[AgentTask] = (),
    ) -> None:
        if provider is not None and (api_key is not None or client is not None):
            raise ConfigurationError(
                "api_key and client are Gemini shortcuts and cannot be used with provider"
            )

        self._config = config or LiveConfig()
        self._provider = provider or GeminiProvider(api_key=api_key, client=client)
        self._state = SessionState.NEW
        self._context: AbstractAsyncContextManager[S2SConnection] | None = None
        self._connection: S2SConnection | None = None
        self._send_lock = asyncio.Lock()
        self._receiving = False
        self._tasks: list[AgentTask] = []
        for task in tasks:
            self.add_task(task)

    @property
    def config(self) -> LiveConfig:
        """The immutable configuration used by this session."""

        return self._config

    @property
    def provider(self) -> S2SProvider:
        """Provider adapter used by this session."""

        return self._provider

    @property
    def state(self) -> SessionState:
        """The current session lifecycle state."""

        return self._state

    @property
    def connected(self) -> bool:
        """Whether the provider connection is open."""

        return self._state is SessionState.CONNECTED

    @property
    def tasks(self) -> tuple[AgentTask, ...]:
        """Async callbacks run alongside default audio capture and playback."""

        return tuple(self._tasks)

    def add_task(self, task: AgentTask) -> None:
        """Add an async callback to run when :meth:`run` starts."""

        if self._state is not SessionState.NEW:
            raise SessionStateError("tasks can only be added before the session starts")
        if not callable(task):
            raise TypeError("task must be callable")
        self._tasks.append(task)

    def run(
        self,
        *,
        input_block_frames: int = DEFAULT_INPUT_BLOCK_FRAMES,
        echo_cancellation: bool = True,
        echo_delay_ms: int = 0,
    ) -> None:
        """Run with the default microphone and speaker until the session ends.

        This synchronous convenience method requires the ``audio`` extra. Use
        :meth:`run_async` when an event loop is already running.
        """

        try:
            asyncio.get_running_loop()
        except RuntimeError:
            pass
        else:
            raise RuntimeError(
                "LiveSession.run() cannot be called from a running event loop; "
                "await session.run_async() instead"
            )

        asyncio.run(
            self.run_async(
                input_block_frames=input_block_frames,
                echo_cancellation=echo_cancellation,
                echo_delay_ms=echo_delay_ms,
            )
        )

    async def run_async(
        self,
        *,
        input_block_frames: int = DEFAULT_INPUT_BLOCK_FRAMES,
        echo_cancellation: bool = True,
        echo_delay_ms: int = 0,
    ) -> None:
        """Asynchronously run with the default microphone and speaker."""

        await run_default_audio(
            self,
            input_block_frames=input_block_frames,
            echo_cancellation=echo_cancellation,
            echo_delay_ms=echo_delay_ms,
        )

    async def __aenter__(self) -> LiveSession:
        return await self.connect()

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        await self._close(exc_type, exc, traceback)

    async def connect(self) -> LiveSession:
        """Open the provider connection."""

        if self._state is not SessionState.NEW:
            raise SessionStateError(f"cannot connect a session in state {self._state.value!r}")

        self._state = SessionState.CONNECTING
        try:
            context = self._provider.connect(self._config)
            self._context = context
            self._connection = await context.__aenter__()
        except Exception as exc:
            self._context = None
            self._connection = None
            self._state = SessionState.NEW
            raise LiveConnectionError(
                f"failed to connect to live provider {self._provider.name!r}"
            ) from exc

        self._state = SessionState.CONNECTED
        return self

    async def send_audio(
        self,
        data: bytes | bytearray | memoryview,
        *,
        sample_rate: int | None = None,
    ) -> None:
        """Send one raw mono, little-endian signed 16-bit PCM audio chunk."""

        connection = self._require_connected()
        audio = bytes(data)
        if not audio:
            raise ValueError("audio data must not be empty")
        if len(audio) % SAMPLE_WIDTH_BYTES:
            raise ValueError("16-bit PCM audio data must contain complete two-byte samples")

        rate = self._config.input_sample_rate if sample_rate is None else sample_rate
        if rate <= 0:
            raise ValueError("sample_rate must be greater than zero")

        try:
            async with self._send_lock:
                await connection.send_audio(audio, sample_rate=rate)
        except Exception as exc:
            raise LiveConnectionError(
                f"failed to send audio to live provider {self._provider.name!r}"
            ) from exc

    async def end_audio(self) -> None:
        """Signal that a temporarily paused input audio stream has ended."""

        connection = self._require_connected()
        try:
            async with self._send_lock:
                await connection.end_audio()
        except Exception as exc:
            raise LiveConnectionError(
                f"failed to end audio for live provider {self._provider.name!r}"
            ) from exc

    async def events(self) -> AsyncIterator[SessionEvent]:
        """Yield normalized events until the provider connection closes."""

        connection = self._require_connected()
        if self._receiving:
            raise SessionStateError("only one events() consumer may be active")

        self._receiving = True
        try:
            try:
                async for event in connection.events():
                    yield event
            except Exception as exc:
                raise LiveConnectionError(
                    f"failed while receiving from live provider {self._provider.name!r}"
                ) from exc
        finally:
            self._receiving = False

    async def close(self) -> None:
        """Close the live connection. Calling this more than once is safe."""

        await self._close(None, None, None)

    async def _close(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        if self._state is SessionState.CLOSED:
            return

        context = self._context
        self._context = None
        self._connection = None
        self._state = SessionState.CLOSED

        if context is None:
            return

        try:
            await context.__aexit__(exc_type, exc, traceback)
        except Exception as close_exc:
            raise LiveConnectionError(
                f"failed to close live provider {self._provider.name!r}"
            ) from close_exc

    def _require_connected(self) -> S2SConnection:
        if self._state is not SessionState.CONNECTED or self._connection is None:
            raise SessionStateError(
                f"operation requires a connected session; current state is {self._state.value!r}"
            )
        return self._connection
