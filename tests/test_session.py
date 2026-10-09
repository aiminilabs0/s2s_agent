from __future__ import annotations

from collections.abc import AsyncIterator
from types import TracebackType

import pytest

from s2s_agent import (
    AudioEvent,
    ConfigurationError,
    GeminiProvider,
    InterruptedEvent,
    LiveConfig,
    LiveSession,
    SessionEvent,
    SessionState,
    SessionStateError,
    TurnCompleteEvent,
)


class FakeConnection:
    def __init__(self, events: list[SessionEvent] | None = None) -> None:
        self.event_values = events or []
        self.sent: list[tuple[bytes, int]] = []
        self.audio_ended = 0

    async def send_audio(self, data: bytes, *, sample_rate: int) -> None:
        self.sent.append((data, sample_rate))

    async def end_audio(self) -> None:
        self.audio_ended += 1

    async def events(self) -> AsyncIterator[SessionEvent]:
        for event in self.event_values:
            yield event


class FakeContext:
    def __init__(self, connection: FakeConnection) -> None:
        self.connection = connection
        self.entered = 0
        self.exited = 0

    async def __aenter__(self) -> FakeConnection:
        self.entered += 1
        return self.connection

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.exited += 1


class FakeProvider:
    name = "fake"

    def __init__(self, connection: FakeConnection | None = None) -> None:
        self.connection = connection or FakeConnection()
        self.context = FakeContext(self.connection)
        self.configs: list[LiveConfig] = []

    def connect(self, config: LiveConfig) -> FakeContext:
        self.configs.append(config)
        return self.context


def make_session(
    connection: FakeConnection | None = None,
) -> tuple[LiveSession, FakeProvider, FakeConnection]:
    provider = FakeProvider(connection)
    config = LiveConfig(instructions="Test instructions")
    return LiveSession(config=config, provider=provider), provider, provider.connection


async def test_context_manager_connects_and_closes_once() -> None:
    session, provider, _ = make_session()

    async with session:
        assert session.connected
        assert session.state is SessionState.CONNECTED
        assert session.provider is provider
        assert provider.context.entered == 1
        assert provider.configs == [session.config]

    assert session.state is SessionState.CLOSED
    assert provider.context.exited == 1

    await session.close()
    assert provider.context.exited == 1


def test_default_provider_is_gemini(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")

    assert isinstance(LiveSession().provider, GeminiProvider)


def test_provider_cannot_be_combined_with_gemini_shortcuts() -> None:
    with pytest.raises(ConfigurationError, match="Gemini shortcuts"):
        LiveSession(provider=FakeProvider(), api_key="test-key")


async def test_run_async_uses_default_audio(monkeypatch: pytest.MonkeyPatch) -> None:
    session, _, _ = make_session()
    calls: list[tuple[LiveSession, int, bool, int]] = []

    async def fake_run_default_audio(
        target: LiveSession,
        *,
        input_block_frames: int,
        echo_cancellation: bool,
        echo_delay_ms: int,
    ) -> None:
        calls.append((target, input_block_frames, echo_cancellation, echo_delay_ms))

    monkeypatch.setattr("s2s_agent.session.run_default_audio", fake_run_default_audio)

    await session.run_async(
        input_block_frames=400,
        echo_cancellation=False,
        echo_delay_ms=25,
    )

    assert calls == [(session, 400, False, 25)]


def test_run_starts_the_audio_event_loop(monkeypatch: pytest.MonkeyPatch) -> None:
    session, _, _ = make_session()
    calls: list[tuple[LiveSession, int, bool, int]] = []

    async def fake_run_default_audio(
        target: LiveSession,
        *,
        input_block_frames: int,
        echo_cancellation: bool,
        echo_delay_ms: int,
    ) -> None:
        calls.append((target, input_block_frames, echo_cancellation, echo_delay_ms))

    monkeypatch.setattr("s2s_agent.session.run_default_audio", fake_run_default_audio)

    session.run(input_block_frames=200)

    assert calls == [(session, 200, True, 0)]


async def test_run_rejects_an_active_event_loop() -> None:
    session, _, _ = make_session()

    with pytest.raises(RuntimeError, match="run_async"):
        session.run()


async def test_tasks_must_be_added_before_session_starts() -> None:
    session, _, _ = make_session()

    async def background_task(target: LiveSession) -> None:
        pass

    session.add_task(background_task)
    assert session.tasks == (background_task,)

    await session.connect()
    with pytest.raises(SessionStateError, match="before the session starts"):
        session.add_task(background_task)
    await session.close()


async def test_send_audio_and_end_stream() -> None:
    session, _, connection = make_session()

    with pytest.raises(SessionStateError, match="connected"):
        await session.send_audio(b"\x00\x00")

    await session.connect()

    with pytest.raises(ValueError, match="empty"):
        await session.send_audio(b"")
    with pytest.raises(ValueError, match="two-byte"):
        await session.send_audio(b"\x00")
    with pytest.raises(ValueError, match="sample_rate"):
        await session.send_audio(b"\x00\x00", sample_rate=0)

    await session.send_audio(bytearray(b"\x01\x02"), sample_rate=8_000)
    await session.end_audio()

    assert connection.sent == [(b"\x01\x02", 8_000)]
    assert connection.audio_ended == 1

    await session.close()


async def test_events_normalize_audio_turn_and_interruption() -> None:
    connection = FakeConnection(
        [
            AudioEvent(data=b"\x01\x02"),
            TurnCompleteEvent(),
            InterruptedEvent(),
        ]
    )
    session, _, _ = make_session(connection)
    await session.connect()

    events = [event async for event in session.events()]

    assert events == [
        AudioEvent(data=b"\x01\x02"),
        TurnCompleteEvent(),
        InterruptedEvent(),
    ]

    await session.close()


async def test_only_one_event_consumer_is_allowed() -> None:
    connection = FakeConnection([AudioEvent(data=b"\x01\x02")])
    session, _, _ = make_session(connection)
    await session.connect()

    first_consumer = session.events()
    assert isinstance(await anext(first_consumer), AudioEvent)

    second_consumer = session.events()
    with pytest.raises(SessionStateError, match="one events"):
        await anext(second_consumer)

    await first_consumer.aclose()
    await session.close()
