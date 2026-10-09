from __future__ import annotations

import asyncio
import struct
from collections.abc import AsyncIterator, Callable
from types import TracebackType
from typing import Any

import pytest

from s2s_agent import AudioEvent, LiveConfig, SessionEvent
from s2s_agent.audio import _Pcm16Resampler, run_default_audio


class FakeStream:
    def __init__(self) -> None:
        self.entered = 0
        self.exited = 0

    def __enter__(self) -> FakeStream:
        self.entered += 1
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.exited += 1


class FakeMicrophone(FakeStream):
    def __init__(self) -> None:
        super().__init__()
        self.frames: list[int] = []

    def read(self, frames: int) -> tuple[bytes, bool]:
        self.frames.append(frames)
        return b"\x00\x00", False


class FakeSpeaker(FakeStream):
    def __init__(self) -> None:
        super().__init__()
        self.writes: list[bytes] = []

    def write(self, data: bytes) -> None:
        self.writes.append(data)


class FakeDuplex(FakeStream):
    def __init__(self, callback: Callable[..., None]) -> None:
        super().__init__()
        self.callback = callback
        self.outputs: list[bytes] = []

    def __enter__(self) -> FakeDuplex:
        super().__enter__()
        self.pump(b"\x00\x00")
        return self

    def pump(self, input_data: bytes) -> None:
        output_data = bytearray(len(input_data))
        self.callback(input_data, output_data, len(input_data) // 2, None, None)
        self.outputs.append(bytes(output_data))


class FakeSoundDevice:
    def __init__(self) -> None:
        self.microphone = FakeMicrophone()
        self.speaker = FakeSpeaker()
        self.duplex: FakeDuplex | None = None
        self.input_options: dict[str, Any] = {}
        self.output_options: dict[str, Any] = {}
        self.duplex_options: dict[str, Any] = {}
        self.CallbackAbort = RuntimeError

    def RawInputStream(self, **options: Any) -> FakeMicrophone:
        self.input_options = options
        return self.microphone

    def RawOutputStream(self, **options: Any) -> FakeSpeaker:
        self.output_options = options
        return self.speaker

    def RawStream(self, **options: Any) -> FakeDuplex:
        self.duplex_options = options
        self.duplex = FakeDuplex(options["callback"])
        return self.duplex


class FakeSession:
    def __init__(self) -> None:
        self.config = LiveConfig(input_sample_rate=8_000)
        self.entered = 0
        self.exited = 0
        self.sent: list[bytes] = []
        self.audio_sent = asyncio.Event()
        self.task_started = asyncio.Event()
        self.task_stopped = asyncio.Event()
        self.tasks: tuple[Any, ...] = ()
        self.response_audio = b"\x01\x02"
        self.after_response: Callable[[], None] | None = None

    async def __aenter__(self) -> FakeSession:
        self.entered += 1
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.exited += 1

    async def send_audio(self, data: bytes) -> None:
        self.sent.append(data)
        self.audio_sent.set()
        await asyncio.Event().wait()

    async def events(self) -> AsyncIterator[SessionEvent]:
        await self.audio_sent.wait()
        if self.tasks:
            await self.task_started.wait()
        yield AudioEvent(data=self.response_audio)
        if self.after_response is not None:
            self.after_response()


async def test_run_default_audio_manages_streams_and_session(monkeypatch) -> None:
    sounddevice = FakeSoundDevice()
    session = FakeSession()
    task_calls: list[FakeSession] = []

    async def background_task(target: FakeSession) -> None:
        task_calls.append(target)
        target.task_started.set()
        try:
            await asyncio.Event().wait()
        finally:
            target.task_stopped.set()

    session.tasks = (background_task,)
    monkeypatch.setattr("s2s_agent.audio._sounddevice", lambda: sounddevice)

    await run_default_audio(
        session,
        input_block_frames=400,
        echo_cancellation=False,
    )

    assert sounddevice.input_options == {
        "samplerate": 8_000,
        "channels": 1,
        "dtype": "int16",
    }
    assert sounddevice.output_options == {
        "samplerate": 24_000,
        "channels": 1,
        "dtype": "int16",
    }
    assert sounddevice.microphone.entered == 1
    assert sounddevice.microphone.exited == 1
    assert sounddevice.microphone.frames == [400]
    assert sounddevice.speaker.writes == [b"\x01\x02"]
    assert sounddevice.speaker.entered == 1
    assert sounddevice.speaker.exited == 1
    assert session.entered == 1
    assert session.exited == 1
    assert session.sent == [b"\x00\x00"]
    assert task_calls == [session]
    assert session.task_stopped.is_set()


async def test_run_default_audio_applies_echo_cancellation(monkeypatch) -> None:
    sounddevice = FakeSoundDevice()
    session = FakeSession()
    session.response_audio = struct.pack("<6h", 100, 200, 300, 400, 500, 600)
    processor_calls: list[tuple[bytes, bytes]] = []

    class FakeEchoCanceller:
        def __init__(self, *, sample_rate: int, stream_delay_ms: int) -> None:
            assert sample_rate == 24_000
            assert stream_delay_ms == 25

        def process(self, near_audio: bytes, far_audio: bytes) -> bytes:
            processor_calls.append((near_audio, far_audio))
            return near_audio

    def pump_response_audio() -> None:
        assert sounddevice.duplex is not None
        sounddevice.duplex.pump(b"\x03\x00\x04\x00")

    session.after_response = pump_response_audio
    monkeypatch.setattr("s2s_agent.audio._sounddevice", lambda: sounddevice)
    monkeypatch.setattr("s2s_agent.audio._WebRTCEchoCanceller", FakeEchoCanceller)

    await run_default_audio(session, input_block_frames=400, echo_delay_ms=25)

    assert sounddevice.duplex_options["samplerate"] == 24_000
    assert sounddevice.duplex_options["blocksize"] == 1_200
    assert sounddevice.duplex_options["channels"] == 1
    assert sounddevice.duplex_options["dtype"] == "int16"
    assert processor_calls == [
        (b"\x00\x00", b"\x00\x00"),
        (b"\x03\x00\x04\x00", struct.pack("<2h", 100, 200)),
    ]
    assert sounddevice.duplex is not None
    assert sounddevice.duplex.outputs == [
        b"\x00\x00",
        struct.pack("<2h", 100, 200),
    ]


def test_pcm16_resampler_preserves_chunk_boundaries() -> None:
    resampler = _Pcm16Resampler(24_000, 16_000)

    first = resampler.process(struct.pack("<2h", 0, 300))
    second = resampler.process(struct.pack("<4h", 600, 900, 1_200, 1_500))

    assert struct.unpack("<h", first) == (0,)
    assert struct.unpack("<3h", second) == (450, 900, 1_350)


async def test_run_default_audio_rejects_negative_echo_delay() -> None:
    with pytest.raises(ValueError, match="must not be negative"):
        await run_default_audio(FakeSession(), echo_delay_ms=-1)
