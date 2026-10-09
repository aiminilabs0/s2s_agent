"""Default microphone and speaker support for live sessions."""

from __future__ import annotations

import asyncio
import sys
import warnings
from array import array
from importlib import import_module
from threading import Lock
from typing import TYPE_CHECKING, Any, cast

from .config import CHANNELS, OUTPUT_SAMPLE_RATE
from .events import AudioEvent, InterruptedEvent, TurnCompleteEvent

if TYPE_CHECKING:
    from .session import AgentTask, LiveSession

DEFAULT_INPUT_BLOCK_FRAMES = 800


def _sounddevice() -> Any:
    try:
        return cast(Any, import_module("sounddevice"))
    except ModuleNotFoundError as exc:
        if exc.name != "sounddevice":
            raise
        raise RuntimeError(
            "Default audio requires the audio extra. "
            "Install it with: pip install 's2s-agent[audio]'"
        ) from exc


async def run_default_audio(
    session: LiveSession,
    *,
    input_block_frames: int = DEFAULT_INPUT_BLOCK_FRAMES,
    echo_cancellation: bool = True,
    echo_delay_ms: int = 0,
) -> None:
    """Run a session using the system's default microphone and speaker.

    Echo cancellation uses WebRTC AEC3 and keeps full-duplex interruption
    support. ``echo_delay_ms`` can be tuned for the speaker-to-microphone
    latency; zero lets AEC3 estimate the delay automatically.
    """

    if input_block_frames <= 0:
        raise ValueError("input_block_frames must be greater than zero")
    if echo_delay_ms < 0:
        raise ValueError("echo_delay_ms must not be negative")

    sd = _sounddevice()
    if echo_cancellation:
        await _run_echo_cancelled_audio(
            session,
            sd,
            input_block_frames=input_block_frames,
            echo_delay_ms=echo_delay_ms,
        )
        return

    await _run_basic_audio(session, sd, input_block_frames=input_block_frames)


async def _run_basic_audio(
    session: LiveSession,
    sd: Any,
    *,
    input_block_frames: int,
) -> None:
    with (
        sd.RawInputStream(
            samplerate=session.config.input_sample_rate,
            channels=CHANNELS,
            dtype="int16",
        ) as microphone,
        sd.RawOutputStream(
            samplerate=OUTPUT_SAMPLE_RATE,
            channels=CHANNELS,
            dtype="int16",
        ) as speaker,
    ):
        async with session:
            await _run_audio_tasks(
                session,
                _send_microphone(session, microphone, input_block_frames),
                _play_responses(session, speaker),
            )


async def _run_echo_cancelled_audio(
    session: LiveSession,
    sd: Any,
    *,
    input_block_frames: int,
    echo_delay_ms: int,
) -> None:
    loop = asyncio.get_running_loop()
    capture_queue: asyncio.Queue[bytes] = asyncio.Queue()
    callback_failure: asyncio.Future[None] = loop.create_future()
    playback = _PlaybackBuffer()
    stream_sample_rate = OUTPUT_SAMPLE_RATE
    stream_block_frames = max(
        1,
        round(input_block_frames * stream_sample_rate / session.config.input_sample_rate),
    )
    capture_resampler = _Pcm16Resampler(
        stream_sample_rate,
        session.config.input_sample_rate,
    )
    canceller = _WebRTCEchoCanceller(
        sample_rate=stream_sample_rate,
        stream_delay_ms=echo_delay_ms,
    )

    def callback(indata: Any, outdata: Any, frames: int, _time: Any, status: Any) -> None:
        try:
            if status:
                loop.call_soon_threadsafe(_warn_stream_status, str(status))

            byte_count = frames * CHANNELS * 2
            far_audio = playback.read(byte_count)
            outdata[:] = far_audio
            clean_audio = capture_resampler.process(canceller.process(bytes(indata), far_audio))
            if clean_audio:
                loop.call_soon_threadsafe(capture_queue.put_nowait, clean_audio)
        except Exception as exc:
            loop.call_soon_threadsafe(_set_callback_failure, callback_failure, exc)
            raise sd.CallbackAbort from exc

    stream = sd.RawStream(
        samplerate=stream_sample_rate,
        blocksize=stream_block_frames,
        channels=CHANNELS,
        dtype="int16",
        callback=callback,
    )

    async with session:
        with stream:
            await _run_audio_tasks(
                session,
                _send_processed_microphone(session, capture_queue),
                _play_echo_cancelled_responses(
                    session,
                    playback,
                    output_sample_rate=stream_sample_rate,
                ),
                _wait_for_callback_failure(callback_failure),
            )


async def _run_audio_tasks(
    session: LiveSession,
    *audio_coroutines: Any,
) -> None:
    tasks = {asyncio.create_task(coroutine) for coroutine in audio_coroutines}
    tasks.update(asyncio.create_task(_run_agent_task(task, session)) for task in session.tasks)
    try:
        done, _ = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
        for task in done:
            task.result()
    finally:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)


async def _send_microphone(
    session: LiveSession,
    microphone: Any,
    input_block_frames: int,
) -> None:
    while True:
        data, overflowed = await asyncio.to_thread(microphone.read, input_block_frames)
        if overflowed:
            warnings.warn("microphone input overflowed", RuntimeWarning, stacklevel=2)
        await session.send_audio(bytes(data))


async def _send_processed_microphone(
    session: LiveSession,
    capture_queue: asyncio.Queue[bytes],
) -> None:
    while True:
        await session.send_audio(await capture_queue.get())


async def _play_responses(session: LiveSession, speaker: Any) -> None:
    async for event in session.events():
        if isinstance(event, AudioEvent):
            await asyncio.to_thread(speaker.write, event.data)


async def _play_echo_cancelled_responses(
    session: LiveSession,
    playback: _PlaybackBuffer,
    *,
    output_sample_rate: int,
) -> None:
    resampler: _Pcm16Resampler | None = None

    async for event in session.events():
        if isinstance(event, AudioEvent):
            if resampler is None or resampler.input_sample_rate != event.sample_rate:
                resampler = _Pcm16Resampler(event.sample_rate, output_sample_rate)
            playback.append(resampler.process(event.data))
        elif isinstance(event, InterruptedEvent):
            playback.clear()
            resampler = None
        elif isinstance(event, TurnCompleteEvent):
            resampler = None


async def _run_agent_task(task: AgentTask, session: LiveSession) -> None:
    await task(session)
    await asyncio.Future()


async def _wait_for_callback_failure(failure: asyncio.Future[None]) -> None:
    await failure


def _set_callback_failure(failure: asyncio.Future[None], exc: Exception) -> None:
    if not failure.done():
        failure.set_exception(exc)


def _warn_stream_status(status: str) -> None:
    warnings.warn(f"audio stream status: {status}", RuntimeWarning, stacklevel=2)


class _PlaybackBuffer:
    def __init__(self) -> None:
        self._data = bytearray()
        self._lock = Lock()

    def append(self, data: bytes) -> None:
        with self._lock:
            self._data.extend(data)

    def read(self, byte_count: int) -> bytes:
        with self._lock:
            available = min(byte_count, len(self._data))
            data = bytes(self._data[:available])
            del self._data[:available]
        return data.ljust(byte_count, b"\0")

    def clear(self) -> None:
        with self._lock:
            self._data.clear()


class _Pcm16Resampler:
    """Small streaming linear resampler for mono little-endian PCM."""

    def __init__(self, input_sample_rate: int, output_sample_rate: int) -> None:
        if input_sample_rate <= 0 or output_sample_rate <= 0:
            raise ValueError("sample rates must be greater than zero")
        self.input_sample_rate = input_sample_rate
        self.output_sample_rate = output_sample_rate
        self._source_samples = 0
        self._next_output_sample = 0
        self._previous_sample: int | None = None

    def process(self, data: bytes) -> bytes:
        if len(data) % 2:
            raise ValueError("16-bit PCM audio data must contain complete two-byte samples")
        if not data or self.input_sample_rate == self.output_sample_rate:
            return data

        samples = array("h")
        samples.frombytes(data)
        if sys.byteorder != "little":
            samples.byteswap()

        start = self._source_samples
        end = start + len(samples)
        output = array("h")
        input_rate = self.input_sample_rate
        output_rate = self.output_sample_rate

        while self._next_output_sample * input_rate <= (end - 1) * output_rate:
            position = self._next_output_sample * input_rate
            left_index, fraction = divmod(position, output_rate)
            left = self._sample_at(left_index, start, samples)
            if fraction:
                right = self._sample_at(left_index + 1, start, samples)
                value = round((left * (output_rate - fraction) + right * fraction) / output_rate)
            else:
                value = left
            output.append(value)
            self._next_output_sample += 1

        self._source_samples = end
        self._previous_sample = samples[-1]
        if sys.byteorder != "little":
            output.byteswap()
        return output.tobytes()

    def _sample_at(self, index: int, start: int, samples: array[int]) -> int:
        if index == start - 1 and self._previous_sample is not None:
            return self._previous_sample
        return samples[index - start]


class _WebRTCEchoCanceller:
    def __init__(self, *, sample_rate: int, stream_delay_ms: int) -> None:
        try:
            numpy = import_module("numpy")
            module = import_module("pywebrtc_audio")
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "Echo cancellation requires the audio extra. "
                "Install it with: pip install 's2s-agent[audio]'"
            ) from exc

        processor_type = module.AudioProcessor
        self._numpy = numpy
        self._processor = processor_type(
            sample_rate=sample_rate,
            num_channels=CHANNELS,
            echo_cancellation=True,
            stream_delay_ms=stream_delay_ms,
        )

    def process(self, near_audio: bytes, far_audio: bytes) -> bytes:
        if len(near_audio) != len(far_audio):
            raise ValueError("capture and playback reference audio must have equal lengths")

        near = self._numpy.frombuffer(near_audio, dtype=self._numpy.int16)
        far = self._numpy.frombuffer(far_audio, dtype=self._numpy.int16)
        return cast(bytes, self._processor.process(near, far).tobytes())
