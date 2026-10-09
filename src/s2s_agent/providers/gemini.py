"""Gemini Live provider adapter."""

from __future__ import annotations

import os
from collections.abc import AsyncIterator
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from typing import Protocol, cast

from google import genai
from google.genai import types

from ..config import LiveConfig
from ..errors import ConfigurationError
from ..events import AudioEvent, InterruptedEvent, SessionEvent, TurnCompleteEvent
from .base import S2SConnection

DEFAULT_MODEL = "gemini-3.8-live"
DEFAULT_VOICE = "Kore"
DEFAULT_API_VERSION = "v1alpha"


class _AsyncGeminiSession(Protocol):
    async def send_realtime_input(
        self,
        *,
        audio: types.Blob | None = None,
        audio_stream_end: bool | None = None,
    ) -> None: ...

    def receive(self) -> AsyncIterator[types.LiveServerMessage]: ...


class GeminiProvider:
    """Built-in provider for the Google Gemini Live API."""

    name = "gemini"

    def __init__(
        self,
        *,
        api_key: str | None = None,
        client: genai.Client | None = None,
        api_version: str = DEFAULT_API_VERSION,
    ) -> None:
        if not api_version.strip():
            raise ConfigurationError("api_version must not be empty")

        self._client = client
        self._api_key = resolve_api_key(api_key) if client is None else None
        self._api_version = api_version

    @property
    def api_version(self) -> str:
        """Google Live API version used for new clients."""

        return self._api_version

    @asynccontextmanager
    async def connect(self, config: LiveConfig) -> AsyncIterator[S2SConnection]:
        """Open and normalize one Gemini Live connection."""

        client = self._client
        if client is None:
            if self._api_key is None:
                raise ConfigurationError("Gemini API key was not resolved")
            client = genai.Client(
                api_key=self._api_key,
                http_options={"api_version": self._api_version},
            )
            self._client = client

        connection = client.aio.live.connect(
            model=config.model or DEFAULT_MODEL,
            config=_to_api_config(config),
        )
        context = cast(AbstractAsyncContextManager[_AsyncGeminiSession], connection)
        async with context as session:
            yield _GeminiConnection(session)


class _GeminiConnection:
    def __init__(self, session: _AsyncGeminiSession) -> None:
        self._session = session

    async def send_audio(self, data: bytes, *, sample_rate: int) -> None:
        blob = types.Blob(data=data, mime_type=f"audio/pcm;rate={sample_rate}")
        await self._session.send_realtime_input(audio=blob)

    async def end_audio(self) -> None:
        await self._session.send_realtime_input(audio_stream_end=True)

    async def events(self) -> AsyncIterator[SessionEvent]:
        # The Google SDK's receive iterator ends after each model turn, so open
        # a fresh iterator for every subsequent turn.
        while True:
            received_message = False
            async for response in self._session.receive():
                received_message = True
                for event in _events_from_response(response):
                    yield event

            if not received_message:
                break


def resolve_api_key(api_key: str | None) -> str:
    """Resolve an explicit key or the standard Gemini environment variable."""

    candidate = api_key if api_key is not None else os.getenv("GEMINI_API_KEY")
    if candidate is None or not candidate.strip():
        raise ConfigurationError("Gemini API key is required; pass api_key or set GEMINI_API_KEY")
    return candidate.strip()


def _to_api_config(config: LiveConfig) -> types.LiveConnectConfig:
    return types.LiveConnectConfig(
        response_modalities=[types.Modality.AUDIO],
        speech_config=types.SpeechConfig(
            voice_config=types.VoiceConfig(
                prebuilt_voice_config=types.PrebuiltVoiceConfig(
                    voice_name=config.voice or DEFAULT_VOICE
                )
            )
        ),
        # A plain string matches Google's Live API examples. Supplying a
        # Content object with role="system" can make the service close the
        # WebSocket with an opaque 1011 internal error.
        system_instruction=config.instructions,
    )


def _events_from_response(response: types.LiveServerMessage) -> tuple[SessionEvent, ...]:
    server_content = response.server_content
    if server_content is None:
        return ()

    events: list[SessionEvent] = []
    interrupted = bool(server_content.interrupted)
    if interrupted:
        events.append(InterruptedEvent())
    else:
        model_turn = server_content.model_turn
        if model_turn is not None and model_turn.parts is not None:
            for part in model_turn.parts:
                inline_data = part.inline_data
                if inline_data is not None and isinstance(inline_data.data, bytes):
                    events.append(AudioEvent(data=inline_data.data))

    if server_content.turn_complete:
        events.append(TurnCompleteEvent())

    return tuple(events)
