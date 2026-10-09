from __future__ import annotations

from types import SimpleNamespace
from typing import cast

import pytest
from google import genai
from google.genai import types

from s2s_agent import (
    DEFAULT_MODEL,
    DEFAULT_VOICE,
    AudioEvent,
    ConfigurationError,
    GeminiProvider,
    InterruptedEvent,
    LiveConfig,
    TurnCompleteEvent,
)
from s2s_agent.providers.gemini import resolve_api_key


class FakeGoogleSession:
    def __init__(self, turns: list[list[types.LiveServerMessage]] | None = None) -> None:
        self.turns = turns or []
        self.sent: list[dict[str, object]] = []

    async def send_realtime_input(self, **kwargs: object) -> None:
        self.sent.append(kwargs)

    def receive(self):
        responses = self.turns.pop(0) if self.turns else []

        async def stream():
            for response in responses:
                yield response

        return stream()


class FakeGoogleContext:
    def __init__(self, session: FakeGoogleSession) -> None:
        self.session = session
        self.entered = 0
        self.exited = 0

    async def __aenter__(self) -> FakeGoogleSession:
        self.entered += 1
        return self.session

    async def __aexit__(self, exc_type, exc, traceback) -> None:
        self.exited += 1


class FakeLiveAPI:
    def __init__(self, context: FakeGoogleContext) -> None:
        self.context = context
        self.calls: list[dict[str, object]] = []

    def connect(self, **kwargs: object) -> FakeGoogleContext:
        self.calls.append(kwargs)
        return self.context


class FakeClient:
    def __init__(self, session: FakeGoogleSession) -> None:
        self.context = FakeGoogleContext(session)
        self.live_api = FakeLiveAPI(self.context)
        self.aio = SimpleNamespace(live=self.live_api)


def fake_response(
    *,
    audio: bytes | None = None,
    interrupted: bool = False,
    turn_complete: bool = False,
) -> types.LiveServerMessage:
    parts = []
    if audio is not None:
        parts.append(SimpleNamespace(inline_data=SimpleNamespace(data=audio)))
    model_turn = SimpleNamespace(parts=parts) if parts else None
    server_content = SimpleNamespace(
        interrupted=interrupted,
        model_turn=model_turn,
        turn_complete=turn_complete,
    )
    return cast(
        types.LiveServerMessage,
        SimpleNamespace(server_content=server_content),
    )


async def test_gemini_provider_maps_config_audio_and_events() -> None:
    google_session = FakeGoogleSession(
        turns=[
            [fake_response(audio=b"\x01\x02", turn_complete=True)],
            [fake_response(interrupted=True)],
        ]
    )
    client = FakeClient(google_session)
    provider = GeminiProvider(client=cast(genai.Client, client))
    config = LiveConfig(
        model="custom-live-model",
        voice="Puck",
        instructions="Be concise.",
    )

    async with provider.connect(config) as connection:
        await connection.send_audio(b"\x03\x04", sample_rate=8_000)
        await connection.end_audio()
        events = [event async for event in connection.events()]

    call = client.live_api.calls[0]
    assert call["model"] == "custom-live-model"
    api_config = call["config"]
    assert isinstance(api_config, types.LiveConnectConfig)
    assert api_config.system_instruction == "Be concise."
    assert api_config.speech_config is not None
    voice_config = api_config.speech_config.voice_config
    assert voice_config is not None
    assert voice_config.prebuilt_voice_config is not None
    assert voice_config.prebuilt_voice_config.voice_name == "Puck"

    blob = google_session.sent[0]["audio"]
    assert isinstance(blob, types.Blob)
    assert blob.data == b"\x03\x04"
    assert blob.mime_type == "audio/pcm;rate=8000"
    assert google_session.sent[1] == {"audio_stream_end": True}
    assert events == [
        AudioEvent(data=b"\x01\x02"),
        TurnCompleteEvent(),
        InterruptedEvent(),
    ]
    assert client.context.entered == 1
    assert client.context.exited == 1


async def test_gemini_provider_uses_defaults_and_creates_client(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_client = FakeClient(FakeGoogleSession())
    client_calls: list[dict[str, object]] = []

    def create_client(**kwargs: object) -> FakeClient:
        client_calls.append(kwargs)
        return fake_client

    monkeypatch.setattr(genai, "Client", create_client)
    provider = GeminiProvider(api_key="test-key", api_version="v1alpha")

    async with provider.connect(LiveConfig()):
        pass

    assert client_calls == [
        {
            "api_key": "test-key",
            "http_options": {"api_version": "v1alpha"},
        }
    ]
    call = fake_client.live_api.calls[0]
    assert call["model"] == DEFAULT_MODEL
    api_config = call["config"]
    assert isinstance(api_config, types.LiveConnectConfig)
    assert api_config.speech_config is not None
    voice_config = api_config.speech_config.voice_config
    assert voice_config is not None
    assert voice_config.prebuilt_voice_config is not None
    assert voice_config.prebuilt_voice_config.voice_name == DEFAULT_VOICE


def test_resolve_api_key_prefers_explicit_value(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "environment-key")

    assert resolve_api_key(" explicit-key ") == "explicit-key"


def test_resolve_api_key_uses_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", " environment-key ")

    assert resolve_api_key(None) == "environment-key"


def test_gemini_provider_rejects_missing_key_and_invalid_version(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)

    with pytest.raises(ConfigurationError, match="GEMINI_API_KEY"):
        GeminiProvider()
    with pytest.raises(ConfigurationError, match="api_version"):
        GeminiProvider(api_key="test-key", api_version=" ")
