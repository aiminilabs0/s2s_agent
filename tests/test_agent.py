from __future__ import annotations

import pytest

from s2s_agent import ConfigurationError, LiveSession, create_agent


class FakeProvider:
    name = "fake"


def test_create_agent_applies_common_customizations() -> None:
    provider = FakeProvider()

    async def background_task(session: LiveSession) -> None:
        pass

    agent = create_agent(
        instructions="Teach with short examples.",
        voice="Puck",
        model="custom-model",
        input_sample_rate=8_000,
        provider=provider,  # type: ignore[arg-type]
        tasks=[background_task],
    )

    assert agent.config.instructions == "Teach with short examples."
    assert agent.config.voice == "Puck"
    assert agent.config.model == "custom-model"
    assert agent.config.input_sample_rate == 8_000
    assert agent.provider is provider
    assert agent.tasks == (background_task,)


def test_create_agent_validates_provider_credentials() -> None:
    with pytest.raises(ConfigurationError, match="Gemini shortcuts"):
        create_agent(
            provider=FakeProvider(),  # type: ignore[arg-type]
            api_key="test-key",
        )
