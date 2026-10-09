from __future__ import annotations

import pytest

from s2s_agent import ConfigurationError, LiveConfig


def test_default_config_is_provider_neutral() -> None:
    config = LiveConfig(instructions="Be concise.")

    assert config.model is None
    assert config.voice is None
    assert config.input_mime_type == "audio/pcm;rate=16000"
    assert config.instructions == "Be concise."


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"model": " "}, "model"),
        ({"voice": ""}, "voice"),
        ({"instructions": "  "}, "instructions"),
        ({"input_sample_rate": 0}, "input_sample_rate"),
    ],
)
def test_invalid_config_is_rejected(kwargs: dict[str, object], message: str) -> None:
    with pytest.raises(ConfigurationError, match=message):
        LiveConfig(**kwargs)  # type: ignore[arg-type]
