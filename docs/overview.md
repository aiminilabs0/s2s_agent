# Introduction

> A small async Python SDK for building real-time speech-to-speech agents.

`s2s-agent` streams microphone audio to a live speech model and exposes the
response as typed Python events. It handles connection setup, audio transport,
turn completion, interruption, and cleanup so your application can focus on the
conversation.

Every session includes:

- **Bidirectional audio streaming** over one live WebSocket connection
- **Pluggable providers** behind a small structural interface
- **Typed events** for audio, completed turns, and interruptions
- **Configurable behavior** through model, voice, and instruction settings
- **Async lifecycle management** with an async context manager
- **Raw PCM access** for custom microphone, speaker, file, or transport layers

## Get started

To start building, you need:

- **Python 3.11 or newer**
- **Gemini API key** available from Google AI Studio
- **Audio dependencies** for the microphone and speaker example

```bash
git clone https://github.com/aiminilabs0/s2s_agent.git
cd s2s_agent
python -m pip install -e ".[audio]"
export GEMINI_API_KEY="your-api-key"
```

Run your first live conversation:

```bash
python examples/talk.py
```

Speak through your default microphone. The response plays through your default
speaker. WebRTC AEC3 removes that speaker signal from the microphone stream
without disabling interruptions. Press `Ctrl+C` to stop.

## Build

### Create and customize an agent

Pass the common model settings directly to `create_agent()`:

```python
from s2s_agent import create_agent

agent = create_agent(
    instructions="Answer clearly and briefly.",
    voice="Kore",
)
```

### Live session

For a local conversation, `run()` manages the default microphone, speaker,
acoustic echo cancellation, connection, and audio tasks: `agent.run()`.

Echo cancellation is enabled by default. AEC3 estimates acoustic delay
automatically, but hardware that needs a hint can use
`agent.run(echo_delay_ms=40)`. Disable processing with
`agent.run(echo_cancellation=False)`, typically only when using headphones.

Use `await agent.run_async()` from an existing event loop. For custom audio
sources and destinations, use the low-level session API: send input with
`send_audio()` and consume model output with `events()`.

```python
async with agent as session:
    await session.send_audio(pcm_bytes)

    async for event in session.events():
        ...
```

### Concurrent tasks

Pass async callbacks to `create_agent()` or add them before the session starts.
Each callback receives the connected session:

```python
import asyncio

from s2s_agent import create_agent


async def report_status(session):
    while session.connected:
        print("Agent is connected")
        await asyncio.sleep(10)


agent = create_agent(tasks=[report_status])
agent.run()
```

Tasks run only with `run()` and `run_async()`. An exception in a task stops the
session; remaining tasks are cancelled when the session ends. The default audio
runner owns the single `events()` consumer.

### Session events

- **`AudioEvent`** — signed 16-bit PCM response audio with format metadata
- **`TurnCompleteEvent`** — the current model response is complete
- **`InterruptedEvent`** — user speech interrupted the current response

Input audio defaults to 16 kHz mono signed 16-bit PCM. Gemini returns 24 kHz
mono signed 16-bit PCM.

## Providers

`LiveSession` delegates transport-specific behavior to an `S2SProvider`.
Gemini is built in and remains the default:

```python
from s2s_agent import GeminiProvider, LiveConfig, LiveSession

provider = GeminiProvider(
    api_key="your-api-key",
    api_version="v1alpha",
)
session = LiveSession(
    provider=provider,
    config=LiveConfig(voice="Kore"),
)
```

If no provider is supplied, `LiveSession` creates `GeminiProvider` and reads
`GEMINI_API_KEY`. The existing `LiveSession(api_key=..., client=...)` arguments
remain available as Gemini shortcuts. Configure the Google API version with
`GeminiProvider(api_version=...)` rather than `LiveConfig`.

### Add a provider

Providers use structural typing, so adapters do not need to inherit a base
class. Implement:

- `S2SProvider.name`
- `S2SProvider.connect(config)`, returning an async context manager
- `S2SConnection.send_audio(data, sample_rate=...)`
- `S2SConnection.end_audio()`
- `S2SConnection.events()`, yielding normalized `SessionEvent` values

```python
from contextlib import asynccontextmanager

from s2s_agent import LiveSession


class MyProvider:
    name = "my-provider"

    @asynccontextmanager
    async def connect(self, config):
        connection = await MyConnection.open(config)
        try:
            yield connection
        finally:
            await connection.close()


session = LiveSession(provider=MyProvider())
```

The adapter owns authentication, provider-specific message conversion, and
connection cleanup. `LiveSession` owns state validation, send serialization,
and single-consumer event access.

## Architecture

![s2s-agent architecture](architecture.svg)

`create_agent()` builds the provider-neutral `LiveSession` at the center of the
SDK. `run()` and `run_async()` add the default microphone and speaker runner;
custom clients can bypass that layer and exchange raw PCM and typed events
directly with the session. The injected provider opens the backend connection,
sends provider-specific audio messages, and normalizes responses into
`AudioEvent`, `InterruptedEvent`, and `TurnCompleteEvent` values.

## Example use cases

### Voice assistants

Build hands-free assistants for desktop tools, devices, or internal workflows.

### Conversational tutors

Create spoken practice sessions with custom instructions and voices.

### Interview and research tools

Stream real-time conversations into applications that manage prompts, notes,
and downstream analysis.

### Custom audio clients

Connect the session to files, media pipelines, or another transport instead of
the included microphone example.

## Current scope

The SDK focuses on one speech-to-speech session and ships with a Gemini adapter.
It does not provide rooms, WebRTC routing, telephony, tool calls, deployment
infrastructure, or automatic session reconnection.

## Next steps

- [Install and run the SDK](../README.md)
- [Explore the microphone example](../examples/talk.py)
- [Review the session API](../src/s2s_agent/session.py)
- [Review the provider interfaces](../src/s2s_agent/providers/base.py)
- [Review the Gemini adapter](../src/s2s_agent/providers/gemini.py)
- [Review configuration options](../src/s2s_agent/config.py)
