"""Talk to a live speech agent through your default microphone and speakers."""

from __future__ import annotations

import argparse

from s2s_agent import (
    DEFAULT_MODEL,
    create_agent,
)


def run(args: argparse.Namespace) -> None:
    agent = create_agent(
        model=args.model,
        voice=args.voice,
        instructions=args.instructions,
    )
    print("Connecting. Speak into your microphone; press Ctrl+C to stop.")
    agent.run(
        echo_cancellation=not args.no_echo_cancellation,
        echo_delay_ms=args.echo_delay_ms,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--voice", default="Kore")
    parser.add_argument(
        "--instructions",
        default="You are a concise and friendly voice assistant.",
    )
    parser.add_argument(
        "--no-echo-cancellation",
        action="store_true",
        help="Disable WebRTC acoustic echo cancellation.",
    )
    parser.add_argument(
        "--echo-delay-ms",
        type=int,
        default=0,
        help="Speaker-to-microphone delay hint; 0 enables automatic estimation.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    try:
        run(parse_args())
    except KeyboardInterrupt:
        print("\nDisconnected.")
