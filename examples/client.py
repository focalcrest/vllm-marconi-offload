"""Minimal client showing prefix-cache reuse against a server launched
via ``examples/serve_example.sh``.

Sends two requests with the same long system prompt. The first one
warms the cache (HBM, then L1 if needed); the second hits prefix
cache. Watch the server log for ``[CPU-OFFLOAD HIT]`` and ``Marconi
eviction: stats=`` lines to see the connector working.
"""

from __future__ import annotations

import time

from openai import OpenAI

SYSTEM_PROMPT = (
    "You are an automation assistant. Today's date is 2026-05-24. "
    "When the user asks for a tool call, respond with exactly that tool "
    "call and nothing else. Otherwise answer concisely."
) * 30  # ~ a few thousand tokens of shared system prefix

QUERIES = [
    "What's the capital of France?",
    "How do you spell 'occurrence'?",
]


def main() -> None:
    client = OpenAI(base_url="http://127.0.0.1:8000/v1", api_key="EMPTY")
    model = client.models.list().data[0].id
    print(f"using model: {model}")

    for i, q in enumerate(QUERIES, 1):
        t0 = time.perf_counter()
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": q},
            ],
            temperature=0.0,
            max_tokens=64,
        )
        dt = (time.perf_counter() - t0) * 1000
        cached = getattr(resp.usage, "prompt_tokens_details", None)
        cached_n = cached.cached_tokens if cached else None
        print(
            f"q{i}  {dt:6.0f} ms  prompt={resp.usage.prompt_tokens}  "
            f"compl={resp.usage.completion_tokens}  cached={cached_n}"
        )


if __name__ == "__main__":
    main()
