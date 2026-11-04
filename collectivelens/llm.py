"""The only place that talks to Claude."""

from __future__ import annotations

import logging
import os

log = logging.getLogger(__name__)


class LLMError(RuntimeError):
    """Claude could not give a usable answer."""


def make_client(api_key: str | None = None):
    """An Anthropic client, or None when there is no key or the package is missing."""
    key = (api_key or os.getenv("ANTHROPIC_API_KEY") or "").strip()
    if not key:
        return None
    try:
        import anthropic
    except ImportError:
        return None
    return anthropic.Anthropic(api_key=key)


def _text_of(response) -> str:
    # Newer models can return a thinking block before the text. Version 1 read content[0].text
    # and would break; here only the text blocks are joined.
    return "".join(getattr(b, "text", "") for b in response.content if getattr(b, "type", "") == "text").strip()


def complete(client, *, model: str, system: str, user: str, max_tokens: int) -> str:
    """One question to Claude, returning plain text. Raises LLMError when nothing usable comes back."""
    request = dict(model=model, max_tokens=max_tokens, system=system,
                   messages=[{"role": "user", "content": user}])
    try:
        # These are short, plain tasks: thinking off keeps them fast, cheap and inside max_tokens.
        response = client.messages.create(thinking={"type": "disabled"}, **request)
    except Exception as exc:
        if exc.__class__.__name__ != "BadRequestError":
            raise LLMError(f"{exc.__class__.__name__}: {exc}") from exc
        # The model does not accept this setting: ask again without it.
        try:
            response = client.messages.create(**request)
        except Exception as exc2:
            raise LLMError(f"{exc2.__class__.__name__}: {exc2}") from exc2
    text = _text_of(response)
    if not text:
        raise LLMError(f"empty answer (stop reason: {getattr(response, 'stop_reason', 'unknown')})")
    return text
