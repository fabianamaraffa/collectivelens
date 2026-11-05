import pytest

from collectivelens.config import SAMPLE_LIBRARY_DIR, load_categories
from collectivelens.embedders import LiteEmbedder
from collectivelens.library import open_library


class Block:
    def __init__(self, type, text=""):
        self.type, self.text = type, text


class BadRequestError(Exception):
    """Named like the Anthropic SDK error so llm.complete treats it the same way."""


class FakeClient:
    """Stands in for anthropic.Anthropic. Records requests; can mimic a model that rejects `thinking`."""

    def __init__(self, text="The documents recommend higher grants [1].", reject_thinking=False, blocks=None, error=None):
        self.text, self.reject_thinking, self.error = text, reject_thinking, error
        self.blocks = blocks
        self.calls = []
        self.messages = self

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if self.error:
            raise self.error
        if self.reject_thinking and "thinking" in kwargs:
            raise BadRequestError("thinking is not supported")
        blocks = self.blocks or [Block("thinking", "private reasoning"), Block("text", self.text)]
        return type("Response", (), {"content": blocks, "stop_reason": "end_turn"})()


@pytest.fixture(scope="session")
def categories():
    return load_categories()


@pytest.fixture(scope="session")
def sample_lib(categories):
    return open_library(SAMPLE_LIBRARY_DIR, LiteEmbedder(), categories)


@pytest.fixture
def fake_client():
    return FakeClient
