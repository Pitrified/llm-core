"""Tests for FakeTranscriber and FakeTranscriberConfig."""

from pathlib import Path

import pytest

from llm_core.transcription.base import TranscriptionResult
from llm_core.transcription.base import TranscriptionSegment
from llm_core.transcription.testing.fake import FakeTranscriber
from llm_core.transcription.testing.fake import FakeTranscriberConfig


class TestFakeTranscriber:
    """Tests for FakeTranscriber."""

    def test_single_response_always_returned(self) -> None:
        """With one response, every call returns the same result."""
        result = TranscriptionResult(text="always this", language="en")
        t = FakeTranscriber(responses=[result])
        for _ in range(3):
            r = t.transcribe(Path("any.mp3"))
            assert r.text == "always this"

    def test_multiple_responses_cycled(self) -> None:
        """With multiple responses, they are cycled in round-robin order."""
        r1 = TranscriptionResult(text="first")
        r2 = TranscriptionResult(text="second")
        r3 = TranscriptionResult(text="third")
        t = FakeTranscriber(responses=[r1, r2, r3])

        texts = [t.transcribe(Path("x.mp3")).text for _ in range(6)]
        assert texts == ["first", "second", "third", "first", "second", "third"]

    def test_provider_name_is_fake(self) -> None:
        """provider_name is 'fake'."""
        t = FakeTranscriber(responses=[TranscriptionResult(text="x")])
        assert t.provider_name == "fake"

    def test_language_arg_ignored(self) -> None:
        """Language argument does not affect the returned result."""
        result = TranscriptionResult(text="pasta", language="it")
        t = FakeTranscriber(responses=[result])
        r = t.transcribe(Path("x.mp3"), language="fr")
        assert r.language == "it"

    def test_result_with_segments(self) -> None:
        """TranscriptionResult with segments is returned as-is."""
        seg = TranscriptionSegment(start=0.0, end=1.5, text="Hi.")
        result = TranscriptionResult(text="Hi.", segments=[seg])
        t = FakeTranscriber(responses=[result])
        r = t.transcribe(Path("audio.mp3"))
        assert len(r.segments) == 1
        assert r.segments[0].text == "Hi."

    @pytest.mark.asyncio
    async def test_atranscribe_delegates_to_sync(self) -> None:
        """Atranscribe returns the same result as transcribe."""
        result = TranscriptionResult(text="async test")
        t = FakeTranscriber(responses=[result])
        r = await t.atranscribe(Path("any.mp3"))
        assert r.text == "async test"

    @pytest.mark.asyncio
    async def test_async_increments_call_count(self) -> None:
        """Async and sync calls share the same round-robin counter."""
        r1 = TranscriptionResult(text="sync")
        r2 = TranscriptionResult(text="async")
        t = FakeTranscriber(responses=[r1, r2])
        s = t.transcribe(Path("x.mp3"))
        a = await t.atranscribe(Path("x.mp3"))
        assert s.text == "sync"
        assert a.text == "async"


class TestFakeTranscriberConfig:
    """Tests for FakeTranscriberConfig."""

    def test_create_transcriber_returns_fake(self) -> None:
        """create_transcriber() returns a FakeTranscriber instance."""
        cfg = FakeTranscriberConfig(
            responses=[TranscriptionResult(text="Boil the pasta.", language="en")]
        )
        t = cfg.create_transcriber()
        assert isinstance(t, FakeTranscriber)

    def test_response_is_returned_correctly(self) -> None:
        """The transcriber returned by the config produces the expected result."""
        cfg = FakeTranscriberConfig(
            responses=[TranscriptionResult(text="Boil the pasta.", language="en")]
        )
        t = cfg.create_transcriber()
        result = t.transcribe(Path("any.mp3"))
        assert result.text == "Boil the pasta."
        assert result.language == "en"

    def test_default_model_field(self) -> None:
        """FakeTranscriberConfig.model defaults to 'fake'."""
        cfg = FakeTranscriberConfig(responses=[TranscriptionResult(text="x")])
        assert cfg.model == "fake"
