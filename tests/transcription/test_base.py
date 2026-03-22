"""Tests for TranscriptionResult, TranscriptionSegment, and BaseTranscriber protocol."""

from pathlib import Path

from llm_core.transcription.base import BaseTranscriber
from llm_core.transcription.base import TranscriptionResult
from llm_core.transcription.base import TranscriptionSegment
from llm_core.transcription.testing.fake import FakeTranscriber


class TestTranscriptionSegment:
    """Tests for the TranscriptionSegment dataclass."""

    def test_basic_construction(self) -> None:
        """TranscriptionSegment stores start, end, and text."""
        seg = TranscriptionSegment(start=0.0, end=3.5, text="Hello world.")
        assert seg.start == 0.0
        assert seg.end == 3.5
        assert seg.text == "Hello world."


class TestTranscriptionResult:
    """Tests for the TranscriptionResult dataclass."""

    def test_minimal_construction(self) -> None:
        """Only 'text' is required; other fields use defaults."""
        result = TranscriptionResult(text="Pasta is ready.")
        assert result.text == "Pasta is ready."
        assert result.language is None
        assert result.segments == []
        assert result.provider == ""

    def test_full_construction(self) -> None:
        """All fields are stored correctly."""
        seg = TranscriptionSegment(start=0.0, end=2.0, text="Hi.")
        result = TranscriptionResult(
            text="Hi.",
            language="en",
            segments=[seg],
            provider="whisper",
        )
        assert result.language == "en"
        assert len(result.segments) == 1
        assert result.segments[0].text == "Hi."
        assert result.provider == "whisper"


class TestBaseTranscriberProtocol:
    """Tests that FakeTranscriber satisfies BaseTranscriber via isinstance."""

    def test_fake_transcriber_satisfies_protocol(self) -> None:
        """FakeTranscriber is recognised as a BaseTranscriber at runtime."""
        t = FakeTranscriber(responses=[TranscriptionResult(text="ok")])
        assert isinstance(t, BaseTranscriber)

    def test_provider_name_present(self) -> None:
        """BaseTranscriber implementors expose a provider_name attribute."""
        t = FakeTranscriber(responses=[TranscriptionResult(text="ok")])
        assert isinstance(t.provider_name, str)

    def test_transcribe_callable(self) -> None:
        """Transcribe is callable and returns a TranscriptionResult."""
        t = FakeTranscriber(responses=[TranscriptionResult(text="hi")])
        result = t.transcribe(Path("dummy.mp3"))
        assert isinstance(result, TranscriptionResult)
