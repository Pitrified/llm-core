"""Tests for TranscriptionConfig base class."""

from llm_core.transcription.base import BaseTranscriber
from llm_core.transcription.base import TranscriptionResult
from llm_core.transcription.config.base import TranscriptionConfig
from llm_core.transcription.config.faster_whisper import FasterWhisperConfig
from llm_core.transcription.config.openai_api import OpenAIAPIConfig
from llm_core.transcription.config.whisper import WhisperConfig
from llm_core.transcription.testing.fake import FakeTranscriber
from llm_core.transcription.testing.fake import FakeTranscriberConfig


class ConcreteTranscriptionConfig(TranscriptionConfig):
    """Minimal concrete subclass for testing the abstract base."""

    model: str = "test-model"

    def create_transcriber(self) -> BaseTranscriber:
        """Return a FakeTranscriber for testing."""
        return FakeTranscriber(
            responses=[TranscriptionResult(text="test", provider="test")]
        )


class TestTranscriptionConfigBase:
    """Tests for TranscriptionConfig base class."""

    def test_default_language_is_none(self) -> None:
        """Language defaults to None (auto-detect)."""
        cfg = ConcreteTranscriptionConfig()
        assert cfg.language is None

    def test_custom_language_stored(self) -> None:
        """Explicit language is stored and accessible."""
        cfg = ConcreteTranscriptionConfig(language="it")
        assert cfg.language == "it"

    def test_to_kw_contains_model(self) -> None:
        """to_kw includes the model field."""
        cfg = ConcreteTranscriptionConfig()
        kw = cfg.to_kw(exclude_none=True)
        assert kw["model"] == "test-model"

    def test_create_transcriber_callable(self) -> None:
        """create_transcriber is callable on a concrete subclass."""
        cfg = ConcreteTranscriptionConfig()
        assert callable(cfg.create_transcriber)


class TestWhisperConfigDefaults:
    """Tests for WhisperConfig defaults."""

    def test_default_model(self) -> None:
        """WhisperConfig defaults to 'medium' model."""
        cfg = WhisperConfig()
        assert cfg.model == "medium"

    def test_default_device(self) -> None:
        """WhisperConfig defaults to 'cpu' device."""
        cfg = WhisperConfig()
        assert cfg.device == "cpu"

    def test_default_fp16(self) -> None:
        """WhisperConfig fp16 defaults to False."""
        cfg = WhisperConfig()
        assert cfg.fp16 is False

    def test_custom_values(self) -> None:
        """WhisperConfig accepts custom model and device."""
        cfg = WhisperConfig(model="large-v3", device="cuda", fp16=True)
        assert cfg.model == "large-v3"
        assert cfg.device == "cuda"
        assert cfg.fp16 is True


class TestFasterWhisperConfigDefaults:
    """Tests for FasterWhisperConfig defaults."""

    def test_default_model(self) -> None:
        """FasterWhisperConfig defaults to 'medium' model."""
        cfg = FasterWhisperConfig()
        assert cfg.model == "medium"

    def test_default_compute_type(self) -> None:
        """FasterWhisperConfig defaults to 'int8' compute_type."""
        cfg = FasterWhisperConfig()
        assert cfg.compute_type == "int8"

    def test_default_beam_size(self) -> None:
        """FasterWhisperConfig defaults to beam_size 5."""
        cfg = FasterWhisperConfig()
        assert cfg.beam_size == 5


class TestOpenAIAPIConfigDefaults:
    """Tests for OpenAIAPIConfig defaults."""

    def test_default_model(self) -> None:
        """OpenAIAPIConfig defaults to 'whisper-1' model."""
        cfg = OpenAIAPIConfig()
        assert cfg.model == "whisper-1"

    def test_default_response_format(self) -> None:
        """OpenAIAPIConfig defaults to 'verbose_json' response_format."""
        cfg = OpenAIAPIConfig()
        assert cfg.response_format == "verbose_json"


class TestFakeTranscriberConfig:
    """Tests for FakeTranscriberConfig."""

    def test_create_transcriber_returns_fake(self) -> None:
        """create_transcriber() returns a FakeTranscriber."""
        cfg = FakeTranscriberConfig(
            responses=[TranscriptionResult(text="Hello.", language="en")]
        )
        t = cfg.create_transcriber()
        assert isinstance(t, FakeTranscriber)

    def test_default_model_is_fake(self) -> None:
        """FakeTranscriberConfig uses 'fake' as default model."""
        cfg = FakeTranscriberConfig(responses=[TranscriptionResult(text="Hi.")])
        assert cfg.model == "fake"
