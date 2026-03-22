"""Deterministic transcriber for unit tests.

Provides ``FakeTranscriber`` and ``FakeTranscriberConfig`` so that consumers
can exercise transcription-dependent code without loading any real model or
making any I/O calls.

Example:
    ::

        from pathlib import Path

        from llm_core.transcription.base import TranscriptionResult
        from llm_core.transcription.testing.fake import FakeTranscriberConfig

        config = FakeTranscriberConfig(
            responses=[TranscriptionResult(text="Boil the pasta.", language="en")]
        )
        transcriber = config.create_transcriber()
        result = transcriber.transcribe(Path("any.mp3"))
        assert result.text == "Boil the pasta."
"""

from __future__ import annotations

from pathlib import Path  # noqa: TC003

from pydantic import ConfigDict

from llm_core.transcription.base import TranscriptionResult  # noqa: TC001
from llm_core.transcription.config.base import TranscriptionConfig


class FakeTranscriber:
    """Deterministic transcriber for unit tests. No model loaded, no I/O.

    Cycles through *responses* in round-robin order, mirroring
    ``FakeChatModel``.

    Attributes:
        provider_name: Always "fake".
    """

    provider_name: str = "fake"

    def __init__(self, responses: list[TranscriptionResult]) -> None:
        """Initialise with a list of pre-loaded responses.

        Args:
            responses: Responses returned in round-robin order. Must contain
                at least one item.
        """
        self._responses = responses
        self._call_count = 0

    def transcribe(
        self,
        audio_fp: Path,  # noqa: ARG002
        *,
        language: str | None = None,  # noqa: ARG002
        **kwargs: object,  # noqa: ARG002
    ) -> TranscriptionResult:
        """Return the next pre-loaded response without any I/O.

        Args:
            audio_fp: Ignored. Accepted to satisfy the BaseTranscriber protocol.
            language: Ignored.
            **kwargs: Ignored.

        Returns:
            The next TranscriptionResult from the response list (cycled).
        """
        result = self._responses[self._call_count % len(self._responses)]
        self._call_count += 1
        return result

    async def atranscribe(
        self,
        audio_fp: Path,
        *,
        language: str | None = None,
        **kwargs: object,
    ) -> TranscriptionResult:
        """Async variant - delegates directly to ``transcribe``.

        Args:
            audio_fp: Ignored.
            language: Ignored.
            **kwargs: Ignored.

        Returns:
            The next TranscriptionResult from the response list (cycled).
        """
        return self.transcribe(audio_fp, language=language, **kwargs)


class FakeTranscriberConfig(TranscriptionConfig):
    """TranscriptionConfig that creates a FakeTranscriber. No heavy deps.

    Suitable as a drop-in replacement for real transcription configs in unit
    tests, integration tests, and CI environments.

    Example:
        ::

            from pathlib import Path

            from llm_core.transcription.base import TranscriptionResult
            from llm_core.transcription.testing.fake import FakeTranscriberConfig

            config = FakeTranscriberConfig(
                responses=[TranscriptionResult(text="Hello world.", language="en")]
            )
            t = config.create_transcriber()
            assert t.transcribe(Path("dummy.mp3")).text == "Hello world."

    Attributes:
        responses: Pre-loaded TranscriptionResult objects cycled in order.
        model: Always "fake". Satisfies the abstract field requirement.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    responses: list[TranscriptionResult]
    model: str = "fake"

    def create_transcriber(self) -> FakeTranscriber:
        """Instantiate a FakeTranscriber from this config.

        Returns:
            A FakeTranscriber that cycles through ``responses`` in order.
        """
        return FakeTranscriber(responses=self._responses_list())

    def _responses_list(self) -> list[TranscriptionResult]:
        return list(self.responses)
