# Transcription in `llm-core` - Design

---

## How it fits the existing pattern

`llm-core` has two precedents to follow:

- **`chat/`**: `ChatConfig` base → `create_chat_model()` → `BaseChatModel` (LangChain type).
  Works cleanly because LangChain's `init_chat_model` is a universal dispatcher across all
  chat providers.
- **`embeddings/`**: `EmbeddingsConfig` base → `create_embeddings()` → `Embeddings`
  (LangChain type). Same idea with `init_embeddings`.

Transcription cannot follow this exactly because there is no LangChain universal
transcription dispatcher. The OpenAI API Whisper endpoint has a LangChain wrapper, but
local `openai-whisper`, `faster-whisper`, and `whisperx` do not - and they are the
primary targets. The return type of `create_transcriber()` will therefore be
`BaseTranscriber`, a protocol/ABC we define and own, not a LangChain type.

In all other respects the pattern is identical: base config with `create_transcriber()`,
specific configs adding provider-specific fields, providers in a separate module,
optional dependencies per provider, a fake for testing.

---

## Module layout

```
llm_core/
└── transcription/
    ├── __init__.py
    ├── base.py               # BaseTranscriber protocol + TranscriptionResult
    ├── config/
    │   ├── __init__.py
    │   ├── base.py           # TranscriptionConfig base → create_transcriber()
    │   ├── whisper.py        # WhisperConfig (local openai-whisper)
    │   ├── faster_whisper.py # FasterWhisperConfig (local faster-whisper)
    │   └── openai_api.py     # OpenAIAPIConfig (remote Whisper API)
    ├── providers/
    │   ├── __init__.py
    │   ├── whisper.py        # WhisperTranscriber
    │   ├── faster_whisper.py # FasterWhisperTranscriber
    │   └── openai_api.py     # OpenAIAPITranscriber
    └── testing/
        └── fake.py           # FakeTranscriberConfig + FakeTranscriber
```

`testing/` mirrors `llm_core/testing/fake_chat_model.py`.

---

## `TranscriptionResult`

Returning `str` from `transcribe()` would discard information the providers already
compute for free. `openai-whisper` returns detected language, confidence, and
segment-level timestamps alongside the text. A structured result captures all of this
without forcing providers that don't have it to fabricate it.

```python
# transcription/base.py

from dataclasses import dataclass, field


@dataclass
class TranscriptionSegment:
    """One timed segment of a transcription.

    Not all providers populate this; treat as best-effort.
    """
    start: float          # seconds
    end: float            # seconds
    text: str


@dataclass
class TranscriptionResult:
    """Output of a transcription call.

    Attributes:
        text:      Full transcript as a single string. Always present.
        language:  BCP-47 language code detected or specified (e.g. "en", "it").
                   None if the provider did not detect or report it.
        segments:  Word- or sentence-level segments with timestamps.
                   Empty list if the provider does not support segmentation.
        provider:  Provider identifier string, for logging and traceability.
    """
    text: str
    language: str | None = None
    segments: list[TranscriptionSegment] = field(default_factory=list)
    provider: str = ""
```

This makes language detection in the downloader trivial: for local Whisper providers,
`result.language` is populated for free. No separate post-processing hook needed for
the common case.

---

## `BaseTranscriber` protocol

```python
# transcription/base.py (continued)

from pathlib import Path
from typing import Protocol, runtime_checkable


@runtime_checkable
class BaseTranscriber(Protocol):
    """Common interface for all transcription backends.

    Both sync and async surfaces are required. For sync-only backends
    (openai-whisper, faster-whisper), `atranscribe` wraps `transcribe`
    in `asyncio.to_thread` - this wrapping lives inside the provider, not
    in the caller.
    """

    provider_name: str

    def transcribe(
        self,
        audio_fp: Path,
        *,
        language: str | None = None,
        **kwargs,
    ) -> TranscriptionResult: ...

    async def atranscribe(
        self,
        audio_fp: Path,
        *,
        language: str | None = None,
        **kwargs,
    ) -> TranscriptionResult: ...
```

`language` is an explicit keyword argument (not buried in `**kwargs`) because it is the
single most common override - passing it through `kwargs` would make callers guess at
the spelling.

`runtime_checkable` allows `isinstance(obj, BaseTranscriber)` checks if needed for
registry validation. The protocol is structural, so providers don't need to inherit
from it - they just need to implement the interface.

---

## `TranscriptionConfig` base

Mirrors `ChatConfig` exactly. Fields common to all providers live here.
`create_transcriber()` is abstract at the base - unlike `ChatConfig` which can call
`init_chat_model` directly, there is no universal factory, so each subclass must
implement it.

```python
# transcription/config/base.py

from abc import abstractmethod
from pathlib import Path

from llm_core.data_models.basemodel_kwargs import BaseModelKwargs
from llm_core.transcription.base import BaseTranscriber


class TranscriptionConfig(BaseModelKwargs):
    """Base config for a transcription model.

    Attributes:
        model:     Model identifier. Meaning is provider-specific
                   (e.g. "medium" for Whisper, "Systran/faster-whisper-medium"
                   for faster-whisper, "whisper-1" for the OpenAI API).
        language:  Default language hint (BCP-47, e.g. "en", "it").
                   None means auto-detect (where supported).
    """

    model: str
    language: str | None = None

    @abstractmethod
    def create_transcriber(self) -> BaseTranscriber:
        """Instantiate the transcriber from this config.

        Returns:
            A BaseTranscriber implementation ready to accept audio files.
        """
```

`TranscriptionConfig` extends `BaseModelKwargs`, so `to_kw()` works for forwarding
config fields to provider constructors - same pattern as chat and embeddings.

---

## Provider configs

### `WhisperConfig`

```python
# transcription/config/whisper.py

from llm_core.transcription.config.base import TranscriptionConfig
from llm_core.transcription.base import BaseTranscriber


class WhisperConfig(TranscriptionConfig):
    """Config for local openai-whisper.

    Attributes:
        model:    Whisper model size. Defaults to "medium".
                  One of: tiny, base, small, medium, large, large-v2, large-v3.
        device:   Torch device. "cpu" or "cuda". Defaults to "cpu".
        fp16:     Use half-precision. Meaningful only on CUDA. Defaults to False.
    """

    model: str = "medium"
    device: str = "cpu"
    fp16: bool = False

    def create_transcriber(self) -> BaseTranscriber:
        from llm_core.transcription.providers.whisper import WhisperTranscriber
        return WhisperTranscriber(config=self)
```

### `FasterWhisperConfig`

```python
# transcription/config/faster_whisper.py

from typing import Literal
from llm_core.transcription.config.base import TranscriptionConfig
from llm_core.transcription.base import BaseTranscriber


class FasterWhisperConfig(TranscriptionConfig):
    """Config for faster-whisper (CTranslate2 backend).

    Substantially faster inference than openai-whisper on the same model size.
    Same model weight naming convention.

    Attributes:
        model:       Model size or HuggingFace model ID.
                     Defaults to "medium".
        device:      "cpu" or "cuda". Defaults to "cpu".
        compute_type: CTranslate2 quantization. "int8" is fastest on CPU,
                      "float16" is fastest on GPU. Defaults to "int8".
        beam_size:   Beam search width. Higher = more accurate, slower.
                     Defaults to 5 (faster-whisper's default).
    """

    model: str = "medium"
    device: str = "cpu"
    compute_type: str = "int8"
    beam_size: int = 5

    def create_transcriber(self) -> BaseTranscriber:
        from llm_core.transcription.providers.faster_whisper import FasterWhisperTranscriber
        return FasterWhisperTranscriber(config=self)
```

### `OpenAIAPIConfig`

```python
# transcription/config/openai_api.py

from langchain_core.utils.utils import secret_from_env
from pydantic import Field, SecretStr

from llm_core.transcription.config.base import TranscriptionConfig
from llm_core.transcription.base import BaseTranscriber


class OpenAIAPIConfig(TranscriptionConfig):
    """Config for the OpenAI Whisper API (remote, no local model).

    Useful for lighter deployments where torch/GPU are not available.

    Attributes:
        model:       Always "whisper-1" for the current OpenAI API.
        api_key:     OpenAI API key. Reads OPENAI_API_KEY env var by default.
        response_format: "text" | "json" | "verbose_json" | "srt" | "vtt".
                     "verbose_json" populates segments in TranscriptionResult.
                     Defaults to "verbose_json".
    """

    model: str = "whisper-1"
    api_key: SecretStr | None = Field(
        default_factory=secret_from_env("OPENAI_API_KEY", default=None)
    )
    response_format: str = "verbose_json"

    def create_transcriber(self) -> BaseTranscriber:
        from llm_core.transcription.providers.openai_api import OpenAIAPITranscriber
        return OpenAIAPITranscriber(config=self)
```

---

## Providers

### `WhisperTranscriber`

```python
# transcription/providers/whisper.py

import asyncio
from pathlib import Path

from loguru import logger as lg

from llm_core.transcription.base import BaseTranscriber, TranscriptionResult, TranscriptionSegment

# TYPE_CHECKING guard - whisper is imported lazily so the module can be
# imported even when openai-whisper is not installed, failing only at
# create_transcriber() time.
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from llm_core.transcription.config.whisper import WhisperConfig


class WhisperTranscriber:
    """Local transcription via openai-whisper."""

    provider_name: str = "whisper"

    def __init__(self, config: "WhisperConfig") -> None:
        import whisper  # deferred import; raises ImportError if not installed
        lg.info(f"Loading Whisper model '{config.model}' on {config.device}...")
        self._config = config
        self._model = whisper.load_model(config.model, device=config.device)
        lg.info("Whisper model loaded.")

    def transcribe(
        self,
        audio_fp: Path,
        *,
        language: str | None = None,
        **kwargs,
    ) -> TranscriptionResult:
        effective_language = language or self._config.language
        result = self._model.transcribe(
            str(audio_fp),
            language=effective_language,
            fp16=self._config.fp16,
            **kwargs,
        )
        segments = [
            TranscriptionSegment(start=s["start"], end=s["end"], text=s["text"])
            for s in result.get("segments", [])
        ]
        return TranscriptionResult(
            text=result["text"],
            language=result.get("language"),
            segments=segments,
            provider=self.provider_name,
        )

    async def atranscribe(
        self,
        audio_fp: Path,
        *,
        language: str | None = None,
        **kwargs,
    ) -> TranscriptionResult:
        return await asyncio.to_thread(self.transcribe, audio_fp, language=language, **kwargs)
```

`asyncio.to_thread` lives here, inside the provider - not exposed to callers.
The model is loaded eagerly in `__init__` so the first `transcribe()` call is fast.

### `FasterWhisperTranscriber`

Structurally identical to `WhisperTranscriber`. Key differences:

- `faster_whisper.WhisperModel(model, device, compute_type)` constructor
- Result iteration: `model.transcribe()` returns `(segments_generator, info)` not a dict
- `info.language` and `info.language_probability` replace `result["language"]`
- Segments are `Segment` objects with `.start`, `.end`, `.text` directly

```python
    def transcribe(self, audio_fp: Path, *, language: str | None = None, **kwargs) -> TranscriptionResult:
        segments_gen, info = self._model.transcribe(
            str(audio_fp),
            language=language or self._config.language,
            beam_size=self._config.beam_size,
            **kwargs,
        )
        segments = [TranscriptionSegment(s.start, s.end, s.text) for s in segments_gen]
        full_text = " ".join(s.text.strip() for s in segments)
        return TranscriptionResult(
            text=full_text,
            language=info.language,
            segments=segments,
            provider=self.provider_name,
        )
```

Note the generator consumption: `faster-whisper` streams segments lazily. Consuming
the generator to build `segments` also materialises `full_text` - no double pass needed.

### `OpenAIAPITranscriber`

Uses `httpx` (already in `fastapi-tools` deps, likely available). Async-native via
`httpx.AsyncClient`; the sync wrapper calls `asyncio.run()` - or better, a dedicated
sync client.

```python
    def transcribe(self, audio_fp: Path, *, language: str | None = None, **kwargs) -> TranscriptionResult:
        import httpx
        with open(audio_fp, "rb") as f:
            with httpx.Client() as client:
                response = client.post(
                    "https://api.openai.com/v1/audio/transcriptions",
                    headers={"Authorization": f"Bearer {self._config.api_key.get_secret_value()}"},
                    data={"model": self._config.model, "response_format": self._config.response_format,
                          **({"language": language or self._config.language} if (language or self._config.language) else {})},
                    files={"file": (audio_fp.name, f, "audio/mpeg")},
                )
        response.raise_for_status()
        data = response.json()
        # verbose_json includes segments; plain text does not
        segments = [
            TranscriptionSegment(s["start"], s["end"], s["text"])
            for s in data.get("segments", [])
        ]
        return TranscriptionResult(
            text=data.get("text", data) if isinstance(data, dict) else data,
            language=data.get("language"),
            segments=segments,
            provider=self.provider_name,
        )

    async def atranscribe(self, audio_fp: Path, *, language: str | None = None, **kwargs) -> TranscriptionResult:
        import httpx
        async with httpx.AsyncClient() as client:
            # same logic, async version
            ...
```

---

## Testing

```python
# transcription/testing/fake.py

from pathlib import Path
from llm_core.transcription.base import BaseTranscriber, TranscriptionResult
from llm_core.transcription.config.base import TranscriptionConfig


class FakeTranscriber:
    """Deterministic transcriber for unit tests. No model loaded, no I/O.

    Cycles through responses in round-robin order, mirroring FakeChatModel.
    """

    provider_name: str = "fake"

    def __init__(self, responses: list[TranscriptionResult]) -> None:
        self._responses = responses
        self._call_count = 0

    def transcribe(self, audio_fp: Path, *, language: str | None = None, **kwargs) -> TranscriptionResult:
        result = self._responses[self._call_count % len(self._responses)]
        self._call_count += 1
        return result

    async def atranscribe(self, audio_fp: Path, *, language: str | None = None, **kwargs) -> TranscriptionResult:
        return self.transcribe(audio_fp, language=language, **kwargs)


class FakeTranscriberConfig(TranscriptionConfig):
    """TranscriptionConfig that creates a FakeTranscriber. No heavy deps.

    Example::

        config = FakeTranscriberConfig(
            responses=[TranscriptionResult(text="Boil the pasta.", language="en")]
        )
        transcriber = config.create_transcriber()
        result = transcriber.transcribe(Path("any.mp3"))
        assert result.text == "Boil the pasta."
    """

    responses: list[TranscriptionResult]
    model: str = "fake"

    def create_transcriber(self) -> FakeTranscriber:
        return FakeTranscriber(responses=self.responses)
```

---

## Optional dependencies

```toml
[project.optional-dependencies]
whisper        = ["openai-whisper>=20240930", "torch>=2.0"]
faster-whisper = ["faster-whisper>=1.0"]
openai         = ["openai>=1.0", "httpx>=0.26"]   # openai SDK or httpx directly
# existing:
# ollama, azure, huggingface, chroma ...
all = ["llm-core[openai,azure,ollama,huggingface,chroma,whisper,faster-whisper]"]
```

`torch` is pulled in by `whisper`. `faster-whisper` uses CTranslate2 instead - lighter,
no `torch` dependency - so it gets its own optional group.

---

## Open points to decide

**1. Model loading eagerness**

`WhisperTranscriber.__init__` loads the model immediately (current proposal).
This is correct for the service pattern - pay the cost once at startup, all calls
are fast. But it makes `FakeTranscriberConfig` import slightly awkward if someone
instantiates a real config in a test context.
Alternative: lazy loading on first `transcribe()` call, guarded by a lock.
Recommendation: keep eager loading, document it clearly. Tests use `FakeTranscriberConfig`.

**2. `language` at config vs call time**

Both `TranscriptionConfig.language` (default for all calls) and `language` parameter
on `transcribe()` (per-call override) exist. The current proposal: call-time `language`
takes precedence over config-level default, with `None` meaning "use config default,
which may also be None (auto-detect)". This should be documented explicitly and tested.

**3. Where `TranscriptionResult` lives relative to the downloader**

The downloader post-processing hook (`TranscriptionHook`) returns a `TranscriptionResult`
and stores `result.text` in `DownloadedMedia.transcript` and `result.language` in
`DownloadedMedia.language`. This means the downloader depends on `llm-core` for the
result type - acceptable given it already depends on `llm-core[whisper]` for the
transcriber itself.

**4. Segment granularity**

`TranscriptionSegment` currently captures sentence-level segments (Whisper's default).
Word-level timestamps require `word_timestamps=True` in `openai-whisper` and a
different segment structure. This is a `WhisperX` / advanced use case - defer for now,
but `TranscriptionSegment` should not be locked down too tightly. A `words` field
(`list[dict] | None = None`) could be added later without breaking the base.

**5. `FasterWhisperTranscriber` full_text construction**

The generator-based segment consumption means `full_text` is built by joining segment
texts. This matches how `faster-whisper` expects to be used, but the join separator
(space vs nothing) and leading/trailing whitespace per segment varies by model. Worth
an integration test against a real audio file to validate before shipping.

**6. Audio format assumptions**

All three providers accept a file path, not bytes. The providers trust the caller to
provide a valid audio file. Format conversion (e.g. extracting audio from an mp4
before transcribing) is the caller's responsibility - the downloader service would
handle this before calling `transcribe()`. This boundary should be documented.
