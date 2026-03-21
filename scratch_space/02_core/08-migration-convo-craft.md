# Block 8 - Migration guide: convo-craft

Parent: [08-release-migration.md](08-release-migration.md)

convo-craft is a Streamlit app with multiple stateless LLM tasks
(`ConversationGenerator`, `ParagraphSplitter`, `Translator`, `TopicsPicker`).
Each task currently defines its own chat model setup and `with_structured_output`
boilerplate inline. The migration replaces this repeated pattern with
`StructuredLLMChain`.

---

## Scope

### Files to update

| File | Change |
|---|---|
| `pyproject.toml` | Replace `langchain-openai` dep with `llm-core[openai]` |
| `src/convo_craft/config/chat_openai.py` | Delete - replaced by `llm_core.chat.ChatOpenAIConfig` |
| `src/convo_craft/llm/conversation_generator.py` | Replace inline chain with `StructuredLLMChain` |
| `src/convo_craft/llm/paragraph_splitter.py` | Replace inline chain with `StructuredLLMChain` |
| `src/convo_craft/llm/translator.py` | Replace inline chain with `StructuredLLMChain` |
| `src/convo_craft/llm/topic_picker.py` | Replace inline chain with `StructuredLLMChain` |

---

## 1. Update `pyproject.toml`

```toml
# uv (pyproject.toml [project])
[project]
dependencies = [
    "llm-core[openai] @ git+https://github.com/pitrified/llm-core@v0.1.0",
    "streamlit>=1.38.0",
    "loguru>=0.7.2",
]

# Poetry (alternative, if staying on Poetry)
[tool.poetry.dependencies]
llm-core = {git = "https://github.com/pitrified/llm-core", tag = "v0.1.0", extras = ["openai"]}
```

---

## 2. Delete `config/chat_openai.py`

The custom `ChatOpenAIConfig(BaseModel)` that passes through `model_dump()` is
replaced by `ChatOpenAIConfig` from llm-core. The key difference is that the new
version extends `BaseModelKwargs` and provides `create_chat_model()` - no need
to call `ChatOpenAI(**config.model_dump())` manually.

---

## 3. Migrate each task class

The pattern is the same for all four task classes. Here is `ParagraphSplitter`
as a representative example:

```python
# Before - src/convo_craft/llm/paragraph_splitter.py
from dataclasses import dataclass
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from convo_craft.config.chat_openai import ChatOpenAIConfig

class ParagraphSplitterResult(BaseModel):
    portions: list[str] = Field(description="The split paragraph.")

split_paragraph_template = "Split the following paragraph into portions:\n\n{paragraph}"
split_paragraph_prompt = ChatPromptTemplate(
    [HumanMessagePromptTemplate.from_template(split_paragraph_template)]
)

@dataclass
class ParagraphSplitter:
    chat_openai_config: ChatOpenAIConfig

    def __post_init__(self):
        self.model = ChatOpenAI(**self.chat_openai_config.model_dump())
        self.structured_llm = self.model.with_structured_output(ParagraphSplitterResult)

    def invoke(self, paragraph: str) -> ParagraphSplitterResult:
        value = split_paragraph_prompt.invoke({"paragraph": paragraph})
        output = self.structured_llm.invoke(value)
        if not isinstance(output, ParagraphSplitterResult):
            raise ValueError(f"Unexpected output type: {type(output)}")
        return output
```

```python
# After - src/convo_craft/llm/paragraph_splitter.py
from llm_core.chains import StructuredLLMChain
from llm_core.chat import ChatOpenAIConfig
from llm_core.data_models import BaseModelKwargs
from pydantic import BaseModel, Field

class ParagraphSplitterInput(BaseModelKwargs):
    paragraph: str  # matches {{ paragraph }} in prompt

class ParagraphSplitterResult(BaseModel):
    portions: list[str] = Field(description="The split paragraph.")

_PROMPT = "Split the following paragraph into portions:\n\n{{ paragraph }}"

class ParagraphSplitter:
    def __init__(self, chat_config: ChatOpenAIConfig | None = None) -> None:
        self._chain = StructuredLLMChain(
            chat_config=chat_config or ChatOpenAIConfig(),
            prompt_str=_PROMPT,
            input_model=ParagraphSplitterInput,
            output_model=ParagraphSplitterResult,
        )

    def invoke(self, paragraph: str) -> ParagraphSplitterResult:
        return self._chain.invoke(ParagraphSplitterInput(paragraph=paragraph))
```

### `ConversationGenerator` - multi-variable prompt

`ConversationGenerator` uses a two-part prompt with multiple variables. Merge
the two `HumanMessagePromptTemplate` parts into a single Jinja2 string:

```python
# After - input model captures all variables
class ConversationGeneratorInput(BaseModelKwargs):
    language: str
    topic: str
    understanding_level: str
    num_messages: int
    num_sentences: int
    topic_sample: str
    conversation_sample: str

_PROMPT = """\
Write a conversation in {{ language }} between two persons, \
that should be used to teach the user the language.
The conversation should be about the following topic: "{{ topic }}".
Assume that the user has an {{ understanding_level }} level of understanding of the language.
The conversation should last about {{ num_messages }} messages in total, \
with each message being about {{ num_sentences }} sentences long.

This is an example of a conversation in {{ language }} between two persons, \
about the topic "{{ topic_sample }}", of the appropriate difficulty level for the user, \
which is {{ understanding_level }}:
{{ conversation_sample }}
"""

class ConversationGenerator:
    def __init__(self, chat_config: ChatOpenAIConfig | None = None) -> None:
        self._chain = StructuredLLMChain(
            chat_config=chat_config or ChatOpenAIConfig(),
            prompt_str=_PROMPT,
            input_model=ConversationGeneratorInput,
            output_model=ConversationGeneratorResult,
        )

    def invoke(self, topic: str, language: str, ...) -> ConversationGeneratorResult:
        return self._chain.invoke(ConversationGeneratorInput(
            language=language,
            topic=topic,
            ...
        ))
```

---

## 4. Prompt format note

LangChain `ChatPromptTemplate.from_template` uses `{variable}` syntax.
`StructuredLLMChain` uses Jinja2 `{{ variable }}`. Update all prompt strings.

---

## 5. Verify

```bash
cd /home/pmn/repos/convo_craft
uv run pytest
```
