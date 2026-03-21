"""llm-core - reusable LLM tooling library.

Provides multi-provider chat/embeddings configs, a structured-output chain,
versioned Jinja2 prompts, and vector store abstractions on top of LangChain.

Quickstart::

    from llm_core.chains import StructuredLLMChain
    from llm_core.chat import ChatOpenAIConfig
    from llm_core.data_models import BaseModelKwargs
    from pydantic import BaseModel

    class MyInput(BaseModelKwargs):
        text: str

    class MyOutput(BaseModel):
        summary: str

    chain = StructuredLLMChain(
        chat_config=ChatOpenAIConfig(),
        prompt_str="Summarise: {{ text }}",
        input_model=MyInput,
        output_model=MyOutput,
    )
    result = chain.invoke(MyInput(text="Long article..."))
"""

from llm_core.params.load_env import load_env

load_env()
