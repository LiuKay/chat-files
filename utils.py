from typing import Any, Dict

from deepmerge import always_merger
from langchain.llms.base import LLM


def merge(a: Dict[Any, Any], b: Dict[Any, Any]) -> Dict[Any, Any]:
    c = {}
    always_merger.merge(c, a)
    always_merger.merge(c, b)
    return c


def add_chainlit_llm_provider(llm: LLM, name: str = "Llama2"):
    from chainlit.playground import add_llm_provider
    from chainlit.playground.providers.langchain import LangchainGenericProvider
    add_llm_provider(
        LangchainGenericProvider(
            # It is important that the id of the provider matches the _llm_type
            id=llm._llm_type,
            # The name is not important. It will be displayed in the UI.
            name=name,
            # This should always be a Langchain llm instance (correctly configured)
            llm=llm,
            # If the LLM works with messages, set this to True
            is_chat=False
        )
    )
