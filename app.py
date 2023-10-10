import chainlit as cl
from langchain import PromptTemplate, LLMChain

from llm import get_llm_from_config

template = """Question: {question}

Answer: Let's think step by step."""


@cl.on_chat_start
def main():
    prompt = PromptTemplate(template=template, input_variables=["question"])
    llm = get_llm_from_config()
    llm_chain = LLMChain(prompt=prompt, llm=llm, verbose=True)
    cl.user_session.set("llm_chain", llm_chain)


@cl.on_message
async def main(message: str):
    llm_chain = cl.user_session.get("llm_chain")  # type: LLMChain
    res = await llm_chain.acall(message, callbacks=[cl.AsyncLangchainCallbackHandler()])
    await cl.Message(content=res["text"]).send()
