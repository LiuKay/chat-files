import os
from typing import Dict, Any

import chainlit as cl
import chromadb
from chromadb import Settings
from langchain.chains import RetrievalQA
from langchain.vectorstores import Chroma

from config import get_config
from embeddings import get_embeddings_from_config
from llm import get_llm_from_config


@cl.on_chat_start
def main():
    llm = get_llm_from_config()
    conf = get_config(os.getenv("CONFIG_FILE"))
    chroma_client = create_chroma_client(conf)
    vector_store = Chroma(
        collection_name="test",
        embedding_function=get_embeddings_from_config(),
        client=chroma_client,
        client_settings=chroma_client.get_settings(),
    )
    qa_chain = RetrievalQA.from_llm(
        llm=llm,
        retriever=vector_store.as_retriever(
            search_type="similarity_score_threshold",
        ),
        return_source_documents=True,
        verbose=True
    )
    cl.user_session.set("qa_chain", qa_chain)


@cl.on_message
async def main(message: str):
    qa_chain = cl.user_session.get("qa_chain")  # type: RetrievalQA
    res = await qa_chain.acall(message, callbacks=[cl.AsyncLangchainCallbackHandler()])
    await cl.Message(content=res["text"]).send()


def create_chroma_client(conf: Dict[str, Any]):
    chroma_settings = conf["chroma"]
    persist_dir = chroma_settings["persist_directory"]
    chroma_client = chromadb.PersistentClient(path=persist_dir, settings=Settings(**chroma_settings))
    return chroma_client
