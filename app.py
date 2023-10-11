import glob
import logging
import os
from typing import Dict, Any, Optional, List

import chainlit as cl
import chromadb
from chainlit import Message, Text
from chromadb import Embeddings
from langchain.chains import RetrievalQA
from langchain.document_loaders import UnstructuredWordDocumentLoader, CSVLoader, EverNoteLoader, \
    UnstructuredEPubLoader, UnstructuredHTMLLoader, UnstructuredMarkdownLoader, UnstructuredODTLoader, PDFMinerLoader, \
    UnstructuredPowerPointLoader, TextLoader
from langchain.schema import Document
from langchain.text_splitter import NLTKTextSplitter
from langchain.vectorstores import Chroma

from config import get_config
from embeddings import get_embeddings_from_config
from llm import get_llm_from_config

logger = logging.getLogger(__name__)


@cl.on_chat_start
def main():
    conf = get_config(os.getenv("CONFIG_FILE"))
    llm = get_llm_from_config(conf)
    embeddings = get_embeddings_from_config(conf)
    vector_store = init_vector_store(conf, embeddings)
    qa_chain = RetrievalQA.from_llm(
        llm=llm,
        retriever=vector_store.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={
                'k': 2,
                'fetch_k': 10,
                'score_threshold': 0.5}
        ),
        return_source_documents=True,
        verbose=True
    )
    cl.user_session.set("qa_chain", qa_chain)


@cl.on_message
async def main(message: str):
    qa_chain = cl.user_session.get("qa_chain")  # type: RetrievalQA
    res = await qa_chain.acall(message, callbacks=[cl.AsyncLangchainCallbackHandler()])
    logger.info(f"response is {res}")

    answer = res["result"]
    source_docs = res.get("source_documents")
    res_msg = cl.Message(content=answer)
    add_source_docs(res_msg, source_docs)
    await res_msg.send()


def init_vector_store(conf: Dict[str, Any], embedding: Optional[Embeddings] = None):
    knowledge_base_dir = conf["knowledge_base_dir"]
    logger.info(f"loading docs from {knowledge_base_dir}...")
    docs = load_documents(knowledge_base_dir)
    splitter_config = conf["splitter"]
    text_splitter = NLTKTextSplitter(**splitter_config)
    documents = text_splitter.split_documents(docs)
    chroma_settings = conf["chroma"]
    db = Chroma.from_documents(
        documents,
        embedding,
        persist_directory=conf["persist_dir"],
        client_settings=chromadb.config.Settings(**chroma_settings)
    )
    return db


def load_documents(source_dir: str):
    # Loads all documents from source documents directory
    all_files = []
    for ext in LOADER_MAPPING:
        all_files.extend(
            glob.glob(os.path.join(source_dir, f"**/*{ext}"), recursive=True)
        )
    return [load_single_document(file_path) for file_path in all_files]


def load_single_document(file_path: str):
    logger.info("Loading document %s...", file_path)
    ext = "." + file_path.rsplit(".", 1)[-1]
    if ext in LOADER_MAPPING:
        loader_class, loader_args = LOADER_MAPPING[ext]
        loader = loader_class(file_path, **loader_args)
        return loader.load()[0]

    raise ValueError(f"Unsupported file extension '{ext}'")


# Map file extensions to document loaders and their arguments
LOADER_MAPPING = {
    ".csv": (CSVLoader, {}),
    # ".docx": (Docx2txtLoader, {}),
    ".doc": (UnstructuredWordDocumentLoader, {}),
    ".docx": (UnstructuredWordDocumentLoader, {}),
    ".enex": (EverNoteLoader, {}),
    ".epub": (UnstructuredEPubLoader, {}),
    ".html": (UnstructuredHTMLLoader, {}),
    ".md": (UnstructuredMarkdownLoader, {}),
    ".odt": (UnstructuredODTLoader, {}),
    ".pdf": (PDFMinerLoader, {}),
    ".ppt": (UnstructuredPowerPointLoader, {}),
    ".pptx": (UnstructuredPowerPointLoader, {}),
    ".txt": (TextLoader, {"encoding": "utf8"}),
    # Add more mappings for other file extensions and loaders as needed
}


def add_source_docs(message: Message, source_docs: Optional[List[Document]]):
    if source_docs:
        message.elements = [
            Text(name=f"{i + 1}. {doc.metadata.get('source')}", content=doc.page_content)
            for i, doc in enumerate(source_docs)
        ]
        source_refs = "\n".join(
            [f"{i + 1}. {doc.metadata.get('source')}" for i, doc in enumerate(source_docs)]
        )
        message.content += f"\n\n**_Source Documents_**: \n{source_refs}"
