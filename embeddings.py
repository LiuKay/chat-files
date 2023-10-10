import logging
import os
from typing import Dict, Any

from langchain.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings, HuggingFaceEmbeddings

from config import get_config

logger = logging.getLogger(__name__)


def get_embeddings_from_config():
    conf = get_config(os.getenv("CONFIG_FILE"))
    return get_embeddings(conf)


def get_embeddings(config: Dict[str, Any]):
    config = {**config["embeddings"]}
    config["model_name"] = config.pop("model")
    if config["model_name"] == "openai":
        logger.info("Loading OpenAI embeddings model...")
        config["model"] = config.pop("model_name")
        embeddings = OpenAIEmbeddings(**config)
    elif config["model_name"].startswith("hkunlp/"):
        logger.info("Loading HuggingFace embeddings model...")
        embeddings = HuggingFaceInstructEmbeddings(**config)
    else:
        embeddings = HuggingFaceEmbeddings(**config)
    return embeddings
