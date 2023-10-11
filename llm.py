import logging
import os
from typing import Dict, Any, Optional, Callable

from langchain import HuggingFaceHub, HuggingFacePipeline
from langchain.callbacks.base import BaseCallbackHandler
from langchain.chat_models import ChatOpenAI
from langchain.llms import CTransformers
from langchain.llms.base import LLM

from config import get_config
from utils import merge

logger = logging.getLogger(__name__)


def get_llm_from_config(
        config: Dict[str, Any],
        callback: Optional[Callable[[str], None]] = None,
        temperature=None,
):
    conf = config or get_config(os.getenv("CONFIG_FILE"))
    return get_llm(conf, callback=callback, temperature=temperature)


def get_llm(config: Dict[str, Any],
            *,
            callback: Optional[Callable[[str], None]] = None,
            temperature=None,
            ) -> LLM:
    if config["llm"] == "openai":
        config = {**config["openai"]}
        model_name = config["model"]
        temperature = temperature or config["temperature"]
        max_tokens = config["max_tokens"]
        llm = ChatOpenAI(model_name=model_name, temperature=temperature, max_tokens=max_tokens)
    else:
        model_type = config.pop("model_type")
        local_files_only = model_type == "local"
        if model_type == "online":
            logger.info("Use HuggingFace online model...")
            config = {**config["huggingface"]}
            config["model_id"] = config.pop("model")
            llm = HuggingFaceHub(repo_id=config["model_id"], model_kwargs=config["model_kwargs"])
        elif config["llm"] == "ctransformers":
            logger.info(f"Use ctransformers...local_files_only={local_files_only}")

            class CallbackHandler(BaseCallbackHandler):
                def on_llm_new_token(self, token: str, **kwargs) -> None:
                    callback(token)

            callbacks = [CallbackHandler()] if callback else None
            config = {**config["ctransformers"]}
            config = merge(config, {"config": {"local_files_only": local_files_only}})
            if temperature:
                config = merge(config, {"config": {"temperature": temperature}})
            llm = CTransformers(callbacks=callbacks, **config)
        else:
            logger.info(f"Use HuggingFace local model...local_files_only={local_files_only}")
            config = {**config["huggingface"]}
            config["model_id"] = config.pop("model")
            config = merge(config, {"model_kwargs": {"local_files_only": local_files_only}})
            llm = HuggingFacePipeline.from_model_id(task="text-generation", **config)
    return llm