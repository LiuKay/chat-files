import logging
import os
from pathlib import Path
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
        callback: Optional[Callable[[str], None]] = None,
        temperature=None,
):
    conf = get_config(os.getenv("CONFIG_FILE"))
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
        elif config["llm"] == "gptq":
            logger.info(f"Use gptq...local_files_only={local_files_only}")
            llm = get_gptq_llm(config)
        else:
            logger.info(f"Use HuggingFace local model...local_files_only={local_files_only}")
            config = {**config["huggingface"]}
            config["model_id"] = config.pop("model")
            config = merge(config, {"model_kwargs": {"local_files_only": local_files_only}})
            llm = HuggingFacePipeline.from_model_id(task="text-generation", **config)
    return llm


def get_gptq_llm(config: Dict[str, Any]) -> LLM:
    try:
        from auto_gptq import AutoGPTQForCausalLM
    except ImportError:
        raise ImportError(
            "Could not import `auto_gptq` package. "
            "Please install it with `pip install chatdocs[gptq]`"
        )

    from transformers import (
        AutoTokenizer,
        TextGenerationPipeline,
        MODEL_FOR_CAUSAL_LM_MAPPING,
    )

    local_files_only = not config["download"]
    config = {**config["gptq"]}
    model_name_or_path = config.pop("model")
    model_file = config.pop("model_file", None)
    pipeline_kwargs = config.pop("pipeline_kwargs", None) or {}

    model_basename = None
    use_safetensors = False
    if model_file:
        model_basename = Path(model_file).stem
        use_safetensors = model_file.endswith(".safetensors")

    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        local_files_only=local_files_only,
    )
    model = AutoGPTQForCausalLM.from_quantized(
        model_name_or_path,
        model_basename=model_basename,
        use_safetensors=use_safetensors,
        local_files_only=local_files_only,
        **config,
    )
    MODEL_FOR_CAUSAL_LM_MAPPING.register("dragonflychatbot-gptq", model.__class__)
    pipeline = TextGenerationPipeline(
        task="text-generation",
        model=model,
        tokenizer=tokenizer,
        **pipeline_kwargs,
    )
    return HuggingFacePipeline(pipeline=pipeline)
