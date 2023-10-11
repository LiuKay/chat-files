# flake8: noqa
from typing import List

from langchain.chains.prompt_selector import ConditionalPromptSelector, is_chat_model
from langchain.chains.question_answering.stuff_prompt import PROMPT_SELECTOR
from langchain.prompts import PromptTemplate
from pydantic import BaseModel


class Llama2Prompt(BaseModel):
    system_msg: str
    user_msg: str
    input_variables: List[str]

    def get_prompt_template(self) -> PromptTemplate:
        msg_list = [
            {
                "role": "system",
                "content": self.system_msg
            },
            {
                "role": "user",
                "content": self.user_msg
            }
        ]
        llama2_msg_str = build_llama2_prompt(msg_list)
        return PromptTemplate(
            template=llama2_msg_str, input_variables=self.input_variables
        )


def build_llama2_prompt(prompt_messages):
    start_prompt = "<s>[INST] "
    end_prompt = " [/INST]"
    conversation = []
    for index, message in enumerate(prompt_messages):
        if message["role"] == "system" and index == 0:
            conversation.append(f"<<SYS>>\n{message['content']}\n<</SYS>>\n\n")
        elif message["role"] == "user":
            conversation.append(message["content"].strip())
        else:
            conversation.append(f" [/INST] {message['content'].strip()}</s><s>[INST] ")

    return start_prompt + "".join(conversation) + end_prompt


system_template = """You are a helpful, respectful and honest assistant. Use the following pieces of context to answer the users question.

Your answer should be about 100 to 500 words. If you don't know the answer, just say that you don't know, don't try to make up an answer.
----------------
{context}."""

user_template = """
{question}
"""

LLAMA2_CHAT_PROMPT = Llama2Prompt(
    system_msg=system_template,
    user_msg=user_template,
    input_variables=["context", "question"]
).get_prompt_template()

LLAMA2_PROMPT_SELECTOR = ConditionalPromptSelector(
    default_prompt=LLAMA2_CHAT_PROMPT,
    conditionals=[(is_chat_model, LLAMA2_CHAT_PROMPT)]
)

QA_PROMPT_MAP = {
    "llama2": LLAMA2_PROMPT_SELECTOR,
    "normal": PROMPT_SELECTOR
}

if __name__ == "__main__":
    print(LLAMA2_CHAT_PROMPT.format(context="this is a context", question="this is a user question"))
