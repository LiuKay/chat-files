# Chat-Files

**Chat-Files** is an AI chatbot built with [Chainlit](https://chainlit.io) and [LangChain](https://python.langchain.com/en/latest/index.html).
This app allows you ask questions about your local documents using the power of LLMs.


- support online models: OpenAI, HuggingFace
- support local models, ctransformers models, llama2 models
- support any type of files, markdown, pdf, txt, doc, html...


## Getting Started

Install the necessary libraries
```bash
pip install -r requirements.txt
```

Create a .env file and input your OpenAI API Key or HuggingFace Hub API Key in the file
or if you use local model, update the config file to your local model path.

## Run locally

```bash
chainlit run app.py -w
```