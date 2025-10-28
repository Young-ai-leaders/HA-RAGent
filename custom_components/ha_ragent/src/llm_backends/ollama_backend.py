from base_backend import ALlmBaseBackend
from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate

class OllamaBackend(ALlmBaseBackend):
    def __init__(self, template: str, model: str = "qwen3:8b"):
        super.__init__(template, model)

    def send_request(self, query):
        promt = ChatPromptTemplate.from_template(self.template)
        chain = promt | self.model
        chain.invoke()
        return super().send_request(query)