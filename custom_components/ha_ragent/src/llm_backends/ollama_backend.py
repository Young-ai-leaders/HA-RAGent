import logging
from .base_backend import ALlmBaseBackend
from langchain_core.prompts import ChatPromptTemplate

_logger = logging.getLogger(__name__)

class OllamaBackend(ALlmBaseBackend):
    def __init__(self, template: str, model: str = "qwen3:0.6b"):
        super.__init__(template, model)

    def send_request(self, query):
        promt = ChatPromptTemplate.from_template(self.template)
        chain = promt | self.model
        chain.invoke()
        return super().send_request(query)