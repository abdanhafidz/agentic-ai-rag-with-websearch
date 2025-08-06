from rag.pipeline.language_model import LM
from rag.inference.inferencer import Inferencer
from abc import ABC, abstractmethod
class Agent(ABC):
    def __init__(self, inferencer:Inferencer, prompt_template = [
        {
            "role" : "system",
            "content":"You are an agent that doing some specic task"
        }
    ]):
        self.inferencer = inferencer
        self.inferencer.model.prompt_template = prompt_template
        self.prompt = prompt_template
    @abstractmethod
    async def get_result(self):
        pass
