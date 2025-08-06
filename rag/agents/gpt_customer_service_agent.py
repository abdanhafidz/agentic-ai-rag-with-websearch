from rag.agents.agents import Agent
from rag.pipeline.language_model import LM
from rag.inference.inferencer import Inferencer

class GPTCSAgent(Agent):
    def __init__(self, inferencer : Inferencer , prompt_template):
        super().__init__(inferencer, prompt_template)
        self.inferencer = inferencer
        self.prompt_template = prompt_template
    async def get_result(self, question : str):
        self.inferencer.model.prompt_template = self.prompt_template
        print("Question received :", question)
        return await self.inferencer.infer(query = question)
