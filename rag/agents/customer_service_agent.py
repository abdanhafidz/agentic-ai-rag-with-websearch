from rag.agents.agents import Agent
from rag.inference.inferencer import Inferencer

class CSAgent(Agent):
    def __init__(self, inferencer : Inferencer , prompt_template):
        super().__init__(inferencer, prompt_template)
        self.inferencer = inferencer
        self.prompt_template = prompt_template
        self.file_paths = [
            "../documents/bpjs.pdf",
            "../documents/pph21.pdf",
            "../documents/lembur.pdf",
            "../documents/uu13.pdf",
            "../documents/file.pdf",
        ]
    async def load_documents(self):
        for file_path in self.file_paths:
            await self.add_doc(file_path)
        
    async def add_doc(self, file_path):
        result = await self.inferencer.retriever.add_document_from_file(file_path)
        if result.success:
                print(f"Successfully processed: {result.document_metadata.file_name}")
                print(f"Chunks created: {result.document_metadata.chunk_count}")
        else:
                print(f"Failed to process: {result.error_message}")

    async def get_result(self, question):
        self.inferencer.model.prompt_template = self.prompt_template
        async for item in self.inferencer.infer_stream(query = question,
                                    enable_reranking=False,
                                    k=3):
                yield item
