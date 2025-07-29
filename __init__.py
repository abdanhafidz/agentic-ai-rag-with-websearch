from pipeline.qwen_llm import QwenLLM, QwenConfig
from retriever.langchain_retriever import LangChainRetriever
from inference.inferencer import Inferencer, InferencerConfig

config = QwenConfig(
        temperature=0.3,
        max_length=512,
        generation_timeout=30,
        repetition_penalty=1.1,
        max_workers = 2
    )
    
llm = QwenLLM(
        config = config
)

inferencer_config = InferencerConfig(
        default_k=5,
        enable_reranking=False,
        default_template_types=["system"]
)

document_retriever = LangChainRetriever(
        embedding_model="all-MiniLM-L6-v2",
        vectorstore_type="chroma",
        vectorstore_path="./vectorstore",
        use_hybrid_search=True,
        chunk_size=1000,
        chunk_overlap=200
)

inferencer = Inferencer(
        model=llm,
        retriever=document_retriever,
        reranker=None,
        config=inferencer_config
)

async def get_response(question):
    result = await inferencer.infer(question, "rag_response")
    return result

async def get_stream_response(question):
    async for item in inferencer.infer_stream(query = question,
                                             enable_reranking=False,
                                             template_type="friendly",
                                             k=3):
            print("Stream Response :", item)
            yield item