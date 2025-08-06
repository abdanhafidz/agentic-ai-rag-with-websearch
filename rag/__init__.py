from rag.pipeline.language_model import LM, LMConfig
from rag.retriever.langchain_retriever import LangChainRetriever
from rag.inference.inferencer import Inferencer, InferencerConfig
from rag.agents.customer_service_agent import CSAgent
from rag.agents.query_maker_agent import QueryMakerAgent
from langchain_core.documents import Document
from rag.web_search.duckduckgo_search import DuckDuckGoSearch
from rag.chat_template import get_chat_template
from transformers import BitsAndBytesConfig
import torch

import logging
import sys

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(funcName)s() - %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
bnb = BitsAndBytesConfig(
                            load_in_4bit=True,                      # Enable 4-bit quantization
                            bnb_4bit_use_double_quant=True,         # Use double quantization
                            bnb_4bit_quant_type="nf4",              # Use NF4 quantization
                            bnb_4bit_compute_dtype=torch.bfloat16,  # Compute dtype for 4bit base models
        )
config = LMConfig(
                model_name = "meta-llama/Llama-3.2-1B-Instruct",
                temperature=0.7,
                max_length=512,
                generation_timeout=100,
                repetition_penalty=1.3,
                max_workers = 1,
                quantization_config = bnb
)

llm = LM(
        config = config
)

inferencer_config = InferencerConfig(
        default_k=5,
        enable_reranking=False,
        default_template_types="main_template"
)

document_retriever = LangChainRetriever(
        embedding_model="sentence-transformers/all-MiniLM-L6-v2",
        vectorstore_type="chroma",
        vectorstore_path="vectorstore/",
        use_hybrid_search=True,
        chunk_size=1000,
        chunk_overlap=200
)

ddgs = DuckDuckGoSearch()

cs_inferencer = Inferencer(
        model=llm,
        retriever=document_retriever,
        # search_engine = ddgs,
        reranker=None,
        config=inferencer_config
)

query_maker_inferencer = Inferencer(
        model=llm,
        config=inferencer_config
)

cs_agent = CSAgent(
    inferencer = cs_inferencer,
    prompt_template = get_chat_template("customer_service")
)

query_maker_chat_template = get_chat_template("query_maker")
query_maker_chat_template[1]["content"] = """{question}"""

query_maker_agent = QueryMakerAgent(
    inferencer = query_maker_inferencer,
    prompt_template = query_maker_chat_template
)




