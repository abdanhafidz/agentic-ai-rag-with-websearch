from rag.retriever.base_retriever import BaseRetriever

# Embeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_openai import OpenAIEmbeddings

# Vector stores
from langchain_community.vectorstores import Chroma, FAISS, Pinecone

# Retriever base
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import ContextualCompressionRetriever

from typing import Dict, Optional, List
from rag.retriever.document_loader import MultiFormatDocumentLoader
from rag.retriever.document_processor import DocumentProcessor
from rag.retriever.retriever_types import ProcessingResult, ProcessingStatus, RetrievalResult, DocumentMetadata

import asyncio
from pathlib import Path
import logging
from langchain_core.documents import Document

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LangChainRetriever(BaseRetriever):
    """LangChain-based retriever with multiple format support"""

    def __init__(self,
                 embedding_model: str = "text-embedding-3-small",
                 vectorstore_type: str = "chroma",
                 vectorstore_path: Optional[str] = None,
                 use_hybrid_search: bool = True,
                 **kwargs):

        self.embedding_model = embedding_model
        self.vectorstore_type = vectorstore_type
        self.vectorstore_path = vectorstore_path or "./vectorstore"
        self.use_hybrid_search = use_hybrid_search

        self.document_loader = MultiFormatDocumentLoader()
        self.document_processor = DocumentProcessor(**kwargs)
        self.embeddings = self._initialize_embeddings()
        self.vectorstore = self._initialize_vectorstore()
        self.retriever = self._initialize_retriever()

        self.processed_documents: Dict[str, DocumentMetadata] = {}

        logger.info(f"LangChainRetriever initialized with {vectorstore_type} vectorstore")

    def _initialize_embeddings(self):
        try:
            if self.embedding_model.startswith("text-embedding"):
                return OpenAIEmbeddings(model=self.embedding_model)
            else:
                return HuggingFaceEmbeddings(model_name=self.embedding_model)
        except Exception as e:
            logger.error(f"Error initializing embeddings: {str(e)}")
            return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    def _initialize_vectorstore(self):
        try:
            if self.vectorstore_type.lower() == "chroma":
                return Chroma(
                    persist_directory=self.vectorstore_path,
                    embedding_function=self.embeddings
                )
            elif self.vectorstore_type.lower() == "faiss":
                return FAISS(
                    embedding_function=self.embeddings,
                    index_path=self.vectorstore_path
                )
            else:
                raise ValueError(f"Unsupported vectorstore type: {self.vectorstore_type}")
        except Exception as e:
            logger.error(f"Error initializing vectorstore: {str(e)}")
            return FAISS.from_documents([], self.embeddings)

    def _initialize_retriever(self):
        try:
            vector_retriever = VectorStoreRetriever(
                vectorstore=self.vectorstore,
                search_kwargs={"k": 10}
            )
            if self.use_hybrid_search:
                self.bm25_retriever = None  # initialized later after adding docs
                return vector_retriever  # temporary fallback
            else:
                return vector_retriever
        except Exception as e:
            logger.error(f"Error initializing retriever: {str(e)}")
            return VectorStoreRetriever(vectorstore=self.vectorstore)

    async def add_document_from_file(self, file_path: str) -> ProcessingResult:
        try:
            file_path = Path(file_path)
            if not file_path.exists():
                return ProcessingResult(
                    success=False,
                    document_metadata=None,
                    chunks=[],
                    error_message=f"File not found: {file_path}"
                )

            doc_metadata = DocumentMetadata(
                file_path=str(file_path),
                file_name=file_path.name,
                file_type=self.document_loader._get_document_type(file_path),
                file_size=file_path.stat().st_size,
                file_hash=self.document_loader._calculate_file_hash(file_path),
                created_at=str(asyncio.get_event_loop().time()),
                processing_status=ProcessingStatus.PROCESSING
            )

            documents = await self.document_loader.load_document(str(file_path))
            chunks = await self.document_processor.process_documents(documents)
            await self.add_documents(chunks)

            doc_metadata.chunk_count = len(chunks)
            doc_metadata.processing_status = ProcessingStatus.COMPLETED
            doc_metadata.processed_at = str(asyncio.get_event_loop().time())
            self.processed_documents[doc_metadata.file_hash] = doc_metadata

            logger.info(f"Successfully processed {file_path}: {len(chunks)} chunks")

            return ProcessingResult(
                success=True,
                document_metadata=doc_metadata,
                chunks=chunks
            )

        except Exception as e:
            error_msg = f"Error processing document {file_path}: {str(e)}"
            logger.error(error_msg)

            return ProcessingResult(
                success=False,
                document_metadata=doc_metadata if 'doc_metadata' in locals() else None,
                chunks=[],
                error_message=error_msg
            )

    async def add_documents(self, documents: List[Document]) -> bool:
        try:
            if not documents:
                return True

            await asyncio.get_event_loop().run_in_executor(
                None, self.vectorstore.add_documents, documents
            )

            if self.use_hybrid_search:
                await self._update_bm25_retriever(documents)

            logger.info(f"Added {len(documents)} documents to vector store")
            return True

        except Exception as e:
            logger.error(f"Error adding documents: {str(e)}")
            return False

    async def _update_bm25_retriever(self, documents: List[Document]):
        try:
            self.bm25_retriever = BM25Retriever.from_documents(documents)
            self.retriever = ContextualCompressionRetriever(
                base_compressor=None,  # Optional: add compressor like CohereRerank or LLM-based
                base_retriever=self.bm25_retriever  # Example: use BM25 as base, can combine
            )
        except Exception as e:
            logger.error(f"Error updating BM25 retriever: {str(e)}")

    async def retrieve(self, query: str, k: int = 5) -> RetrievalResult:
        try:
            import time
            start_time = time.time()
            logger.info(f"Retrieving documents for query: '{query}'")

            retrieved_docs = await asyncio.get_event_loop().run_in_executor(
                None, self.retriever.get_relevant_documents, query
            )
            retrieved_docs = retrieved_docs[:k]
            scores = [0.9 - (i * 0.1) for i in range(len(retrieved_docs))]

            retrieval_time = time.time() - start_time

            logger.info(f"Retrieved {len(retrieved_docs)} documents in {retrieval_time:.2f}s")

            return RetrievalResult(
                documents=retrieved_docs,
                scores=scores,
                query=query,
                retrieval_time=retrieval_time,
                metadata={
                    "vectorstore_type": self.vectorstore_type,
                    "embedding_model": self.embedding_model,
                    "hybrid_search": self.use_hybrid_search
                }
            )

        except Exception as e:
            logger.error(f"Error retrieving documents: {str(e)}")
            raise

    async def delete_documents(self, document_ids: List[str]) -> bool:
        try:
            if hasattr(self.vectorstore, 'delete'):
                await asyncio.get_event_loop().run_in_executor(
                    None, self.vectorstore.delete, document_ids
                )
            logger.info(f"Deleted {len(document_ids)} documents")
            return True
        except Exception as e:
            logger.error(f"Error deleting documents: {str(e)}")
            return False

    def get_document_metadata(self, file_hash: str) -> Optional[DocumentMetadata]:
        return self.processed_documents.get(file_hash)

    def list_processed_documents(self) -> List[DocumentMetadata]:
        return list(self.processed_documents.values())

    def get_supported_formats(self) -> List[str]:
        return self.document_loader.get_supported_extensions()
