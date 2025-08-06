from rag.retriever.retriever_types import (
    DocumentType, 
    RetrievalResult, 
)
from typing import List
from abc import ABC, abstractmethod

from langchain_community.document_loaders import (
    PyMuPDFLoader,
    Docx2txtLoader,
    UnstructuredPowerPointLoader,
    TextLoader
)


from rag.retriever.document_loader import BaseDocumentLoader
from langchain_core.documents import Document


import asyncio
import logging
from pathlib import Path

import hashlib
import os


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BaseRetriever(ABC):
    """Abstract base class for retrievers"""
    
    @abstractmethod
    async def add_documents(self, documents: List[Document]) -> bool:
        """Add documents to retriever"""
        pass
    
    @abstractmethod
    async def retrieve(self, query: str, k: int = 5) -> RetrievalResult:
        """Retrieve relevant documents"""
        pass
    
    @abstractmethod
    async def delete_documents(self, document_ids: List[str]) -> bool:
        """Delete documents by IDs"""
        pass

# ===== DOCUMENT LOADERS =====

class MultiFormatDocumentLoader(BaseDocumentLoader):
    """Document loader supporting multiple formats"""
    
    def __init__(self):
        self.loaders = {
            DocumentType.PDF: self._load_pdf,
            DocumentType.DOCX: self._load_docx,
            DocumentType.PPT: self._load_ppt,
            DocumentType.PPTX: self._load_pptx,
            DocumentType.TXT: self._load_txt
        }
    
    async def load_document(self, file_path: str) -> List[Document]:
        """Load document based on file extension"""
        try:
            file_path = Path(file_path)
            if not file_path.exists():
                raise FileNotFoundError(f"File not found: {file_path}")
            
            # Determine document type
            doc_type = self._get_document_type(file_path)
            
            # Load document
            loader_func = self.loaders.get(doc_type)
            if not loader_func:
                raise ValueError(f"Unsupported file type: {doc_type}")
            
            logger.info(f"Loading {doc_type} document: {file_path}")
            documents = await loader_func(str(file_path))
            
            # Add metadata to documents
            for doc in documents:
                doc.metadata.update({
                    "file_path": str(file_path),
                    "file_name": file_path.name,
                    "file_type": doc_type.value,
                    "file_size": file_path.stat().st_size,
                    "file_hash": self._calculate_file_hash(file_path)
                })
            
            return documents
            
        except Exception as e:
            logger.error(f"Error loading document {file_path}: {str(e)}")
            raise
    
    def get_supported_extensions(self) -> List[str]:
        """Get supported file extensions"""
        return [".pdf", ".docx", ".ppt", ".pptx", ".txt"]
    
    def _get_document_type(self, file_path: Path) -> DocumentType:
        """Determine document type from file extension"""
        extension = file_path.suffix.lower()
        mapping = {
            ".pdf": DocumentType.PDF,
            ".docx": DocumentType.DOCX,
            ".ppt": DocumentType.PPT,
            ".pptx": DocumentType.PPTX,
            ".txt": DocumentType.TXT
        }
        
        doc_type = mapping.get(extension)
        if not doc_type:
            raise ValueError(f"Unsupported file extension: {extension}")
        
        return doc_type
    
    def _calculate_file_hash(self, file_path: Path) -> str:
        """Calculate MD5 hash of file"""
        hash_md5 = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
    
    async def _load_pdf(self, file_path: str) -> List[Document]:
        """Load PDF document"""
        try:
            loader = PyMuPDFLoader(file_path)
            documents = await asyncio.get_event_loop().run_in_executor(
                None, loader.load
            )
            return documents
        except Exception as e:
            raise Exception(f"Error loading PDF: {str(e)}")
    
    async def _load_docx(self, file_path: str) -> List[Document]:
        """Load DOCX document"""
        try:
            loader = Docx2txtLoader(file_path)
            documents = await asyncio.get_event_loop().run_in_executor(
                None, loader.load
            )
            return documents
        except Exception as e:
            raise Exception(f"Error loading DOCX: {str(e)}")
    
    async def _load_ppt(self, file_path: str) -> List[Document]:
        """Load PPT document"""
        try:
            loader = UnstructuredPowerPointLoader(file_path)
            documents = await asyncio.get_event_loop().run_in_executor(
                None, loader.load
            )
            return documents
        except Exception as e:
            raise Exception(f"Error loading PPT: {str(e)}")
    
    async def _load_pptx(self, file_path: str) -> List[Document]:
        """Load PPTX document"""
        try:
            loader = UnstructuredPowerPointLoader(file_path)
            documents = await asyncio.get_event_loop().run_in_executor(
                None, loader.load
            )
            return documents
        except Exception as e:
            raise Exception(f"Error loading PPTX: {str(e)}")
    
    async def _load_txt(self, file_path: str) -> List[Document]:
        """Load TXT document"""
        try:
            loader = TextLoader(file_path)
            documents = await asyncio.get_event_loop().run_in_executor(
                None, loader.load
            )
            return documents
        except Exception as e:
            raise Exception(f"Error loading TXT: {str(e)}")