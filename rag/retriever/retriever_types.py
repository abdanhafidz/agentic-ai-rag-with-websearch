
from typing import List, Dict, Any, Optional, Union


from dataclasses import dataclass
from enum import Enum

from langchain_core.documents import Document

class DocumentType(str, Enum):
    PDF = "pdf"
    DOCX = "docx"
    PPT = "ppt"
    PPTX = "pptx"
    TXT = "txt"

class ProcessingStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    ERROR = "error"

@dataclass
class DocumentMetadata:
    """Document metadata"""
    file_path: str
    file_name: str
    file_type: DocumentType
    file_size: int
    file_hash: str
    created_at: str
    processed_at: Optional[str] = None
    chunk_count: int = 0
    processing_status: ProcessingStatus = ProcessingStatus.PENDING
    error_message: Optional[str] = None

@dataclass
class RetrievalResult:
    """Retrieval result"""
    documents: List[Document]
    scores: List[float]
    query: Optional[str] = None
    retrieval_time: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = None

@dataclass
class ProcessingResult:
    """Document processing result"""
    success: bool
    document_metadata: DocumentMetadata
    chunks: List[Document]
    error_message: Optional[str] = None