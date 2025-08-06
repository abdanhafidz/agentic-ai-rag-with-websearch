from typing import List, Dict, Any, Optional, Union
from langchain_text_splitters import RecursiveCharacterTextSplitter
import asyncio
import logging

from langchain_core.documents import Document
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DocumentProcessor:
    """Document processor for chunking and preprocessing"""
    
    def __init__(self, 
                 chunk_size: int = 1000,
                 chunk_overlap: int = 200,
                 separators: Optional[List[str]] = None):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # Default separators for better chunking
        if separators is None:
            separators = ["\n\n", "\n", " ", ""]
        
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=separators,
            length_function=len
        )
    
    async def process_documents(self, documents: List[Document]) -> List[Document]:
        """Process documents by splitting into chunks"""
        try:
            logger.info(f"Processing {len(documents)} documents")
            
            # Split documents into chunks
            chunks = await asyncio.get_event_loop().run_in_executor(
                None, self.text_splitter.split_documents, documents
            )
            
            # Add chunk metadata
            for i, chunk in enumerate(chunks):
                chunk.metadata.update({
                    "chunk_id": i,
                    "chunk_size": len(chunk.page_content),
                    "processed_at": str(asyncio.get_event_loop().time())
                })
            
            logger.info(f"Created {len(chunks)} chunks from {len(documents)} documents")
            return chunks
            
        except Exception as e:
            logger.error(f"Error processing documents: {str(e)}")
            raise