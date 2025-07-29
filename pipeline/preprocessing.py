import re
import json
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass
import logging
from datetime import datetime
import hashlib

# Import types yang sudah ada
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass
from enum import Enum
from langchain_core.documents import Document
from rag.retriever.retriever_types import *
@dataclass
class PreprocessingConfig:
    """Konfigurasi untuk preprocessing"""
    # Text cleaning options
    remove_extra_whitespace: bool = True
    remove_special_chars: bool = False
    normalize_unicode: bool = True
    remove_urls: bool = False
    remove_emails: bool = False
    
    # Chunking options
    enable_chunking: bool = False        # Apakah perlu chunking lagi
    chunk_size: int = 500
    chunk_overlap: int = 50
    chunk_method: str = "sentence"       # "sentence", "paragraph", "fixed"
    
    # Content filtering
    min_content_length: int = 20
    max_content_length: int = 3000
    filter_empty_content: bool = True
    filter_duplicate_content: bool = True
    
    # Metadata options
    extract_metadata: bool = True
    include_retrieval_info: bool = True
    include_document_info: bool = True
    include_timestamps: bool = True
    
    # Scoring options
    use_retrieval_scores: bool = True    # Use scores dari retrieval system
    normalize_scores: bool = True        # Normalize scores ke range 0-1
    min_score_threshold: float = 0.0    # Filter berdasarkan minimum score
    score_boost_factor: float = 1.0     # Boost factor untuk scores

class RetrievalPreprocessor:
    """
    Preprocessor untuk RetrievalResult
    Mengkonversi RetrievalResult menjadi List[RetrievalResult] yang siap untuk RAG
    """
    
    def __init__(self, config: Optional[PreprocessingConfig] = None):
        """
        Initialize preprocessor
        
        Args:
            config: Preprocessing configuration
        """
        self.config = config or PreprocessingConfig()
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Regex patterns untuk cleaning
        self.url_pattern = re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
        self.email_pattern = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b')
        self.special_chars_pattern = re.compile(r'[^\w\s\.\,\!\?\;\:\-\(\)\[\]\{\}\"\'\/]')
        self.whitespace_pattern = re.compile(r'\s+')
        
        # Cache untuk duplicate detection
        self._seen_content_hashes = set()
    
    def process_retrieval_result(self, retrieval_result: RetrievalResult) -> List[RetrievalResult]:
        """
        Main method: Process RetrievalResult menjadi List[RetrievalResult]
        
        Args:
            retrieval_result: RetrievalResult dari retrieval system
            
        Returns:
            List[RetrievalResult] yang siap untuk RAG
        """
        if not retrieval_result.documents:
            self.logger.warning("No documents in retrieval result")
            return []
        
        if len(retrieval_result.documents) != len(retrieval_result.scores):
            self.logger.warning(
                f"Documents count ({len(retrieval_result.documents)}) != "
                f"Scores count ({len(retrieval_result.scores)})"
            )
        
        self.logger.info(
            f"Processing {len(retrieval_result.documents)} documents from retrieval result for query: '{retrieval_result.query}'"
        )
        
        # Clear cache untuk setiap batch baru
        self._seen_content_hashes.clear()
        
        contexts = []
        
        # Process setiap document
        for i, doc in enumerate(retrieval_result.documents):
            try:
                # Get corresponding score
                score = retrieval_result.scores[i] if i < len(retrieval_result.scores) else 0.0
                
                # Process single document
                processed_contexts = self._process_single_document(
                    document=doc,
                    retrieval_score=score,
                    document_index=i,
                    total_documents=len(retrieval_result.documents),
                    retrieval_result=retrieval_result
                )
                
                contexts.extend(processed_contexts)
                
            except Exception as e:
                self.logger.error(f"Error processing document {i}: {e}")
                continue
        
        # Post-processing
        contexts = self._post_process_contexts(contexts)
        
        self.logger.info(f"Successfully processed {len(contexts)} contexts from retrieval result")
        
        return contexts
    
    def _process_single_document(self, 
                                document: Document,
                                retrieval_score: float,
                                document_index: int,
                                total_documents: int,
                                retrieval_result: RetrievalResult) -> List[RetrievalResult]:
        """
        Process single document menjadi RetrievalResult(s)
        
        Args:
            document: Langchain Document object
            retrieval_score: Score dari retrieval system
            document_index: Index document dalam batch
            total_documents: Total documents dalam batch
            retrieval_result: Original retrieval result untuk metadata
            
        Returns:
            List[RetrievalResult]
        """
        if not document.page_content or not document.page_content.strip():
            self.logger.warning(f"Empty content in document {document_index}")
            return []
        
        # Clean content
        cleaned_content = self._clean_text(document.page_content)
        
        if not cleaned_content:
            return []
        
        # Filter by length
        if len(cleaned_content) < self.config.min_content_length:
            self.logger.debug(f"Content too short in document {document_index}: {len(cleaned_content)} chars")
            return []
        
        if len(cleaned_content) > self.config.max_content_length:
            # Truncate content
            cleaned_content = self._truncate_content(cleaned_content)
            self.logger.debug(f"Content truncated in document {document_index}")
        
        # Check for duplicates
        if self.config.filter_duplicate_content:
            content_hash = hashlib.md5(cleaned_content.encode()).hexdigest()
            if content_hash in self._seen_content_hashes:
                self.logger.debug(f"Duplicate content detected in document {document_index}")
                return []
            self._seen_content_hashes.add(content_hash)
        
        # Filter by score threshold
        if self.config.use_retrieval_scores and retrieval_score < self.config.min_score_threshold:
            self.logger.debug(f"Score too low in document {document_index}: {retrieval_score}")
            return []
        
        # Chunking (if enabled)
        if self.config.enable_chunking:
            chunks = self._chunk_content(cleaned_content)
            contexts = []
            
            for chunk_index, chunk in enumerate(chunks):
                context = self._create_retrieved_context(
                    content=chunk,
                    document=document,
                    retrieval_score=retrieval_score,
                    document_index=document_index,
                    chunk_index=chunk_index,
                    total_chunks=len(chunks),
                    total_documents=total_documents,
                    retrieval_result=retrieval_result
                )
                contexts.append(context)
            
            return contexts
        else:
            # Single context per document
            context = self._create_retrieved_context(
                content=cleaned_content,
                document=document,
                retrieval_score=retrieval_score,
                document_index=document_index,
                chunk_index=None,
                total_chunks=1,
                total_documents=total_documents,
                retrieval_result=retrieval_result
            )
            
            return [context]
    
    def _create_retrieved_context(self,
                                 content: str,
                                 document: Document,
                                 retrieval_score: float,
                                 document_index: int,
                                 chunk_index: Optional[int],
                                 total_chunks: int,
                                 total_documents: int,
                                 retrieval_result: RetrievalResult) -> RetrievalResult:
        """
        Create RetrievalResult object
        """
        # Process score
        final_score = self._process_score(retrieval_score, document_index, total_documents)
        
        # Extract source
        source = self._extract_source(document)
        
        # Build metadata
        metadata = self._build_metadata(
            document=document,
            retrieval_result=retrieval_result,
            document_index=document_index,
            chunk_index=chunk_index,
            total_chunks=total_chunks,
            total_documents=total_documents,
            content=content
        )
        
        return RetrievalResult(
            content=content,
            source=source,
            score=final_score,
            metadata=metadata
        )
    
    def _clean_text(self, text: str) -> str:
        """Clean text berdasarkan konfigurasi"""
        if not text:
            return ""
        
        cleaned = text
        
        # Normalize unicode
        if self.config.normalize_unicode:
            import unicodedata
            cleaned = unicodedata.normalize('NFKC', cleaned)
        
        # Remove URLs
        if self.config.remove_urls:
            cleaned = self.url_pattern.sub('', cleaned)
        
        # Remove emails
        if self.config.remove_emails:
            cleaned = self.email_pattern.sub('', cleaned)
        
        # Remove special characters
        if self.config.remove_special_chars:
            cleaned = self.special_chars_pattern.sub(' ', cleaned)
        
        # Remove extra whitespace
        if self.config.remove_extra_whitespace:
            cleaned = self.whitespace_pattern.sub(' ', cleaned)
        
        return cleaned.strip()
    
    def _truncate_content(self, content: str) -> str:
        """Truncate content yang terlalu panjang"""
        max_length = self.config.max_content_length
        
        if len(content) <= max_length:
            return content
        
        # Try to cut at sentence boundary
        truncated = content[:max_length - 20]
        last_sentence_end = max(
            truncated.rfind('.'),
            truncated.rfind('!'),
            truncated.rfind('?')
        )
        
        if last_sentence_end > len(truncated) * 0.7:
            return truncated[:last_sentence_end + 1]
        else:
            # Cut at word boundary
            last_space = truncated.rfind(' ')
            if last_space > len(truncated) * 0.8:
                return truncated[:last_space] + "..."
            else:
                return truncated + "..."
    
    def _chunk_content(self, content: str) -> List[str]:
        """Chunk content jika diperlukan"""
        if len(content) <= self.config.chunk_size:
            return [content]
        
        if self.config.chunk_method == "sentence":
            return self._chunk_by_sentence(content)
        elif self.config.chunk_method == "paragraph":
            return self._chunk_by_paragraph(content)
        elif self.config.chunk_method == "fixed":
            return self._chunk_by_fixed_size(content)
        else:
            return [content]  # No chunking
    
    def _chunk_by_sentence(self, text: str) -> List[str]:
        """Chunk by sentences"""
        sentences = re.split(r'[.!?]+\s+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            if len(current_chunk) + len(sentence) > self.config.chunk_size and current_chunk:
                chunks.append(current_chunk.strip())
                
                # Handle overlap
                if self.config.chunk_overlap > 0:
                    overlap_text = current_chunk[-self.config.chunk_overlap:]
                    current_chunk = overlap_text + " " + sentence
                else:
                    current_chunk = sentence
            else:
                if current_chunk:
                    current_chunk += " " + sentence
                else:
                    current_chunk = sentence
        
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return chunks
    
    def _chunk_by_paragraph(self, text: str) -> List[str]:
        """Chunk by paragraphs"""
        paragraphs = text.split('\n\n')
        paragraphs = [p.strip() for p in paragraphs if p.strip()]
        
        chunks = []
        current_chunk = ""
        
        for paragraph in paragraphs:
            if len(current_chunk) + len(paragraph) > self.config.chunk_size and current_chunk:
                chunks.append(current_chunk.strip())
                current_chunk = paragraph
            else:
                if current_chunk:
                    current_chunk += "\n\n" + paragraph
                else:
                    current_chunk = paragraph
        
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return chunks
    
    def _chunk_by_fixed_size(self, text: str) -> List[str]:
        """Chunk by fixed size dengan overlap"""
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + self.config.chunk_size
            chunk = text[start:end]
            
            # Try to break at word boundary
            if end < len(text):
                last_space = chunk.rfind(' ')
                if last_space > len(chunk) * 0.8:
                    chunk = chunk[:last_space]
                    end = start + last_space
            
            chunks.append(chunk.strip())
            
            # Move with overlap
            start = end - self.config.chunk_overlap
            if start <= 0:
                start = end
        
        return [chunk for chunk in chunks if chunk]
    
    def _process_score(self, retrieval_score: float, document_index: int, total_documents: int) -> float:
        """Process and normalize score"""
        if not self.config.use_retrieval_scores:
            return 1.0
        
        score = retrieval_score * self.config.score_boost_factor
        
        # Normalize to 0-1 range jika diperlukan
        if self.config.normalize_scores:
            # Assume retrieval scores are already normalized, but ensure they are in range
            score = max(0.0, min(1.0, score))
        
        return round(score, 4)
    
    def _extract_source(self, document: Document) -> str:
        """Extract source dari document metadata"""
        metadata = document.metadata or {}
        
        # Try different metadata keys for source
        source_keys = ['source', 'file_name', 'filename', 'title', 'file_path', 'path']
        
        for key in source_keys:
            if key in metadata and metadata[key]:
                return str(metadata[key])
        
        # Fallback to generic source
        return "unknown_source"
    
    def _build_metadata(self,
                       document: Document,
                       retrieval_result: RetrievalResult,
                       document_index: int,
                       chunk_index: Optional[int],
                       total_chunks: int,
                       total_documents: int,
                       content: str) -> Dict[str, Any]:
        """Build comprehensive metadata"""
        metadata = {}
        
        if self.config.extract_metadata:
            # Include original document metadata
            if document.metadata and self.config.include_document_info:
                metadata.update({
                    "original_metadata": document.metadata,
                    "document_index": document_index,
                    "total_documents": total_documents
                })
            
            # Include chunking info
            if chunk_index is not None:
                metadata.update({
                    "chunk_index": chunk_index,
                    "total_chunks": total_chunks,
                    "is_chunked": total_chunks > 1
                })
            
            # Include retrieval info
            if self.config.include_retrieval_info:
                metadata.update({
                    "retrieval_query": retrieval_result.query,
                    "retrieval_time": retrieval_result.retrieval_time,
                    "retrieval_metadata": retrieval_result.metadata
                })
            
            # Include processing info
            if self.config.include_timestamps:
                metadata.update({
                    "processed_at": datetime.now().isoformat(),
                    "processor_config": {
                        "chunking_enabled": self.config.enable_chunking,
                        "chunk_method": self.config.chunk_method if self.config.enable_chunking else None,
                        "cleaning_enabled": any([
                            self.config.remove_extra_whitespace,
                            self.config.remove_special_chars,
                            self.config.normalize_unicode,
                            self.config.remove_urls,
                            self.config.remove_emails
                        ])
                    }
                })
            
            # Content statistics
            word_count = len(content.split())
            sentence_count = len(re.split(r'[.!?]+', content))
            
            metadata.update({
                "content_stats": {
                    "character_count": len(content),
                    "word_count": word_count,
                    "sentence_count": max(1, sentence_count),
                    "avg_words_per_sentence": round(word_count / max(1, sentence_count), 1)
                }
            })
        
        return metadata
    
    def _post_process_contexts(self, contexts: List[RetrievalResult]) -> List[RetrievalResult]:
        """Post-processing untuk final contexts"""
        if not contexts:
            return contexts
        
        # Sort by score (descending)
        if self.config.use_retrieval_scores:
            contexts.sort(key=lambda x: x.score or 0.0, reverse=True)
        
        # Additional filtering jika diperlukan
        filtered_contexts = []
        for ctx in contexts:
            if self.config.filter_empty_content and not ctx.content.strip():
                continue
            filtered_contexts.append(ctx)
        
        return filtered_contexts
    
    def get_processing_stats(self, contexts: List[RetrievalResult]) -> Dict[str, Any]:
        """Get statistics tentang processed contexts"""
        if not contexts:
            return {"total_contexts": 0}
        
        total_contexts = len(contexts)
        total_words = sum(len(ctx.content.split()) for ctx in contexts)
        total_chars = sum(len(ctx.content) for ctx in contexts)
        
        # Score distribution
        scores = [ctx.score for ctx in contexts if ctx.score is not None]
        
        # Source distribution
        sources = {}
        for ctx in contexts:
            if ctx.source:
                sources[ctx.source] = sources.get(ctx.source, 0) + 1
        
        # Chunking stats
        chunked_contexts = sum(1 for ctx in contexts 
                             if ctx.metadata and ctx.metadata.get("is_chunked", False))
        
        stats = {
            "total_contexts": total_contexts,
            "total_words": total_words,
            "total_characters": total_chars,
            "avg_words_per_context": round(total_words / total_contexts, 1),
            "avg_chars_per_context": round(total_chars / total_contexts, 1),
            "chunked_contexts": chunked_contexts,
            "chunking_percentage": round((chunked_contexts / total_contexts) * 100, 1)
        }
        
        if scores:
            stats["score_stats"] = {
                "min_score": min(scores),
                "max_score": max(scores),
                "avg_score": round(sum(scores) / len(scores), 4),
                "median_score": round(sorted(scores)[len(scores)//2], 4)
            }
        
        if sources:
            stats["source_distribution"] = sources
            stats["unique_sources"] = len(sources)
        
        # Content length distribution
        lengths = [len(ctx.content) for ctx in contexts]
        stats["content_length_stats"] = {
            "min_length": min(lengths),
            "max_length": max(lengths),
            "avg_length": round(sum(lengths) / len(lengths), 1)
        }
        
        return stats
    
    def batch_process_retrieval_results(self, 
                                       retrieval_results: List[RetrievalResult]) -> List[RetrievalResult]:
        """
        Process multiple RetrievalResult objects sekaligus
        
        Args:
            retrieval_results: List of RetrievalResult objects
            
        Returns:
            Combined List[RetrievalResult]
        """
        if not retrieval_results:
            return []
        
        self.logger.info(f"Batch processing {len(retrieval_results)} retrieval results")
        
        all_contexts = []
        
        for i, result in enumerate(retrieval_results):
            try:
                contexts = self.process_retrieval_result(result)
                
                # Add batch info to metadata
                for ctx in contexts:
                    if ctx.metadata:
                        ctx.metadata["batch_index"] = i
                        ctx.metadata["batch_query"] = result.query
                    else:
                        ctx.metadata = {
                            "batch_index": i,
                            "batch_query": result.query
                        }
                
                all_contexts.extend(contexts)
                
            except Exception as e:
                self.logger.error(f"Error processing retrieval result {i}: {e}")
                continue
        
        # Final post-processing untuk batch
        all_contexts = self._post_process_contexts(all_contexts)
        
        self.logger.info(f"Batch processing completed: {len(all_contexts)} total contexts")
        
        return all_contexts
    
    def filter_contexts_by_query_relevance(self, 
                                          contexts: List[RetrievalResult],
                                          query: str,
                                          min_relevance_score: float = 0.5) -> List[RetrievalResult]:
        """
        Filter contexts berdasarkan relevance dengan query (simple keyword matching)
        
        Args:
            contexts: List of RetrievalResult
            query: Original query string
            min_relevance_score: Minimum relevance score threshold
            
        Returns:
            Filtered List[RetrievalResult]
        """
        if not contexts or not query:
            return contexts
        
        query_words = set(query.lower().split())
        filtered_contexts = []
        
        for ctx in contexts:
            content_words = set(ctx.content.lower().split())
            
            # Simple relevance calculation: overlap of words
            overlap = len(query_words.intersection(content_words))
            relevance_score = overlap / len(query_words) if query_words else 0.0
            
            if relevance_score >= min_relevance_score:
                # Update metadata dengan relevance info
                if ctx.metadata:
                    ctx.metadata["query_relevance_score"] = round(relevance_score, 3)
                    ctx.metadata["matched_query_words"] = list(query_words.intersection(content_words))
                else:
                    ctx.metadata = {
                        "query_relevance_score": round(relevance_score, 3),
                        "matched_query_words": list(query_words.intersection(content_words))
                    }
                
                filtered_contexts.append(ctx)
        
        # Sort by relevance score
        filtered_contexts.sort(
            key=lambda x: x.metadata.get("query_relevance_score", 0.0), 
            reverse=True
        )
        
        self.logger.info(
            f"Filtered {len(contexts)} contexts to {len(filtered_contexts)} "
            f"based on query relevance (min_score: {min_relevance_score})"
        )
        
        return filtered_contexts
    
    def deduplicate_contexts(self, 
                           contexts: List[RetrievalResult],
                           similarity_threshold: float = 0.8) -> List[RetrievalResult]:
        """
        Remove duplicate atau very similar contexts
        
        Args:
            contexts: List of RetrievalResult
            similarity_threshold: Threshold for considering contexts as duplicates
            
        Returns:
            Deduplicated List[RetrievalResult]
        """
        if not contexts:
            return contexts
        
        from difflib import SequenceMatcher
        
        def similarity(a, b):
            return SequenceMatcher(None, a, b).ratio()
        
        deduplicated = []
        
        for ctx in contexts:
            is_duplicate = False
            
            for existing_ctx in deduplicated:
                sim_score = similarity(ctx.content, existing_ctx.content)
                
                if sim_score >= similarity_threshold:
                    is_duplicate = True
                    
                    # Keep the one with higher score
                    if (ctx.score or 0.0) > (existing_ctx.score or 0.0):
                        # Replace existing with current
                        idx = deduplicated.index(existing_ctx)
                        deduplicated[idx] = ctx
                    
                    break
            
            if not is_duplicate:
                deduplicated.append(ctx)
        
        self.logger.info(
            f"Deduplicated {len(contexts)} contexts to {len(deduplicated)} "
            f"(similarity_threshold: {similarity_threshold})"
        )
        
        return deduplicated
    
    def merge_processing_results(self, 
                               processing_results: List[ProcessingResult]) -> List[RetrievalResult]:
        """
        Merge multiple ProcessingResult objects menjadi RetrievalResult list
        
        Args:
            processing_results: List of ProcessingResult objects
            
        Returns:
            List[RetrievalResult]
        """
        if not processing_results:
            return []
        
        all_contexts = []
        
        for i, proc_result in enumerate(processing_results):
            if not proc_result.success:
                self.logger.warning(f"Skipping failed processing result {i}: {proc_result.error_message}")
                continue
            
            if not proc_result.chunks:
                continue
            
            # Convert Document chunks to RetrievalResult
            for j, chunk in enumerate(proc_result.chunks):
                # Extract source dari document metadata
                source = self._extract_source(chunk)
                
                # Build metadata from ProcessingResult
                metadata = {
                    "document_metadata": proc_result.document_metadata.__dict__,
                    "chunk_index": j,
                    "total_chunks": len(proc_result.chunks),
                    "processing_result_index": i,
                    "processed_at": datetime.now().isoformat()
                }
                
                # Include original chunk metadata
                if chunk.metadata:
                    metadata["original_chunk_metadata"] = chunk.metadata
                
                # Clean content
                cleaned_content = self._clean_text(chunk.page_content)
                
                if not cleaned_content or len(cleaned_content) < self.config.min_content_length:
                    continue
                
                # Create RetrievalResult
                context = RetrievalResult(
                    content=cleaned_content,
                    source=source,
                    score=1.0,  # Default score for processing results
                    metadata=metadata
                )
                
                all_contexts.append(context)
        
        self.logger.info(f"Merged {len(processing_results)} processing results into {len(all_contexts)} contexts")
        
        return all_contexts