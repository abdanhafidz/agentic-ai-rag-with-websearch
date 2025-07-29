from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import time
from langchain_core.documents import Document
from rag.FlagEmbedding import BGEM3FlagModel

class BGEM3Reranker:
    """BGE-M3 based reranker with support for dense, sparse, and multi-vector scoring"""
    
    def __init__(self, 
                 model_name: str = 'BAAI/bge-m3',
                 use_fp16: bool = True,
                 weights: Dict[str, float] = None):
        """
        Initialize BGE-M3 reranker
        
        Args:
            model_name: Model name/path for BGE-M3
            use_fp16: Use FP16 for faster computation
            weights: Weights for different scoring methods
                    {'dense': 1.0, 'sparse': 1.0, 'colbert': 1.0}
        """
        self.model = BGEM3FlagModel(model_name, use_fp16=use_fp16)
        self.weights = weights or {'dense': 1.0, 'sparse': 1.0, 'colbert': 0.0}
        
    def _extract_text_from_documents(self, documents: List[Document]) -> List[str]:
        """Extract text content from LangChain documents"""
        return [doc.page_content for doc in documents]
    
    def _compute_dense_scores(self, query_embedding: np.ndarray, doc_embeddings: np.ndarray) -> np.ndarray:
        """Compute dense (semantic) similarity scores using matrix multiplication"""
        # BGE-M3 embeddings are already normalized, so we can use direct matrix multiplication
        # query_embedding shape: (embedding_dim,)
        # doc_embeddings shape: (num_docs, embedding_dim)
        scores = doc_embeddings @ query_embedding  # Direct matrix multiplication
        return scores
    
    def _compute_sparse_scores(self, query_sparse: Dict, doc_sparse_list: List[Dict]) -> List[float]:
        """Compute sparse (lexical) similarity scores"""
        scores = []
        for doc_sparse in doc_sparse_list:
            score = self.model.compute_lexical_matching_score(query_sparse, doc_sparse)
            scores.append(score)
        return scores
    
    def _compute_colbert_scores(self, query_colbert: np.ndarray, doc_colbert_list: List[np.ndarray]) -> List[float]:
        """Compute ColBERT multi-vector interaction scores using BGE-M3's native method"""
        scores = []
        for doc_colbert in doc_colbert_list:
            # Use BGE-M3's native ColBERT scoring method
            score = self.model.colbert_score(query_colbert, doc_colbert)
            scores.append(float(score))
        return scores
    
    def rerank(self, retrieval_result: RetrievalResult, top_k: Optional[int] = None) -> RetrievalResult:
        """
        Rerank documents using BGE-M3 multi-vector scoring
        
        Args:
            retrieval_result: Original retrieval result
            top_k: Number of top documents to return (None = return all)
            
        Returns:
            RetrievalResult: Reranked retrieval result
        """
        start_time = time.time()
        
        if not retrieval_result.query:
            raise ValueError("Query is required for reranking")
        
        if not retrieval_result.documents:
            return retrieval_result
        
        # Extract texts
        query_text = retrieval_result.query
        doc_texts = self._extract_text_from_documents(retrieval_result.documents)
        
        # Encode query and documents
        query_output = self.model.encode(
            [query_text], 
            return_dense=self.weights['dense'] > 0,
            return_sparse=self.weights['sparse'] > 0,
            return_colbert_vecs=self.weights['colbert'] > 0
        )
        
        doc_output = self.model.encode(
            doc_texts,
            return_dense=self.weights['dense'] > 0,
            return_sparse=self.weights['sparse'] > 0,
            return_colbert_vecs=self.weights['colbert'] > 0
        )
        
        # Compute individual scores
        final_scores = np.zeros(len(doc_texts))
        score_components = {}
        
        # Dense scores
        if self.weights['dense'] > 0:
            dense_scores = self._compute_dense_scores(
                query_output['dense_vecs'][0],
                doc_output['dense_vecs']
            )
            final_scores += self.weights['dense'] * dense_scores
            score_components['dense'] = dense_scores.tolist()
        
        # Sparse scores
        if self.weights['sparse'] > 0:
            sparse_scores = self._compute_sparse_scores(
                query_output['lexical_weights'][0],
                doc_output['lexical_weights']
            )
            final_scores += self.weights['sparse'] * np.array(sparse_scores)
            score_components['sparse'] = sparse_scores
        
        # ColBERT scores
        if self.weights['colbert'] > 0:
            colbert_scores = self._compute_colbert_scores(
                query_output['colbert_vecs'][0],
                doc_output['colbert_vecs']
            )
            final_scores += self.weights['colbert'] * np.array(colbert_scores)
            score_components['colbert'] = colbert_scores
        
        # Sort by scores (descending)
        sorted_indices = np.argsort(final_scores)[::-1]
        
        # Apply top_k filtering
        if top_k:
            sorted_indices = sorted_indices[:top_k]
        
        # Reorder documents and scores
        reranked_documents = [retrieval_result.documents[i] for i in sorted_indices]
        reranked_scores = [float(final_scores[i]) for i in sorted_indices]
        
        # Create metadata with score components
        rerank_metadata = {
            'reranker': 'BGE-M3',
            'weights': self.weights,
            'score_components': {
                component: [scores[i] for i in sorted_indices] 
                for component, scores in score_components.items()
            },
            'original_scores': [retrieval_result.scores[i] for i in sorted_indices] if retrieval_result.scores else None,
            'rerank_time': time.time() - start_time
        }
        
        # Merge with existing metadata
        final_metadata = retrieval_result.metadata.copy() if retrieval_result.metadata else {}
        final_metadata.update(rerank_metadata)
        
        return RetrievalResult(
            documents=reranked_documents,
            scores=reranked_scores,
            query=retrieval_result.query,
            retrieval_time=retrieval_result.retrieval_time,
            metadata=final_metadata
        )
    
    def rerank_with_scores(self, 
                          query: str, 
                          documents: List[Document], 
                          original_scores: Optional[List[float]] = None,
                          top_k: Optional[int] = None) -> RetrievalResult:
        """
        Convenience method to rerank documents directly
        
        Args:
            query: Query string
            documents: List of documents to rerank
            original_scores: Original retrieval scores (optional)
            top_k: Number of top documents to return
            
        Returns:
            RetrievalResult: Reranked result
        """
        retrieval_result = RetrievalResult(
            documents=documents,
            scores=original_scores or [0.0] * len(documents),
            query=query
        )
        
        return self.rerank(retrieval_result, top_k=top_k)
    
    def rerank_dense_only(self, retrieval_result: RetrievalResult, 
                         batch_size: int = 12, 
                         max_length: int = 8192,
                         top_k: Optional[int] = None) -> RetrievalResult:
        """
        Fast reranking using only dense embeddings (optimized for speed)
        
        Args:
            retrieval_result: Original retrieval result
            batch_size: Batch size for encoding
            max_length: Maximum sequence length
            top_k: Number of top documents to return
            
        Returns:
            RetrievalResult: Reranked result using only dense similarity
        """
        start_time = time.time()
        
        if not retrieval_result.query:
            raise ValueError("Query is required for reranking")
        
        if not retrieval_result.documents:
            return retrieval_result
        
        # Extract texts
        query_text = retrieval_result.query
        doc_texts = self._extract_text_from_documents(retrieval_result.documents)
        
        # Encode query
        query_embedding = self.model.encode(
            [query_text], 
            batch_size=batch_size,
            max_length=max_length
        )['dense_vecs'][0]
        
        # Encode documents
        doc_embeddings = self.model.encode(
            doc_texts,
            batch_size=batch_size,
            max_length=max_length
        )['dense_vecs']
        
        # Compute similarity scores using matrix multiplication
        similarity_scores = doc_embeddings @ query_embedding
        
        # Sort by scores (descending)
        sorted_indices = np.argsort(similarity_scores)[::-1]
        
        # Apply top_k filtering
        if top_k:
            sorted_indices = sorted_indices[:top_k]
        
        # Reorder documents and scores
        reranked_documents = [retrieval_result.documents[i] for i in sorted_indices]
        reranked_scores = [float(similarity_scores[i]) for i in sorted_indices]
        
        # Create metadata
        rerank_metadata = {
            'reranker': 'BGE-M3-Dense',
            'method': 'dense_only',
            'batch_size': batch_size,
            'max_length': max_length,
            'original_scores': [retrieval_result.scores[i] for i in sorted_indices] if retrieval_result.scores else None,
            'rerank_time': time.time() - start_time
        }
        
        # Merge with existing metadata
        final_metadata = retrieval_result.metadata.copy() if retrieval_result.metadata else {}
        final_metadata.update(rerank_metadata)
        
        return RetrievalResult(
            documents=reranked_documents,
            scores=reranked_scores,
            query=retrieval_result.query,
            retrieval_time=retrieval_result.retrieval_time,
            metadata=final_metadata
        )
    
    def rerank_colbert_only(self, retrieval_result: RetrievalResult, 
                           top_k: Optional[int] = None) -> RetrievalResult:
        """
        Reranking using only ColBERT multi-vector interaction
        
        Args:
            retrieval_result: Original retrieval result
            top_k: Number of top documents to return
            
        Returns:
            RetrievalResult: Reranked result using only ColBERT scoring
        """
        start_time = time.time()
        
        if not retrieval_result.query:
            raise ValueError("Query is required for reranking")
        
        if not retrieval_result.documents:
            return retrieval_result
        
        # Extract texts
        query_text = retrieval_result.query
        doc_texts = self._extract_text_from_documents(retrieval_result.documents)
        
        # Encode query and documents with ColBERT vectors
        query_output = self.model.encode([query_text], return_colbert_vecs=True)
        doc_output = self.model.encode(doc_texts, return_colbert_vecs=True)
        
        # Compute ColBERT scores using BGE-M3's native method
        colbert_scores = []
        query_colbert = query_output['colbert_vecs'][0]
        
        for doc_colbert in doc_output['colbert_vecs']:
            score = self.model.colbert_score(query_colbert, doc_colbert)
            colbert_scores.append(float(score))
        
        colbert_scores = np.array(colbert_scores)
        
        # Sort by scores (descending)
        sorted_indices = np.argsort(colbert_scores)[::-1]
        
        # Apply top_k filtering
        if top_k:
            sorted_indices = sorted_indices[:top_k]
        
        # Reorder documents and scores
        reranked_documents = [retrieval_result.documents[i] for i in sorted_indices]
        reranked_scores = [float(colbert_scores[i]) for i in sorted_indices]
        
        # Create metadata
        rerank_metadata = {
            'reranker': 'BGE-M3-ColBERT',
            'method': 'colbert_only',
            'original_scores': [retrieval_result.scores[i] for i in sorted_indices] if retrieval_result.scores else None,
            'rerank_time': time.time() - start_time
        }
        
        # Merge with existing metadata
        final_metadata = retrieval_result.metadata.copy() if retrieval_result.metadata else {}
        final_metadata.update(rerank_metadata)
        
        return RetrievalResult(
            documents=reranked_documents,
            scores=reranked_scores,
            query=retrieval_result.query,
            retrieval_time=retrieval_result.retrieval_time,
            metadata=final_metadata
        )


# Usage example
def example_usage():
    """Example of how to use the BGE-M3 reranker"""
    
    # Initialize reranker
    reranker = BGEM3Reranker(
        weights={
            'dense': 1.0,    # Semantic similarity
            'sparse': 0.3,   # Lexical matching  
            'colbert': 0.0   # Multi-vector interaction (disabled for faster computation)
        }
    )
    
    # Create sample documents
    documents = [
        Document(page_content="BGE M3 is an embedding model supporting dense retrieval, lexical matching and multi-vector interaction."),
        Document(page_content="BM25 is a bag-of-words retrieval function that ranks documents based on query terms."),
        Document(page_content="Dense retrieval uses neural embeddings to find semantically similar documents."),
        Document(page_content="Sparse retrieval methods like TF-IDF focus on exact term matching.")
    ]
    
    # Create retrieval result
    retrieval_result = RetrievalResult(
        documents=documents,
        scores=[0.8, 0.6, 0.7, 0.5],  # Original retrieval scores
        query="What is BGE M3?",
        retrieval_time=0.1
    )
    
    print("=== Multi-vector Reranking (Dense + Sparse) ===")
    # Rerank documents using multi-vector approach
    reranked_result = reranker.rerank(retrieval_result, top_k=3)
    
    print(f"Query: {reranked_result.query}")
    print(f"Reranked {len(reranked_result.documents)} documents:")
    
    for i, (doc, score) in enumerate(zip(reranked_result.documents, reranked_result.scores)):
        print(f"\n{i+1}. Score: {score:.4f}")
        print(f"   Content: {doc.page_content[:100]}...")
    
    print("\n=== Dense-only Reranking (Fast) ===")
    # Fast dense-only reranking
    dense_result = reranker.rerank_dense_only(retrieval_result, top_k=3)
    
    for i, (doc, score) in enumerate(zip(dense_result.documents, dense_result.scores)):
        print(f"\n{i+1}. Dense Score: {score:.4f}")
        print(f"   Content: {doc.page_content[:80]}...")
    
    print("\n=== ColBERT-only Reranking (Precise) ===")
    # ColBERT-only reranking for high precision
    colbert_result = reranker.rerank_colbert_only(retrieval_result, top_k=3)
    
    for i, (doc, score) in enumerate(zip(colbert_result.documents, colbert_result.scores)):
        print(f"\n{i+1}. ColBERT Score: {score:.4f}")
        print(f"   Content: {doc.page_content[:80]}...")
    
    print(f"\nPerformance comparison:")
    print(f"Multi-vector time: {reranked_result.metadata['rerank_time']:.4f}s")
    print(f"Dense-only time: {dense_result.metadata['rerank_time']:.4f}s") 
    print(f"ColBERT-only time: {colbert_result.metadata['rerank_time']:.4f}s")