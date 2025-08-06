from rag.retriever.langchain_retriever import LangChainRetriever
from rag.pipeline.language_model import LM, LMConfig
from rag.retriever.retriever_types import RetrievalResult
from rag.web_search.duckduckgo_search import DuckDuckGoSearch
from langchain_core.documents import Document
# from rag.pipeline.reranker import BGEM3Reranker
from typing import List, Union, Dict, Any, Optional, AsyncGenerator
import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime

@dataclass
class InferencerConfig:
    """Konfigurasi untuk Inferencer"""
    default_k: int = 5
    max_contexts: int = 10
    enable_reranking: bool = False
    reranker_top_k: int = 5
    default_template_types: List[str] = None
    enable_logging: bool = True
    response_timeout: float = 30.0
    
    def __post_init__(self):
        if self.default_template_types is None:
            self.default_template_types = ["system", "instruction", "friendly"]

class Inferencer:
    """
    Advanced RAG Inferencer dengan support untuk streaming, reranking, dan multiple response types
    """
    
    def __init__(self, 
                 model: LM, 
                 retriever: LangChainRetriever = None, 
                 search_engine = None,
                 reranker=None,
                 config: Optional[InferencerConfig] = None):
        """
        Initialize Inferencer
        
        Args:
            model: LM instance
            retriever: LangChainRetriever instance
            reranker: Reranker instance (optional)
            config: InferencerConfig (optional)
        """
        self.model = model
        self.retriever = retriever
        self.reranker = reranker
        self.search_engine = search_engine
        self.config = config or InferencerConfig()
        
        # Setup logging
        if self.config.enable_logging:
            logging.basicConfig(level=logging.INFO)
            self.logger = logging.getLogger(__name__)
        else:
            self.logger = logging.getLogger(__name__)
            self.logger.setLevel(logging.ERROR)
        
        # Model loading flag
        self._model_loaded = False
    
    async def _ensure_model_loaded(self):
        """Pastikan model sudah diload (hanya sekali)"""
        if not self._model_loaded:
            self.logger.info("Loading model...")
            await self.model.load_model()
            self._model_loaded = True
            self.logger.info("Model loaded successfully")
    
    async def retrieve_context(self, 
                             query: str, 
                             k: Optional[int] = None) -> RetrievalResult:
        """
        Retrieve context documents
        
        Args:
            query: Search query
            k: Number of documents to retrieve
            
        Returns:
            RetrievalResult object
        """
        k = k or self.config.default_k
        self.logger.info(f"Retrieving {k} contexts for query: {query[:50]}...")
        
        try:
            start_time = datetime.now()
            contexts = await self.retriever.retrieve(query, k=k)
            self.logger.info(f"Retrieved Contexts : {contexts}")
            retrieval_time = (datetime.now() - start_time).total_seconds()
            
            self.logger.info(f"Retrieved {len(contexts.documents) if hasattr(contexts, 'documents') else len(contexts)} contexts in {retrieval_time:.2f}s")
            return contexts
            
        except Exception as e:
            self.logger.error(f"Error during retrieval: {e}")
            raise
    
    async def rerank_contexts(self, 
                            contexts: RetrievalResult, 
                            query: str,
                            top_k: Optional[int] = None) -> RetrievalResult:
        """
        Rerank retrieved contexts
        
        Args:
            contexts: Retrieved contexts
            query: Original query
            top_k: Number of top contexts to keep after reranking
            
        Returns:
            Reranked RetrievalResult object
        """
        if not self.reranker or not self.config.enable_reranking:
            self.logger.info("Reranking disabled or reranker not available")
            return contexts
        
        top_k = top_k or self.config.reranker_top_k
        self.logger.info(f"Reranking contexts, keeping top {top_k}")
        
        try:
            start_time = datetime.now()
            reranked_contexts = await self.reranker.rerank(
                query=query,
                contexts=contexts,
                top_k=top_k
            )
            rerank_time = (datetime.now() - start_time).total_seconds()
            
            self.logger.info(f"Reranking completed in {rerank_time:.2f}s")
            return reranked_contexts
            
        except Exception as e:
            self.logger.error(f"Error during reranking: {e}")
            # Return original contexts if reranking fails
            return contexts
    
    async def generate_response(self, 
                              contexts: RetrievalResult, 
                              query: Union[str, List[str]], 
                              response_type: Union[List[str], str] = None,
                              template_types: Optional[List[str]] = None,
                              max_new_tokens: Optional[int] = None,
                              **generation_kwargs) -> List[Dict[str, Any]]:
        """
        Generate responses based on contexts and query
        
        Args:
            contexts: Retrieved contexts
            query: User query or list of queries
            response_type: Type(s) of response to generate
            template_types: Template types for multi_response
            max_new_tokens: Maximum tokens to generate
            **generation_kwargs: Additional generation parameters
            
        Returns:
            List of response dictionaries
        """
        await self._ensure_model_loaded()
        
        # Default response types
        if response_type is None:
            response_type = ["rag_response"]
        elif isinstance(response_type, str):
            response_type = [response_type]
        
        # Default template types
        if template_types is None:
            template_types = self.config.default_template_types
        
        responses = []
        
        try:
            # RAG Response
            if "rag_response" in response_type:
                self.logger.info("Generating RAG response...")
                start_time = datetime.now()
                
                if isinstance(query, list):
                    # Handle multiple queries
                    rag_responses = {}
                    for i, q in enumerate(query):
                        rag_response = await self.model.rag_generate(
                            question=q,
                            contexts=contexts,
                            template_type="friendly",
                            max_new_tokens=max_new_tokens,
                            **generation_kwargs
                        )
                        rag_responses[f"query_{i}"] = rag_response
                    responses.append({"rag_response": rag_responses})
                else:
                    rag_response = await self.model.rag_generate(
                        question=query,
                        contexts=contexts,
                        template_type="friendly",
                        max_new_tokens=max_new_tokens,
                        **generation_kwargs
                    )
                    responses.append({"rag_response": rag_response})
                
                generation_time = (datetime.now() - start_time).total_seconds()
                self.logger.info(f"RAG response generated in {generation_time:.2f}s")
            
            # Multi-template Response
            if "multi_response" in response_type:
                self.logger.info("Generating multi-template responses...")
                start_time = datetime.now()
                
                if isinstance(query, list):
                    multi_responses = {}
                    for i, q in enumerate(query):
                        multi_response = await self.model.multi_template_generate(
                            question=q,
                            contexts=contexts,
                            template_types=template_types,
                            max_new_tokens=max_new_tokens,
                            **generation_kwargs
                        )
                        multi_responses[f"query_{i}"] = multi_response
                    responses.append({"multi_responses": multi_responses})
                else:
                    multi_responses = await self.model.multi_template_generate(
                        question=query,
                        contexts=contexts,
                        template_types=template_types,
                        max_new_tokens=max_new_tokens,
                        **generation_kwargs
                    )
                    responses.append({"multi_responses": multi_responses})
                
                generation_time = (datetime.now() - start_time).total_seconds()
                self.logger.info(f"Multi-template responses generated in {generation_time:.2f}s")
            
            # Batch Response (untuk multiple prompts tanpa RAG context)
            if "batch_response" in response_type:
                self.logger.info("Generating batch responses...")
                start_time = datetime.now()
                
                if isinstance(query, list):
                    batch_responses = await self.model.batch_generate(
                        query, 
                        max_new_tokens=max_new_tokens,
                        **generation_kwargs
                    )
                else:
                    batch_responses = await self.model.batch_generate(
                        [query], 
                        max_new_tokens=max_new_tokens,
                        **generation_kwargs
                    )
                
                responses.append({"batch_responses": batch_responses})
                
                generation_time = (datetime.now() - start_time).total_seconds()
                self.logger.info(f"Batch responses generated in {generation_time:.2f}s")
            
            return responses
            
        except Exception as e:
            self.logger.error(f"Error during response generation: {e}")
            raise
    
    async def generate_response_stream(self, 
                                     contexts: RetrievalResult, 
                                     query: str,
                                     template_type: str = "main_template",
                                     max_new_tokens: Optional[int] = None,
                                     **generation_kwargs) -> AsyncGenerator[str, None]:
        """
        Generate RAG response with streaming
        
        Args:
            contexts: Retrieved contexts
            query: User query
            template_type: Template type to use
            max_new_tokens: Maximum tokens to generate
            **generation_kwargs: Additional generation parameters
            
        Yields:
            Response chunks
        """
        await self._ensure_model_loaded()
        
        self.logger.info(f"Generating streaming RAG response with template: {template_type}")
        
        async for chunk in self.model.rag_generate_stream(
            question=query,
            contexts=contexts,
            template_type=template_type,
            max_new_tokens=max_new_tokens,
            **generation_kwargs
        ):
            yield chunk
    
    async def infer(self, 
                   query: str, 
                   response_type: Union[List[str], str] = None,
                   k: Optional[int] = None,
                   enable_reranking: Optional[bool] = None,
                   template_types: Optional[List[str]] = None,
                   max_new_tokens: Optional[int] = None,
                   **generation_kwargs) -> Dict[str, Any]:
        """
        Complete inference pipeline
        
        Args:
            query: User query or list of queries
            response_type: Type(s) of response to generate
            k: Number of contexts to retrieve
            enable_reranking: Whether to enable reranking
            template_types: Template types for multi_response
            max_new_tokens: Maximum tokens to generate
            **generation_kwargs: Additional generation parameters
            
        Returns:
            Dictionary with results and metadata
        """
        start_time = datetime.now()
        
        # Handle single query
        main_query = query[0] if isinstance(query, list) else query
        
        try:
            # Step 1: Retrieve contexts
            if(self.search_engine):
                await self.retrieve_from_search_engine(query, k = k)
            if(self.retriever):
                retrieved_contexts = await self.retrieve_context(main_query, k=k)
            else:
                retrieved_contexts  = ""
            # Step 2: Rerank contexts (if enabled)
            enable_rerank = enable_reranking if enable_reranking is not None else self.config.enable_reranking
            if enable_rerank:
                contexts = await self.rerank_contexts(retrieved_contexts, main_query)
            else:
                contexts = retrieved_contexts
            
            # Step 3: Generate responses
            responses = await self.generate_response(
                contexts=contexts,
                query=query,
                response_type=response_type,
                template_types=template_types,
                max_new_tokens=max_new_tokens,
                **generation_kwargs
            )
            
            total_time = (datetime.now() - start_time).total_seconds()
            
            # Prepare result
            result = {
                "query": query,
                "responses": responses,
                "contexts": contexts,
                "metadata": {
                    "total_time": total_time,
                    "retrieval_enabled": True,
                    "reranking_enabled": enable_rerank,
                    "num_contexts": len(contexts.documents) if hasattr(contexts, 'documents') else len(contexts),
                    "response_types": response_type,
                    "timestamp": datetime.now().isoformat()
                }
            }
            
            self.logger.info(f"Inference completed in {total_time:.2f}s")
            return result
            
        except Exception as e:
            self.logger.error(f"Error during inference: {e}")
            raise
    async def retrieve_from_search_engine(self, query: str, k: int = 3):
        """
        Alternative method: Process results as they come
        """
        from langchain_core.documents import Document
        
        search_results = []
        
        try:
            # Process results one by one as they come
            async for result in self.search_engine.search(query, max_results=k):
                self.logger.info(f"Processing SEO Result: {result[:100]}...")
                
                doc = Document(
                    page_content=result,
                    metadata={"source": "internet_search", "query": query}
                )
                search_results.append(doc)
                
                # Optionally add to retriever immediately
                await self.retriever.add_documents([doc])
            
            self.logger.info(f"Processed {len(search_results)} search results")
            return search_results
            
        except Exception as e:
            self.logger.error(f"Error in retrieve_from_search_engine_alternative: {e}", exc_info=True)
            raise
    async def infer_stream(self, 
                          query: str,
                          k: Optional[int] = None,
                          enable_reranking: Optional[bool] = None,
                          template_type: str = "main_template",
                          max_new_tokens: Optional[int] = None,
                          **generation_kwargs) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Complete inference pipeline with streaming response
        
        Args:
            query: User query
            k: Number of contexts to retrieve
            enable_reranking: Whether to enable reranking
            template_type: Template type to use
            max_new_tokens: Maximum tokens to generate
            **generation_kwargs: Additional generation parameters
            
        Yields:
            Dictionaries with stream data and metadata
        """
        start_time = datetime.now()
        
        try:
            # Step 1: Retrieve contexts
            if(self.search_engine):
                await self.retrieve_from_search_engine(query, k = k)
            if(self.retriever is not None):
                retrieved_contexts = await self.retrieve_context(query, k=k)
            else:
                retrieved_contexts = ""
            

            # Step 2: Rerank contexts (if enabled)
            enable_rerank = enable_reranking if enable_reranking is not None else self.config.enable_reranking
            if enable_rerank:
                contexts = await self.rerank_contexts(retrieved_contexts, query)
            else:
                contexts = retrieved_contexts
            
            # Yield metadata first
            setup_time = (datetime.now() - start_time).total_seconds()
            yield {
                "type": "metadata",
                "data": {
                    "query": query,
                    "setup_time": setup_time,
                    "num_contexts": len(contexts.documents) if hasattr(contexts, 'documents') else len(contexts),
                    "reranking_enabled": enable_rerank,
                    "template_type": template_type
                }
            }
            
            # Step 3: Stream response
            response_start = datetime.now()
            accumulated_text = ""
            
            async for chunk in self.generate_response_stream(
                contexts=contexts,
                query=query,
                template_type=template_type,
                max_new_tokens=max_new_tokens,
                **generation_kwargs
            ):
                accumulated_text += chunk
                yield {
                    "type": "chunk",
                    "data": {
                        "chunk": chunk,
                        "accumulated_text": accumulated_text,
                        "generation_time": (datetime.now() - response_start).total_seconds()
                    }
                }
            
            # Yield final metadata
            total_time = (datetime.now() - start_time).total_seconds()
            yield {
                "type": "complete",
                "data": {
                    "total_time": total_time,
                    "final_response": accumulated_text,
                    "contexts": contexts
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error during streaming inference: {e}")
            yield {
                "type": "error",
                "data": {
                    "error": str(e),
                    "error_time": (datetime.now() - start_time).total_seconds()
                }
            }
    
    async def batch_infer(self, 
                         queries: List[str],
                         response_type: Union[List[str], str] = None,
                         k: Optional[int] = None,
                         enable_reranking: Optional[bool] = None,
                         **generation_kwargs) -> List[Dict[str, Any]]:
        """
        Batch inference untuk multiple queries
        
        Args:
            queries: List of queries
            response_type: Type(s) of response to generate
            k: Number of contexts to retrieve per query
            enable_reranking: Whether to enable reranking
            **generation_kwargs: Additional generation parameters
            
        Returns:
            List of inference results
        """
        self.logger.info(f"Starting batch inference for {len(queries)} queries")
        
        # Create tasks untuk concurrent processing
        tasks = [
            asyncio.create_task(
                self.infer(
                    query=query,
                    response_type=response_type,
                    k=k,
                    enable_reranking=enable_reranking,
                    **generation_kwargs
                )
            )
            for query in queries
        ]
        
        # Wait for all tasks
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                self.logger.error(f"Error processing query {i}: {result}")
                processed_results.append({
                    "query": queries[i],
                    "error": str(result),
                    "success": False
                })
            else:
                result["success"] = True
                processed_results.append(result)
        
        return processed_results
    
    async def get_available_templates(self) -> List[str]:
        """Get available template types from model"""
        await self._ensure_model_loaded()
        return self.model.get_available_templates()
    
    async def preview_template(self, 
                              template_type: str, 
                              sample_query: str = "Apa itu AI?") -> str:
        """Preview template formatting"""
        await self._ensure_model_loaded()
        return self.model.preview_template(
            template_type=template_type,
            sample_question=sample_query,
            sample_context="Sample context untuk preview template..."
        )
    
    async def get_model_info(self) -> Dict[str, Any]:
        """Get model information"""
        await self._ensure_model_loaded()
        model_info = await self.model.get_model_info()
        
        return {
            "model_info": model_info,
            "inferencer_config": self.config.__dict__,
            "reranker_available": self.reranker is not None,
            "available_templates": await self.get_available_templates()
        }
    
    async def close(self):
        """Clean up resources"""
        self.logger.info("Closing Inferencer...")
        if self.model:
            await self.model.close()
        self.logger.info("Inferencer closed successfully")
    
    async def __aenter__(self):
        """Async context manager entry"""
        await self._ensure_model_loaded()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.close()