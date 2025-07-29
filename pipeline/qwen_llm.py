import torch
import asyncio
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig, TextIteratorStreamer
from typing import Optional, Dict, Any, List, Union, Callable, Awaitable, AsyncGenerator
import logging
from dataclasses import dataclass
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from threading import Thread
from rag.prompt_tuner.chat_template import RAG_TEMPLATES
from rag.retriever.retriever_types import RetrievalResult
from langchain_core.documents import Document

@dataclass
class QwenConfig:
    """Konfigurasi untuk model Qwen 0.5B"""
    model_name: str = "Qwen/Qwen2.5-0.5B-Instruct"
    device: str = "cuda"
    torch_dtype: torch.dtype = torch.float16
    max_length: int = 2048
    temperature: float = 0.7
    top_p: float = 0.8
    top_k: int = 50
    do_sample: bool = True
    pad_token_id: Optional[int] = None
    eos_token_id: Optional[int] = None
    # RAG-specific configs
    max_context_length: int = 1500
    context_separator: str = "\n---\n"
    instruction_template: str = "system"  # "system", "instruction", "custom"
    # Async-specific configs
    max_workers: int = 2
    generation_timeout: float = 9999999.0
    repetition_penalty: float = 1.0
    # Streaming-specific configs
    stream_timeout: float = 10  # timeout untuk stream chunk
    skip_prompt: bool = True     # skip prompt dari streaming output

class QwenLLM:
    """
    Async LLM Qwen 0.5B dengan interface yang mudah digunakan
    Termasuk prompt formatting khusus untuk RAG (Retrieval-Augmented Generation)
    Dan support untuk text streaming
    """
    
    def __init__(self, config: Optional[QwenConfig] = None, rag_templates = RAG_TEMPLATES()):
        """
        Inisialisasi QwenLLM
        
        Args:
            config: Konfigurasi model (optional, akan menggunakan default jika None)
        """
        self.config = config or QwenConfig()
        self.tokenizer = None
        self.model = None
        self.generation_config = None
        self.is_loaded = False
        self.executor = ThreadPoolExecutor(max_workers=self.config.max_workers)
        self._lock = asyncio.Lock()
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # RAG prompt templates
        self.rag_templates = rag_templates
    
    async def load_model(self) -> None:
        """Load model dan tokenizer secara async"""
        async with self._lock:
            if self.is_loaded:
                self.logger.info("Model already loaded")
                return
            
            try:
                self.logger.info(f"Loading model: {self.config.model_name}")
                
                # Load tokenizer dalam thread pool
                self.tokenizer = await asyncio.get_event_loop().run_in_executor(
                    self.executor,
                    lambda: AutoTokenizer.from_pretrained(
                        self.config.model_name,
                        trust_remote_code=True
                    )
                )
                
                # Load model dalam thread pool
                self.model = await asyncio.get_event_loop().run_in_executor(
                    self.executor,
                    lambda: AutoModelForCausalLM.from_pretrained(
                        self.config.model_name,
                        torch_dtype=self.config.torch_dtype,
                        device_map=self.config.device,
                        trust_remote_code=True
                    )
                )
                
                # Setup generation config
                self.generation_config = GenerationConfig(
                    max_length=self.config.max_length,
                    temperature=self.config.temperature,
                    top_p=self.config.top_p,
                    top_k=self.config.top_k,
                    do_sample=self.config.do_sample,
                    pad_token_id=self.config.pad_token_id or self.tokenizer.eos_token_id,
                    eos_token_id=self.config.eos_token_id or self.tokenizer.eos_token_id,
                    repetition_penalty = self.config.repetition_penalty,
                )
                
                self.is_loaded = True
                self.logger.info("Model loaded successfully!")
                
            except Exception as e:
                self.logger.error(f"Error loading model: {e}")
                raise
    
    def get_available_templates(self) -> List[str]:
        """
        Dapatkan list template yang tersedia
        
        Returns:
            List of available template names
        """
        return list(self.rag_templates.keys())
    
    def preview_template(self, template_type: str, sample_question: str = "Apa itu AI?", 
                        sample_context: str = "Artificial Intelligence adalah teknologi...") -> str:
        """
        Preview template dengan sample data
        
        Args:
            template_type: Template type to preview
            sample_question: Sample question
            sample_context: Sample context
            
        Returns:
            Preview of formatted template
        """
        if template_type not in self.rag_templates:
            return f"Template '{template_type}' tidak tersedia. Available: {self.get_available_templates()}"
        
        template_data = self.rag_templates[template_type]
        template_key = "user_template" if "user_template" in template_data else "template"
        
        return template_data[template_key].format(
            context=sample_context,
            question=sample_question
        )
    
    def _format_context(self, contexts: Union[List[str], RetrievalResult], numbering: bool = True) -> str:
        """
        Format retrieved contexts menjadi string yang coherent
        
        Args:
            contexts: List of contexts (string atau RetrievalResult objects)
            numbering: Whether to add document numbering
            
        Returns:
            Formatted context string
        """
        if not contexts:
            return ""
        
        formatted_contexts = []
        if isinstance(contexts, RetrievalResult):
                for i, ctx in enumerate(contexts.documents, 1):
                    if numbering:
                        header = f"[Dokumen {i}"
                        if contexts.scores[i - 1]:
                            header += f" (Skor: {contexts.scores[i - 1]:.3f})"
                        header += "]"
                    else:
                        header = "[Dokumen"
                        header += "]"
                    formatted_contexts.append(f"{header}\n{ctx.page_content}")
        else:
            for i, ctx in enumerate(contexts, 1):
                if isinstance(ctx, str):
                    header = f"[Dokumen {i}]" if numbering else "[Dokumen]"
                    formatted_contexts.append(f"{header}\n{ctx}")
                else:
                    header = f"[Dokumen {i}]" if numbering else "[Dokumen]"
                    formatted_contexts.append(f"{header}\n{str(ctx)}")
        
        return self.config.context_separator.join(formatted_contexts)
    
    def _truncate_context(self, context: str, max_length: int) -> str:
        """
        Truncate context jika terlalu panjang
        
        Args:
            context: Context string
            max_length: Maximum length in characters
            
        Returns:
            Truncated context
        """
        if len(context) <= max_length:
            return context
        
        # Truncate dan tambahkan indicator
        truncated = context[:max_length - 50]
        return truncated + "\n\n[... Context dipotong karena terlalu panjang ...]"

    async def format_rag_prompt(self, 
                                question: str, 
                                contexts: Union[List[str], RetrievalResult],
                                template_type: Optional[str] = None,
                                custom_template: Optional[str] = None,
                                include_metadata: bool = True,
                                context_numbering: bool = True,
                                max_contexts: Optional[int] = None) -> str:
        """
        Format prompt untuk RAG dengan berbagai template options (async)
        """
        
        def _format_sync():
            # Handle RetrievalResult secara eksplisit
            if isinstance(contexts, RetrievalResult):
                docs = contexts.documents
                if max_contexts:
                    docs = docs[:max_contexts]
                processed_contexts = RetrievalResult(
                    documents=docs,
                    scores=contexts.scores[:len(docs)] if contexts.scores else [],
                    query=contexts.query,
                    retrieval_time=contexts.retrieval_time,
                    metadata=contexts.metadata
                )
            else:
                # contexts diasumsikan sebagai list biasa (list[str] atau list[Document])
                processed_contexts = contexts[:max_contexts] if max_contexts and len(contexts) > max_contexts else contexts

            # Format context menjadi string
            formatted_context = self._format_context(processed_contexts, context_numbering)

            # Truncate jika panjang melebihi batas
            formatted_context = self._truncate_context(
                formatted_context, 
                self.config.max_context_length
            )

            # Tambah metadata jika diizinkan dan konteks adalah RetrievalResult
            if include_metadata and isinstance(processed_contexts, RetrievalResult):
                metadata_info = []
                for i, doc in enumerate(processed_contexts.documents, 1):
                    if hasattr(doc, "metadata") and doc.metadata:
                        metadata_info.append(f"Dokumen {i}: {doc.metadata}")
                if metadata_info:
                    formatted_context += f"\n\n[Metadata]\n" + "\n".join(metadata_info)

            return formatted_context

        # Jalankan _format_sync di thread pool
        formatted_context = await asyncio.get_event_loop().run_in_executor(
            self.executor, _format_sync
        )

        # Tentukan template yang akan dipakai
        template_type = template_type or self.config.instruction_template

        # Gunakan custom template jika disediakan
        if custom_template:
            return custom_template.format(
                context=formatted_context,
                question=question
            )
        elif template_type in self.rag_templates:
            template_data = self.rag_templates[template_type]
            template_key = "user_template" if "user_template" in template_data else "template"
            return template_data[template_key].format(
                context=formatted_context,
                question=question
            )
        else:
            # Fallback default template
            return f"Konteks: {formatted_context}\n\nPertanyaan: {question}\n\nJawaban:"

    async def generate_stream(self, 
                             prompt: str, 
                             max_new_tokens: Optional[int] = None,
                             temperature: Optional[float] = None,
                             top_p: Optional[float] = None,
                             **kwargs) -> AsyncGenerator[str, None]:
        """
        Generate text dari prompt secara streaming async
        
        Args:
            prompt: Input text prompt
            max_new_tokens: Maximum token baru yang akan di-generate
            temperature: Temperature untuk generation (override config)
            top_p: Top-p untuk generation (override config)
            **kwargs: Parameter tambahan untuk generation
            
        Yields:
            Generated text chunks
        """
        await self._check_model_loaded()
        
        # Setup streamer
        streamer = TextIteratorStreamer(
            self.tokenizer, 
            timeout=self.config.stream_timeout,
            skip_prompt=self.config.skip_prompt,
            skip_special_tokens=True
        )
        
        def _generate_sync():
            try:
                # Tokenize input
                inputs = self.tokenizer(prompt, return_tensors="pt")
                
                # Override generation config jika diperlukan
                gen_config = self.generation_config
                if any([max_new_tokens, temperature, top_p]):
                    gen_config = GenerationConfig(
                        max_new_tokens=max_new_tokens or self.config.max_length,
                        temperature=temperature or self.config.temperature,
                        top_p=top_p or self.config.top_p,
                        top_k=self.config.top_k,
                        do_sample=self.config.do_sample,
                        pad_token_id=self.config.pad_token_id or self.tokenizer.eos_token_id,
                        eos_token_id=self.config.eos_token_id or self.tokenizer.eos_token_id,
                        repetition_penalty=self.config.repetition_penalty,
                        **kwargs
                    )
                
                # Move to GPU
                self.model.to("cuda")
                input_ids = inputs.input_ids.to("cuda")
                
                # Generate dalam thread terpisah
                generation_kwargs = {
                    "input_ids": input_ids,
                    "generation_config": gen_config,
                    "streamer": streamer,
                    **kwargs
                }
                
                thread = Thread(target=self.model.generate, kwargs=generation_kwargs)
                thread.start()
                
                return thread
                
            except Exception as e:
                self.logger.error(f"Error during stream generation setup: {e}")
                raise
        
        # Setup generation thread
        generation_thread = await asyncio.get_event_loop().run_in_executor(
            self.executor, _generate_sync
        )
        
        try:
            # Stream tokens
            for token in streamer:
                if token:  # Skip empty tokens
                    yield token
                    
            # Wait for generation thread to finish
            await asyncio.get_event_loop().run_in_executor(
                self.executor, generation_thread.join
            )
            
        except Exception as e:
            self.logger.error(f"Error during streaming: {e}")
            # Make sure thread is cleaned up
            if generation_thread.is_alive():
                generation_thread.join(timeout=1.0)
            raise

    async def rag_generate_stream(self,
                                 question: str,
                                 contexts: Union[List[str], RetrievalResult],
                                 template_type: Optional[str] = None,
                                 max_new_tokens: Optional[int] = None,
                                 temperature: Optional[float] = None,
                                 **kwargs) -> AsyncGenerator[str, None]:
        """
        Generate jawaban untuk RAG secara streaming async
        
        Args:
            question: User question
            contexts: List of retrieved contexts
            template_type: Template type untuk formatting
            max_new_tokens: Maximum token baru yang akan di-generate
            temperature: Temperature untuk generation
            **kwargs: Parameter tambahan untuk generation
            
        Yields:
            Generated answer chunks
        """
        await self._check_model_loaded()
        
        # Format prompt
        prompt = await self.format_rag_prompt(question, contexts, template_type)
        
        # Generate dengan temperature yang lebih rendah untuk RAG (lebih faktual)
        temp = temperature if temperature is not None else 0.3
        
        async for chunk in self.generate_stream(
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            temperature=temp,
            **kwargs
        ):
            yield chunk

    async def chat_stream(self, 
                         messages: List[Dict[str, str]], 
                         max_new_tokens: Optional[int] = None,
                         **kwargs) -> AsyncGenerator[str, None]:
        """
        Chat dengan format conversation secara streaming async
        
        Args:
            messages: List of messages dengan format [{"role": "user", "content": "..."}]
            max_new_tokens: Maximum token baru yang akan di-generate
            **kwargs: Parameter tambahan untuk generation
            
        Yields:
            Response text chunks
        """
        await self._check_model_loaded()
        
        def _format_chat():
            try:
                # Format messages untuk chat
                formatted_prompt = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
                return formatted_prompt
                
            except Exception as e:
                self.logger.error(f"Error during chat formatting: {e}")
                raise
        
        # Format chat template dalam thread pool
        formatted_prompt = await asyncio.get_event_loop().run_in_executor(
            self.executor, _format_chat
        )
        
        async for chunk in self.generate_stream(
            formatted_prompt, 
            max_new_tokens=max_new_tokens,
            **kwargs
        ):
            yield chunk

    async def rag_chat_stream(self,
                             messages: List[Dict[str, str]],
                             contexts: Union[List[str], RetrievalResult],
                             template_type: Optional[str] = None,
                             max_new_tokens: Optional[int] = None,
                             **kwargs) -> AsyncGenerator[str, None]:
        """
        RAG Chat dengan format conversation secara streaming async
        
        Args:
            messages: List of messages dengan format [{"role": "user", "content": "..."}]
            contexts: List of retrieved contexts
            template_type: Template type untuk formatting
            max_new_tokens: Maximum token baru yang akan di-generate
            **kwargs: Parameter tambahan untuk generation
            
        Yields:
            Response text chunks
        """
        await self._check_model_loaded()
        
        # Ambil last user message sebagai question
        user_messages = [msg for msg in messages if msg.get("role") == "user"]
        if not user_messages:
            raise ValueError("No user message found in conversation")
        
        last_question = user_messages[-1]["content"]
        
        # Generate RAG response secara streaming
        async for chunk in self.rag_generate_stream(
            question=last_question,
            contexts=contexts,
            template_type=template_type,
            max_new_tokens=max_new_tokens,
            **kwargs
        ):
            yield chunk

    # Utility method untuk collect full response dari stream
    async def collect_stream(self, stream_generator: AsyncGenerator[str, None]) -> str:
        """
        Collect semua chunks dari stream generator menjadi full text
        
        Args:
            stream_generator: AsyncGenerator yang menghasilkan text chunks
            
        Returns:
            Complete generated text
        """
        chunks = []
        async for chunk in stream_generator:
            chunks.append(chunk)
        return "".join(chunks)
    
    async def multi_template_generate(self,
                                    question: str,
                                    contexts: Union[List[str], RetrievalResult],
                                    template_types: List[str],
                                    max_new_tokens: Optional[int] = None,
                                    **kwargs) -> Dict[str, str]:
        """
        Generate jawaban menggunakan multiple templates secara concurrent
        
        Args:
            question: User question
            contexts: List of retrieved contexts
            template_types: List of template types to use
            max_new_tokens: Maximum token baru yang akan di-generate
            **kwargs: Parameter tambahan untuk generation
            
        Returns:
            Dictionary dengan template_type sebagai key dan response sebagai value
        """
        await self._check_model_loaded()
        
        # Create tasks untuk concurrent generation
        tasks = []
        for template_type in template_types:
            task = asyncio.create_task(
                self._generate_single_template(
                    question, contexts, template_type, max_new_tokens, **kwargs
                )
            )
            tasks.append((template_type, task))
        
        # Wait for all tasks
        results = {}
        for template_type, task in tasks:
            try:
                response = await task
                results[template_type] = response
            except Exception as e:
                self.logger.error(f"Error generating with template {template_type}: {e}")
                results[template_type] = f"Error: {str(e)}"
        
        return results
    
    async def _generate_single_template(self,
                                      question: str,
                                      contexts: Union[List[str], RetrievalResult],
                                      template_type: str,
                                      max_new_tokens: Optional[int] = None,
                                      **kwargs) -> str:
        """Helper method untuk single template generation"""
        return await self.rag_generate(
            question=question,
            contexts=contexts,
            template_type=template_type,
            max_new_tokens=max_new_tokens,
            **kwargs
        )
    
    async def rag_generate(self,
                          question: str,
                          contexts: Union[List[str], RetrievalResult],
                          template_type: Optional[str] = None,
                          max_new_tokens: Optional[int] = None,
                          temperature: Optional[float] = None,
                          **kwargs) -> str:
        """
        Generate jawaban untuk RAG secara async
        
        Args:
            question: User question
            contexts: List of retrieved contexts
            template_type: Template type untuk formatting
            max_new_tokens: Maximum token baru yang akan di-generate
            temperature: Temperature untuk generation
            **kwargs: Parameter tambahan untuk generation
            
        Returns:
            Generated answer
        """
        await self._check_model_loaded()
        
        # Format prompt
        prompt = await self.format_rag_prompt(question, contexts, template_type)
        
        # Generate dengan temperature yang lebih rendah untuk RAG (lebih faktual)
        temp = temperature if temperature is not None else 0.3
        
        return await self.generate(
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            temperature=temp,
            **kwargs
        )
    
    async def rag_chat(self,
                      messages: List[Dict[str, str]],
                      contexts: Union[List[str], RetrievalResult],
                      template_type: Optional[str] = None,
                      max_new_tokens: Optional[int] = None,
                      **kwargs) -> str:
        """
        RAG Chat dengan format conversation secara async
        
        Args:
            messages: List of messages dengan format [{"role": "user", "content": "..."}]
            contexts: List of retrieved contexts
            template_type: Template type untuk formatting
            max_new_tokens: Maximum token baru yang akan di-generate
            **kwargs: Parameter tambahan untuk generation
            
        Returns:
            Response text
        """
        await self._check_model_loaded()
        
        # Ambil last user message sebagai question
        user_messages = [msg for msg in messages if msg.get("role") == "user"]
        if not user_messages:
            raise ValueError("No user message found in conversation")
        
        last_question = user_messages[-1]["content"]
        
        # Generate RAG response
        return await self.rag_generate(
            question=last_question,
            contexts=contexts,
            template_type=template_type,
            max_new_tokens=max_new_tokens,
            **kwargs
        )
    
    async def _check_model_loaded(self) -> None:
        """Cek apakah model sudah di-load secara async"""
        if not self.is_loaded:
            raise RuntimeError("Model belum di-load. Panggil await load_model() terlebih dahulu.")
    
    async def generate(self, 
                      prompt: str, 
                      max_new_tokens: Optional[int] = None,
                      temperature: Optional[float] = None,
                      top_p: Optional[float] = None,
                      **kwargs) -> str:
        """
        Generate text dari prompt secara async
        
        Args:
            prompt: Input text prompt
            max_new_tokens: Maximum token baru yang akan di-generate
            temperature: Temperature untuk generation (override config)
            top_p: Top-p untuk generation (override config)
            **kwargs: Parameter tambahan untuk generation
            
        Returns:
            Generated text
        """
        await self._check_model_loaded()
        
        def _generate_sync():
            try:
                # Tokenize input
                inputs = self.tokenizer(prompt, return_tensors="pt")
                
                # Override generation config jika diperlukan
                gen_config = self.generation_config
                if any([max_new_tokens, temperature, top_p]):
                    gen_config = GenerationConfig(
                        max_new_tokens=max_new_tokens or self.config.max_length,
                        temperature=temperature or self.config.temperature,
                        top_p=top_p or self.config.top_p,
                        top_k=self.config.top_k,
                        do_sample=self.config.do_sample,
                        pad_token_id=self.config.pad_token_id or self.tokenizer.eos_token_id,
                        eos_token_id=self.config.eos_token_id or self.tokenizer.eos_token_id,
                        repetition_penalty = self.config.repetition_penalty,
                        **kwargs
                    )
                
                # Generate
                with torch.no_grad():
                    
                    self.model.to("cuda")
                    input_ids = inputs.input_ids.to("cuda")

                    outputs = self.model.generate(
                        input_ids,
                        generation_config=gen_config,
                        **kwargs
                    )
                
                # Decode output
                generated_text = self.tokenizer.decode(
                    outputs[0], 
                    skip_special_tokens=True
                )
                
                # Remove input prompt dari output
                if generated_text.startswith(prompt):
                    generated_text = generated_text[len(prompt):].strip()
                
                return generated_text
                
            except Exception as e:
                self.logger.error(f"Error during generation: {e}")
                raise
        
        # Run generation in thread pool dengan timeout
        try:
            result = await asyncio.wait_for(
                asyncio.get_event_loop().run_in_executor(self.executor, _generate_sync),
                timeout=self.config.generation_timeout
            )
            return result
        except asyncio.TimeoutError:
            self.logger.error(f"Generation timeout after {self.config.generation_timeout} seconds")
            raise TimeoutError(f"Generation timeout after {self.config.generation_timeout} seconds")
    
    async def chat(self, 
                  messages: List[Dict[str, str]], 
                  max_new_tokens: Optional[int] = None,
                  **kwargs) -> str:
        """
        Chat dengan format conversation secara async
        
        Args:
            messages: List of messages dengan format [{"role": "user", "content": "..."}]
            max_new_tokens: Maximum token baru yang akan di-generate
            **kwargs: Parameter tambahan untuk generation
            
        Returns:
            Response text
        """
        await self._check_model_loaded()
        
        def _format_chat():
            try:
                # Format messages untuk chat
                formatted_prompt = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
                return formatted_prompt
                
            except Exception as e:
                self.logger.error(f"Error during chat formatting: {e}")
                raise
        
        # Format chat template dalam thread pool
        formatted_prompt = await asyncio.get_event_loop().run_in_executor(
            self.executor, _format_chat
        )
        
        return await self.generate(
            formatted_prompt, 
            max_new_tokens=max_new_tokens,
            **kwargs
        )
    
    async def update_config(self, **kwargs) -> None:
        """
        Update konfigurasi model secara async
        
        Args:
            **kwargs: Parameter konfigurasi yang akan diupdate
        """
        async with self._lock:
            for key, value in kwargs.items():
                if hasattr(self.config, key):
                    setattr(self.config, key, value)
                    self.logger.info(f"Updated {key} to {value}")
                else:
                    self.logger.warning(f"Unknown config parameter: {key}")
            
            # Update generation config jika model sudah loaded
            if self.is_loaded:
                self.generation_config = GenerationConfig(
                    max_length=self.config.max_length,
                    temperature=self.config.temperature,
                    top_p=self.config.top_p,
                    top_k=self.config.top_k,
                    do_sample=self.config.do_sample,
                    pad_token_id=self.config.pad_token_id or self.tokenizer.eos_token_id,
                    eos_token_id=self.config.eos_token_id or self.tokenizer.eos_token_id,
                    repetition_penalty = self.config.repetition_penalty,

                )
    
    async def get_model_info(self) -> Dict[str, Any]:
        """
        Dapatkan informasi model secara async
        
        Returns:
            Dictionary dengan informasi model
        """
        info = {
            "model_name": self.config.model_name,
            "is_loaded": self.is_loaded,
            "config": self.config.__dict__
        }
        
        if self.is_loaded:
            # Get model info dalam thread pool
            def _get_info():
                return {
                    "vocab_size": self.tokenizer.vocab_size,
                    "model_parameters": sum(p.numel() for p in self.model.parameters()),
                    "device": str(next(self.model.parameters()).device)
                }
            
            model_info = await asyncio.get_event_loop().run_in_executor(
                self.executor, _get_info
            )
            info.update(model_info)
        
        return info
    
    async def batch_generate(self, 
                           prompts: List[str], 
                           max_new_tokens: Optional[int] = None,
                           **kwargs) -> List[str]:
        """
        Generate multiple prompts secara batch dan concurrent
        
        Args:
            prompts: List of prompts to generate
            max_new_tokens: Maximum token baru yang akan di-generate
            **kwargs: Parameter tambahan untuk generation
            
        Returns:
            List of generated texts
        """
        await self._check_model_loaded()
        
        # Create tasks untuk concurrent generation
        tasks = [
            asyncio.create_task(
                self.generate(prompt, max_new_tokens=max_new_tokens, **kwargs)
            )
            for prompt in prompts
        ]
        
        # Wait for all tasks
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                self.logger.error(f"Error generating prompt {i}: {result}")
                processed_results.append(f"Error: {str(result)}")
            else:
                processed_results.append(result)
        
        return processed_results
    
    async def close(self) -> None:
        """
        Cleanup resources secara async
        """
        self.logger.info("Closing QwenLLM...")
        
        # Shutdown executor
        self.executor.shutdown(wait=True)
        
        # Clear GPU memory
        if hasattr(self, 'model') and self.model is not None:
            del self.model
        if hasattr(self, 'tokenizer') and self.tokenizer is not None:
            del self.tokenizer
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        self.is_loaded = False
        self.logger.info("QwenLLM closed successfully")
    
    async def __aenter__(self):
        """Async context manager entry"""
        await self.load_model()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.close()