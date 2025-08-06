from ddgs import DDGS
from langchain_community.document_loaders import AsyncChromiumLoader
from langchain_community.document_transformers import BeautifulSoupTransformer
import re
import logging
from typing import AsyncGenerator, List

class DuckDuckGoSearch:
    def __init__(self, html_loader: AsyncChromiumLoader = None, html_parser = None):
        # Initialize dengan default values jika tidak diberikan
        self.html_loader = html_loader or AsyncChromiumLoader([])
        self.html_parser = html_parser or BeautifulSoupTransformer()
        self.logger = logging.getLogger("ddgs_logger")
    
    async def get_page(self, urls: List[str]):
        """Get page content from URLs - returns list of documents"""
        try:
            self.html_loader.urls = urls
            html = await self.html_loader.aload()  # This returns a LIST
            self.logger.info(f"search engine aload result: {len(html)} documents loaded")
            
            docs_transformed = self.html_parser.transform_documents(
                html, 
                tags_to_extract=["p"], 
                remove_unwanted_tags=["a"]
            )
            return docs_transformed  # Returns LIST of documents
            
        except Exception as e:
            self.logger.error(f"Error loading pages: {e}", exc_info=True)
            return []  # Return empty list on error
    
    def truncate(self, text: str, max_words: int = 400) -> str:
        """Truncate text to specified number of words"""
        if not text:
            return ""
        
        words = text.split()
        if len(words) <= max_words:
            return text
            
        truncated = " ".join(words[:max_words])
        return truncated + "..." if len(words) > max_words else truncated
    
    async def search(self, query: str, max_results: int = 5) -> AsyncGenerator[str, None]:
        """
        Search and yield page contents one by one
        
        FIXED VERSION: Properly handle async iteration
        """
        try:
            self.logger.info(f"Searching for: {query} (max_results: {max_results})")
            
            # Step 1: Get search results from DDGS (regular iterator)
            results = DDGS().text(query, max_results=max_results)
            urls = []
            
            # Step 2: Extract URLs using regular for loop (NOT async for)
            for result in results:  # ← FIXED: Regular for loop
                url = result.get('href')
                if url:
                    urls.append(url)
            
            self.logger.info(f"Found {len(urls)} URLs to process")
            
            if not urls:
                self.logger.warning("No URLs found from search results")
                return
            
            # Step 3: Get page content (await the coroutine first)
            docs = await self.get_page(urls)  # ← FIXED: Await first, get list
            
            # Step 4: Process documents using regular for loop (NOT async for)
            for doc in docs:  # ← FIXED: Regular for loop on list
                try:
                    if hasattr(doc, 'page_content') and doc.page_content:
                        # Clean up text
                        page_text = re.sub(r"\n\n+", "\n", doc.page_content)
                        page_text = page_text.strip()
                        
                        if page_text:  # Only yield if there's actual content
                            text = self.truncate(page_text)
                            yield text  # Yield makes this an async generator
                        
                except Exception as e:
                    self.logger.error(f"Error processing document: {e}")
                    continue
                    
        except Exception as e:
            self.logger.error(f"Error in search method: {e}", exc_info=True)
            # Don't re-raise, just log and return (generator will be empty)
    
    async def search_with_metadata(self, query: str, max_results: int = 5) -> AsyncGenerator[dict, None]:
        """
        Alternative method that yields dictionaries with metadata
        """
        try:
            results = DDGS().text(query, max_results=max_results)
            urls_and_titles = []
            
            # Collect URLs and titles
            for result in results:
                url = result.get('href')
                title = result.get('title', 'No title')
                if url:
                    urls_and_titles.append({'url': url, 'title': title})
            
            if not urls_and_titles:
                return
            
            # Get page content
            urls = [item['url'] for item in urls_and_titles]
            docs = await self.get_page(urls)
            
            # Process and yield with metadata
            for i, doc in enumerate(docs):
                try:
                    if hasattr(doc, 'page_content') and doc.page_content:
                        page_text = re.sub(r"\n\n+", "\n", doc.page_content)
                        page_text = page_text.strip()
                        
                        if page_text:
                            text = self.truncate(page_text)
                            
                            # Get metadata if available
                            metadata = {}
                            if i < len(urls_and_titles):
                                metadata = urls_and_titles[i]
                            
                            yield {
                                'content': text,
                                'url': metadata.get('url', 'Unknown'),
                                'title': metadata.get('title', 'No title'),
                                'word_count': len(text.split())
                            }
                            
                except Exception as e:
                    self.logger.error(f"Error processing document {i}: {e}")
                    continue
                    
        except Exception as e:
            self.logger.error(f"Error in search_with_metadata: {e}", exc_info=True)