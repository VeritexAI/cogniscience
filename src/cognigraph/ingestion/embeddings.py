"""
OpenAI embedding generation with batching and error handling.

Provides async embedding generation optimized for throughput with
intelligent batching, retries, and rate limiting.
"""

import asyncio
import os
from typing import List, Tuple, Optional
from openai import AsyncOpenAI, RateLimitError, APIError
import time


class OpenAIEmbeddingGenerator:
    """
    Async OpenAI embedding generator with batching and retries.
    
    Optimized for high throughput with configurable batch sizes,
    automatic retries, and rate limit handling.
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "text-embedding-3-small",
        max_batch_size: int = 2048,
        max_retries: int = 3,
        retry_delay: float = 1.0
    ):
        """
        Initialize embedding generator.
        
        Args:
            api_key: OpenAI API key (defaults to OPENAI_API_KEY env var)
            model: Embedding model (text-embedding-3-small or text-embedding-3-large)
            max_batch_size: Maximum texts per API call (OpenAI limit: 2048)
            max_retries: Number of retries on failure
            retry_delay: Initial retry delay in seconds (exponential backoff)
        """
        self.client = AsyncOpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))
        self.model = model
        self.max_batch_size = min(max_batch_size, 2048)  # OpenAI hard limit
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        
        # Model dimensions
        self.dimensions = {
            "text-embedding-3-small": 1536,
            "text-embedding-3-large": 3072,
            "text-embedding-ada-002": 1536
        }
    
    def get_dimension(self) -> int:
        """Get embedding dimension for current model."""
        return self.dimensions.get(self.model, 1536)
    
    async def embed_single(self, text: str, retry_count: int = 0) -> List[float]:
        """
        Generate embedding for single text with retries.
        
        Args:
            text: Input text
            retry_count: Current retry attempt
            
        Returns:
            Embedding vector
            
        Raises:
            Exception: If all retries exhausted
        """
        try:
            response = await self.client.embeddings.create(
                model=self.model,
                input=[text]
            )
            return response.data[0].embedding
        
        except RateLimitError as e:
            if retry_count < self.max_retries:
                delay = self.retry_delay * (2 ** retry_count)  # Exponential backoff
                await asyncio.sleep(delay)
                return await self.embed_single(text, retry_count + 1)
            raise
        
        except APIError as e:
            if retry_count < self.max_retries:
                delay = self.retry_delay * (2 ** retry_count)
                await asyncio.sleep(delay)
                return await self.embed_single(text, retry_count + 1)
            raise
    
    async def embed_batch(
        self,
        texts: List[str],
        retry_count: int = 0
    ) -> Tuple[List[List[float]], float]:
        """
        Generate embeddings for batch of texts with retries.
        
        Args:
            texts: List of input texts (max 2048)
            retry_count: Current retry attempt
            
        Returns:
            Tuple of (embeddings, elapsed_time)
            
        Raises:
            Exception: If all retries exhausted
        """
        if len(texts) > self.max_batch_size:
            raise ValueError(
                f"Batch size {len(texts)} exceeds maximum {self.max_batch_size}"
            )
        
        start = time.time()
        
        try:
            response = await self.client.embeddings.create(
                model=self.model,
                input=texts
            )
            elapsed = time.time() - start
            embeddings = [item.embedding for item in response.data]
            return embeddings, elapsed
        
        except RateLimitError as e:
            if retry_count < self.max_retries:
                delay = self.retry_delay * (2 ** retry_count)
                await asyncio.sleep(delay)
                return await self.embed_batch(texts, retry_count + 1)
            raise
        
        except APIError as e:
            if retry_count < self.max_retries:
                delay = self.retry_delay * (2 ** retry_count)
                await asyncio.sleep(delay)
                return await self.embed_batch(texts, retry_count + 1)
            raise
    
    async def embed_many(
        self,
        texts: List[str],
        batch_size: Optional[int] = None,
        show_progress: bool = False
    ) -> Tuple[List[List[float]], float]:
        """
        Generate embeddings for many texts with automatic batching.
        
        Args:
            texts: List of input texts
            batch_size: Batch size (defaults to max_batch_size)
            show_progress: Print progress updates
            
        Returns:
            Tuple of (embeddings, total_elapsed_time)
        """
        if batch_size is None:
            batch_size = self.max_batch_size
        
        batch_size = min(batch_size, self.max_batch_size)
        
        all_embeddings = []
        total_time = 0
        num_batches = (len(texts) + batch_size - 1) // batch_size
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            batch_num = i // batch_size + 1
            
            if show_progress:
                print(f"   Processing batch {batch_num}/{num_batches} ({len(batch)} texts)...", end=" ")
            
            embeddings, elapsed = await self.embed_batch(batch)
            all_embeddings.extend(embeddings)
            total_time += elapsed
            
            if show_progress:
                print(f"{elapsed*1000:.0f}ms")
        
        return all_embeddings, total_time
    
    async def embed_many_parallel(
        self,
        texts: List[str],
        batch_size: Optional[int] = None,
        max_concurrent: int = 5
    ) -> Tuple[List[List[float]], float]:
        """
        Generate embeddings with parallel batches (advanced).
        
        Args:
            texts: List of input texts
            batch_size: Batch size per request
            max_concurrent: Maximum concurrent API calls
            
        Returns:
            Tuple of (embeddings, total_elapsed_time)
        """
        if batch_size is None:
            batch_size = self.max_batch_size // max_concurrent
        
        batch_size = min(batch_size, self.max_batch_size)
        
        # Split into batches
        batches = [
            texts[i:i + batch_size]
            for i in range(0, len(texts), batch_size)
        ]
        
        start = time.time()
        
        # Process batches with concurrency limit
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def process_batch(batch):
            async with semaphore:
                embeddings, _ = await self.embed_batch(batch)
                return embeddings
        
        # Execute all batches
        results = await asyncio.gather(*[process_batch(batch) for batch in batches])
        
        elapsed = time.time() - start
        
        # Flatten results
        all_embeddings = [emb for batch_result in results for emb in batch_result]
        
        return all_embeddings, elapsed


def create_embedding_generator(
    model: str = "text-embedding-3-small",
    api_key: Optional[str] = None
) -> OpenAIEmbeddingGenerator:
    """
    Factory function to create embedding generator.
    
    Args:
        model: OpenAI embedding model
        api_key: Optional API key (defaults to env var)
        
    Returns:
        Configured OpenAIEmbeddingGenerator instance
    """
    return OpenAIEmbeddingGenerator(api_key=api_key, model=model)
