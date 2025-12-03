"""
Latency benchmarking for OpenAI embeddings + Supabase storage pipeline.

Measures end-to-end performance from text input to stored memory:
- OpenAI API embedding generation
- Supabase pgvector insert
- Supabase similarity search
- Batch size optimization
"""

import time
import asyncio
import os
from typing import List, Dict, Tuple
import numpy as np
from dataclasses import dataclass
from pathlib import Path

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    # Load from project root
    env_path = Path(__file__).parent.parent / '.env'
    load_dotenv(dotenv_path=env_path)
except ImportError:
    pass

try:
    from openai import AsyncOpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    print("⚠️  OpenAI not installed. Run: pip install openai")

try:
    import psycopg2
    from psycopg2.extras import execute_batch
    POSTGRES_AVAILABLE = True
except ImportError:
    POSTGRES_AVAILABLE = False
    print("⚠️  psycopg2 not installed. Run: pip install psycopg2-binary")


@dataclass
class LatencyMetrics:
    """Performance metrics for a benchmark run."""
    batch_size: int
    num_batches: int
    total_items: int
    
    # Timing breakdowns (seconds)
    embed_time: float
    insert_time: float
    search_time: float
    total_time: float
    
    # Derived metrics
    items_per_second: float
    avg_latency_ms: float
    embed_latency_ms: float
    insert_latency_ms: float
    search_latency_ms: float


class OpenAIEmbeddingBenchmark:
    """Benchmark OpenAI embedding generation."""
    
    def __init__(self, api_key: str = None, model: str = "text-embedding-3-small"):
        """
        Initialize OpenAI client.
        
        Args:
            api_key: OpenAI API key (defaults to OPENAI_API_KEY env var)
            model: Embedding model to use
        """
        if not OPENAI_AVAILABLE:
            raise ImportError("OpenAI package not installed")
        
        self.client = AsyncOpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))
        self.model = model
        
    async def embed_batch(self, texts: List[str]) -> Tuple[List[List[float]], float]:
        """
        Generate embeddings for a batch of texts.
        
        Args:
            texts: List of text strings (max 2048 per batch)
            
        Returns:
            (embeddings, elapsed_time)
        """
        start = time.time()
        
        response = await self.client.embeddings.create(
            model=self.model,
            input=texts
        )
        
        elapsed = time.time() - start
        embeddings = [item.embedding for item in response.data]
        
        return embeddings, elapsed
    
    async def benchmark(self, texts: List[str], batch_size: int = 100) -> LatencyMetrics:
        """
        Benchmark embedding generation with different batch sizes.
        
        Args:
            texts: List of text samples
            batch_size: Number of texts per batch
            
        Returns:
            LatencyMetrics with timing results
        """
        total_time = 0
        num_batches = (len(texts) + batch_size - 1) // batch_size
        
        print(f"\n🔬 Benchmarking OpenAI embeddings...")
        print(f"   Model: {self.model}")
        print(f"   Total texts: {len(texts)}")
        print(f"   Batch size: {batch_size}")
        print(f"   Num batches: {num_batches}")
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            _, elapsed = await self.embed_batch(batch)
            total_time += elapsed
            print(f"   Batch {i // batch_size + 1}/{num_batches}: {elapsed*1000:.1f}ms ({len(batch)} items)")
        
        return LatencyMetrics(
            batch_size=batch_size,
            num_batches=num_batches,
            total_items=len(texts),
            embed_time=total_time,
            insert_time=0,
            search_time=0,
            total_time=total_time,
            items_per_second=len(texts) / total_time,
            avg_latency_ms=(total_time / len(texts)) * 1000,
            embed_latency_ms=(total_time / len(texts)) * 1000,
            insert_latency_ms=0,
            search_latency_ms=0
        )


class PostgresBenchmark:
    """Benchmark Postgres pgvector operations."""
    
    def __init__(self, database_url: str = None):
        """
        Initialize Postgres connection.
        
        Args:
            database_url: PostgreSQL connection string (defaults to DATABASE_URL env var)
        """
        if not POSTGRES_AVAILABLE:
            raise ImportError("psycopg2 package not installed")
        
        self.database_url = database_url or os.getenv(
            "DATABASE_URL",
            "postgresql://cognigraph:cognigraph_dev@localhost:5432/cognigraph"
        )
        self.conn = psycopg2.connect(self.database_url)
        self.table_name = "memories"
    
    def check_connection(self) -> bool:
        """
        Check if database connection is working.
        
        Returns:
            True if connected successfully
        """
        try:
            with self.conn.cursor() as cur:
                cur.execute("SELECT version();")
                version = cur.fetchone()[0]
                print(f"\n✅ Connected to: {version}")
                
                # Check if pgvector is enabled
                cur.execute(
                    "SELECT EXISTS(SELECT 1 FROM pg_extension WHERE extname = 'vector');"
                )
                has_vector = cur.fetchone()[0]
                if has_vector:
                    print("✅ pgvector extension enabled")
                else:
                    print("❌ pgvector extension not found")
                    return False
                
                # Check if table exists
                cur.execute(
                    f"SELECT EXISTS(SELECT 1 FROM information_schema.tables WHERE table_name = '{self.table_name}');"
                )
                has_table = cur.fetchone()[0]
                if has_table:
                    print(f"✅ Table '{self.table_name}' exists")
                else:
                    print(f"❌ Table '{self.table_name}' not found")
                    print(f"   Run: docker-compose up -d to initialize")
                    return False
                
                return True
        except Exception as e:
            print(f"❌ Connection failed: {e}")
            return False
    
    def insert_batch(self, texts: List[str], embeddings: List[List[float]], 
                     metadata: List[Dict] = None) -> float:
        """
        Insert batch of embeddings into Postgres.
        
        Args:
            texts: List of text strings
            embeddings: List of embedding vectors
            metadata: Optional metadata for each item
            
        Returns:
            Elapsed time in seconds
        """
        start = time.time()
        
        import json
        
        records = [
            (
                text,
                embedding,
                json.dumps(meta or {})
            )
            for text, embedding, meta in zip(
                texts, 
                embeddings, 
                metadata or [{}] * len(texts)
            )
        ]
        
        with self.conn.cursor() as cur:
            execute_batch(
                cur,
                f"INSERT INTO {self.table_name} (text, embedding, metadata) VALUES (%s, %s, %s)",
                records
            )
            self.conn.commit()
        
        return time.time() - start
    
    def search_similar(self, query_embedding: List[float], limit: int = 5, 
                       threshold: float = 0.7) -> Tuple[List[Dict], float]:
        """
        Search for similar embeddings using cosine distance.
        
        Args:
            query_embedding: Query vector
            limit: Number of results to return
            threshold: Minimum similarity threshold (0-1)
            
        Returns:
            (results, elapsed_time)
        """
        start = time.time()
        
        with self.conn.cursor() as cur:
            # Use the match_memories function from init.sql
            cur.execute(
                "SELECT * FROM match_memories(%s::vector, %s, %s)",
                (query_embedding, threshold, limit)
            )
            
            results = []
            for row in cur.fetchall():
                results.append({
                    'id': row[0],
                    'text': row[1],
                    'similarity': row[2],
                    'metadata': row[3]
                })
        
        elapsed = time.time() - start
        return results, elapsed
    
    def close(self):
        """Close database connection."""
        self.conn.close()


def generate_sample_texts(num_samples: int = 100) -> List[str]:
    """Generate sample text for benchmarking."""
    templates = [
        "Machine learning enables computers to learn from data without explicit programming.",
        "The gradient descent algorithm optimizes neural network parameters iteratively.",
        "Natural language processing helps computers understand human language.",
        "Transformer models revolutionized the field of deep learning in 2017.",
        "Vector embeddings capture semantic meaning in high-dimensional space.",
        "Knowledge graphs represent relationships between entities and concepts.",
        "Cognitive architectures model human memory and reasoning processes.",
        "Reinforcement learning trains agents through trial and error feedback.",
        "Attention mechanisms allow models to focus on relevant information.",
        "Transfer learning applies knowledge from one domain to another."
    ]
    
    return [
        f"{templates[i % len(templates)]} Example {i}." 
        for i in range(num_samples)
    ]


async def benchmark_end_to_end(
    num_samples: int = 100,
    batch_size: int = 50,
    openai_model: str = "text-embedding-3-small",
    test_search: bool = True
):
    """
    Run end-to-end latency benchmark.
    
    Args:
        num_samples: Number of text samples to process
        batch_size: Batch size for OpenAI API
        openai_model: OpenAI embedding model
        test_search: Whether to test similarity search
    """
    print("="*70)
    print("LATENCY BENCHMARK: OpenAI + Local Postgres Pipeline")
    print("="*70)
    
    # Check environment
    if not OPENAI_AVAILABLE or not POSTGRES_AVAILABLE:
        print("\n❌ Missing dependencies. Install with:")
        print("   pip install openai psycopg2-binary python-dotenv")
        return
    
    if not os.getenv("OPENAI_API_KEY"):
        print("\n❌ OPENAI_API_KEY environment variable not set")
        print("   Add to .env file: OPENAI_API_KEY=your-key")
        return
    
    # Check Postgres connection
    print("\n🔌 Checking Postgres connection...")
    pg_bench = PostgresBenchmark()
    if not pg_bench.check_connection():
        print("\n❌ Database not ready. Start with: docker-compose up -d")
        return
    
    # Generate sample data
    texts = generate_sample_texts(num_samples)
    print(f"\n📝 Generated {len(texts)} sample texts")
    
    # Benchmark OpenAI embeddings
    print("\n" + "="*70)
    print("PHASE 1: OpenAI Embedding Generation")
    print("="*70)
    
    openai_bench = OpenAIEmbeddingBenchmark(model=openai_model)
    
    all_embeddings = []
    embed_total_time = 0
    
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        embeddings, elapsed = await openai_bench.embed_batch(batch)
        all_embeddings.extend(embeddings)
        embed_total_time += elapsed
        print(f"   Batch {i // batch_size + 1}: {elapsed*1000:.1f}ms ({len(batch)} texts)")
    
    print(f"\n✅ OpenAI Results:")
    print(f"   Total time: {embed_total_time:.2f}s")
    print(f"   Throughput: {len(texts) / embed_total_time:.1f} texts/sec")
    print(f"   Avg latency: {(embed_total_time / len(texts)) * 1000:.1f}ms per text")
    
    # Benchmark Postgres insert
    print("\n" + "="*70)
    print("PHASE 2: Postgres Batch Insert")
    print("="*70)
    
    insert_time = pg_bench.insert_batch(texts, all_embeddings)
    
    print(f"\n✅ Insert Results:")
    print(f"   Total time: {insert_time:.2f}s")
    print(f"   Throughput: {len(texts) / insert_time:.1f} inserts/sec")
    print(f"   Avg latency: {(insert_time / len(texts)) * 1000:.1f}ms per insert")
    
    # Benchmark similarity search
    if test_search and len(all_embeddings) > 0:
        print("\n" + "="*70)
        print("PHASE 3: Similarity Search")
        print("="*70)
        
        # Use first embedding as query
        query_embedding = all_embeddings[0]
        
        search_times = []
        for i in range(10):  # Run 10 searches
            results, elapsed = pg_bench.search_similar(query_embedding, limit=5)
            search_times.append(elapsed)
            if i == 0:
                print(f"   Found {len(results)} similar memories")
                if results:
                    print(f"   Top result similarity: {results[0]['similarity']:.3f}")
        
        avg_search_time = np.mean(search_times)
        
        print(f"\n✅ Search Results (10 queries):")
        print(f"   Avg time: {avg_search_time*1000:.1f}ms per search")
        print(f"   Min time: {min(search_times)*1000:.1f}ms")
        print(f"   Max time: {max(search_times)*1000:.1f}ms")
    
    pg_bench.close()
    
    print("\n" + "="*70)
    print("END-TO-END LATENCY SUMMARY")
    print("="*70)
    print(f"Per memory (batched):")
    print(f"  - Text chunking: ~1ms")
    print(f"  - OpenAI embedding: ~{(embed_total_time / len(texts)) * 1000:.0f}ms (amortized)")
    print(f"  - Postgres insert: ~{(insert_time / len(texts)) * 1000:.0f}ms (amortized)")
    if test_search:
        print(f"  - Similarity search: ~{avg_search_time * 1000:.0f}ms")
    print(f"  - Node allocation: ~1ms")
    total_avg = ((embed_total_time + insert_time) / len(texts)) * 1000 + 2
    print(f"\n  Total (without search): ~{total_avg:.0f}ms per memory")
    if test_search:
        print(f"  Total (with search): ~{total_avg + avg_search_time * 1000:.0f}ms per memory")
    print("="*70)


async def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Benchmark OpenAI + Supabase latency'
    )
    parser.add_argument('--samples', '-n', type=int, default=100,
                       help='Number of samples (default: 100)')
    parser.add_argument('--batch-size', '-b', type=int, default=50,
                       help='Batch size for OpenAI (default: 50)')
    parser.add_argument('--model', '-m', type=str, default='text-embedding-3-small',
                       choices=['text-embedding-3-small', 'text-embedding-3-large'],
                       help='OpenAI model (default: text-embedding-3-small)')
    parser.add_argument('--no-search', action='store_true',
                       help='Skip similarity search benchmarking')
    
    args = parser.parse_args()
    
    await benchmark_end_to_end(
        num_samples=args.samples,
        batch_size=args.batch_size,
        openai_model=args.model,
        test_search=not args.no_search
    )


if __name__ == '__main__':
    asyncio.run(main())
