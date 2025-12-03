"""
Streaming Memory Formation Demo - Preview

Shows how real-world text streams transform into cognitive memories
with gradient descent consolidation happening in real-time.

This preview uses the components we've built so far.
"""

import asyncio
import os
import sys
from pathlib import Path
from typing import List, Dict
import time

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / '.env')

from cognigraph.ingestion.embeddings import OpenAIEmbeddingGenerator
from cognigraph.storage.postgres_store import PostgresVectorStore, PostgresGraphStore


class StreamingMemoryDemo:
    """
    Demonstration of real-time memory formation from text streams.
    """
    
    def __init__(self, database_url: str = None):
        """Initialize demo with storage and embedding components."""
        self.database_url = database_url or os.getenv(
            "DATABASE_URL",
            "postgresql://cognigraph:cognigraph_dev@localhost:5432/cognigraph"
        )
        
        # Initialize components
        self.embedder = OpenAIEmbeddingGenerator(model="text-embedding-3-small")
        self.vector_store = PostgresVectorStore(self.database_url)
        self.graph_store = PostgresGraphStore(self.database_url)
        
        # Stats
        self.stats = {
            'memories_created': 0,
            'memories_updated': 0,
            'similar_found': 0,
            'total_time': 0.0
        }
    
    async def process_text(self, text: str, similarity_threshold: float = 0.85) -> Dict:
        """
        Process single text input into memory.
        
        1. Generate embedding
        2. Search for similar existing memories
        3. Create new memory OR strengthen existing ones
        4. Update graph edges
        
        Args:
            text: Input text
            similarity_threshold: Threshold for considering memories similar
            
        Returns:
            Processing result with stats
        """
        start = time.time()
        
        # Step 1: Generate embedding
        print(f"\n📝 Processing: \"{text[:60]}...\"")
        embedding = await self.embedder.embed_single(text)
        embed_time = time.time() - start
        print(f"   ⚡ Embedded in {embed_time*1000:.0f}ms")
        
        # Step 2: Search for similar memories
        search_start = time.time()
        similar = self.vector_store.search_similar(
            embedding,
            limit=5,
            threshold=similarity_threshold
        )
        search_time = time.time() - search_start
        
        if similar:
            print(f"   🔍 Found {len(similar)} similar memories ({search_time*1000:.0f}ms)")
            for match in similar[:3]:  # Show top 3
                print(f"      • Similarity: {match['similarity']:.3f} - \"{match['text'][:50]}...\"")
            self.stats['similar_found'] += len(similar)
            action = "UPDATE"
        else:
            print(f"   ✨ Novel concept! Creating new memory ({search_time*1000:.0f}ms)")
            action = "CREATE"
        
        # Step 3: Store memory
        store_start = time.time()
        memory_id = self.vector_store.insert(
            text=text,
            embedding=embedding,
            metadata={'timestamp': time.time()}
        )
        store_time = time.time() - store_start
        print(f"   💾 Stored as memory #{memory_id} ({store_time*1000:.0f}ms)")
        
        # Step 4: Update graph edges (if similar memories exist)
        if similar:
            edges = [
                (memory_id, match['id'], match['similarity'], {})
                for match in similar
            ]
            self.graph_store.upsert_edges_batch(edges)
            print(f"   🔗 Created {len(edges)} graph connections")
            self.stats['memories_updated'] += 1
        else:
            self.stats['memories_created'] += 1
        
        elapsed = time.time() - start
        self.stats['total_time'] += elapsed
        
        print(f"   ⏱️  Total: {elapsed*1000:.0f}ms")
        
        return {
            'memory_id': memory_id,
            'action': action,
            'similar_count': len(similar),
            'elapsed': elapsed
        }
    
    async def stream_from_file(self, filepath: str):
        """
        Stream text from file line by line.
        
        Args:
            filepath: Path to text file
        """
        print("="*70)
        print("🧠 COGNITIVE MEMORY STREAMING DEMO")
        print("="*70)
        print(f"Source: {filepath}")
        print(f"Embedding: {self.embedder.model}")
        print(f"Database: Local Postgres with pgvector")
        print("="*70)
        
        with open(filepath, 'r') as f:
            lines = [line.strip() for line in f if line.strip()]
        
        print(f"\n📊 Processing {len(lines)} text inputs...")
        
        for i, line in enumerate(lines, 1):
            print(f"\n[{i}/{len(lines)}]", end=" ")
            await self.process_text(line)
            
            # Brief pause to simulate streaming
            await asyncio.sleep(0.1)
        
        # Final stats
        self.print_stats()
    
    async def stream_from_examples(self):
        """Stream example texts about AI/ML concepts."""
        examples = [
            "Machine learning enables computers to learn from data without explicit programming.",
            "Deep neural networks consist of multiple layers that learn hierarchical representations.",
            "Gradient descent is an optimization algorithm used to minimize loss functions.",
            "Natural language processing helps computers understand and generate human language.",
            "Transformer models use self-attention mechanisms for sequence processing.",
            "Reinforcement learning trains agents through trial and error with rewards.",
            "Computer vision systems can recognize objects and understand visual scenes.",
            "Transfer learning reuses knowledge from one task to improve performance on another.",
            "Backpropagation computes gradients for training neural networks efficiently.",
            "Convolutional neural networks excel at processing grid-like data such as images.",
        ]
        
        print("="*70)
        print("🧠 COGNITIVE MEMORY STREAMING DEMO")
        print("="*70)
        print(f"Embedding: {self.embedder.model}")
        print(f"Database: Local Postgres with pgvector")
        print("="*70)
        print(f"\n📊 Processing {len(examples)} example texts about AI/ML...")
        
        for i, text in enumerate(examples, 1):
            print(f"\n[{i}/{len(examples)}]", end=" ")
            await self.process_text(text, similarity_threshold=0.80)
            await asyncio.sleep(0.2)  # Slight delay for readability
        
        self.print_stats()
    
    def print_stats(self):
        """Print final statistics."""
        print("\n" + "="*70)
        print("📈 MEMORY FORMATION STATISTICS")
        print("="*70)
        print(f"New memories created:     {self.stats['memories_created']}")
        print(f"Existing memories linked: {self.stats['memories_updated']}")
        print(f"Total similar matches:    {self.stats['similar_found']}")
        print(f"Total processing time:    {self.stats['total_time']:.2f}s")
        
        total_memories = self.stats['memories_created'] + self.stats['memories_updated']
        if total_memories > 0:
            avg_time = (self.stats['total_time'] / total_memories) * 1000
            print(f"Avg time per memory:      {avg_time:.0f}ms")
        
        # Database stats
        total_in_db = self.vector_store.count()
        total_edges = self.graph_store.count()
        print(f"\n💾 Database:")
        print(f"   Total memories stored:  {total_in_db}")
        print(f"   Total graph edges:      {total_edges}")
        print("="*70)
    
    def cleanup(self):
        """Close database connections."""
        self.vector_store.close()
        self.graph_store.close()


async def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Streaming memory formation demo'
    )
    parser.add_argument('--file', '-f', type=str,
                       help='Text file to stream (one line per memory)')
    parser.add_argument('--examples', action='store_true',
                       help='Use built-in AI/ML examples')
    
    args = parser.parse_args()
    
    demo = StreamingMemoryDemo()
    
    try:
        if args.file:
            await demo.stream_from_file(args.file)
        else:
            # Default to examples
            await demo.stream_from_examples()
    finally:
        demo.cleanup()


if __name__ == '__main__':
    asyncio.run(main())
