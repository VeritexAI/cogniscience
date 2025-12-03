"""
Full Streaming Memory Formation Demo with Cognitive Engine

Shows real-world text transforming into memories with gradient descent
consolidation, energy minimization, and emergent graph structure.
"""

import asyncio
import os
import sys
from pathlib import Path
from typing import List
import time
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / '.env')

from cognigraph import CognitiveEngine
from cognigraph.ingestion.embeddings import OpenAIEmbeddingGenerator
from cognigraph.storage.postgres_store import PostgresVectorStore, PostgresGraphStore
from cognigraph.memory.semantic_memory import SemanticMemoryManager


class FullStreamingDemo:
    """
    Complete streaming demo with cognitive engine integration.
    """
    
    def __init__(
        self,
        N: int = 100,
        d: int = 1536,
        database_url: str = None,
        similarity_threshold: float = 0.80
    ):
        """
        Initialize full streaming system.
        
        Args:
            N: Maximum number of memory nodes
            d: Embedding dimensionality
            database_url: Postgres connection string
            similarity_threshold: Similarity threshold for linking memories
        """
        self.N = N
        self.d = d
        
        # Database connection
        self.database_url = database_url or os.getenv(
            "DATABASE_URL",
            "postgresql://cognigraph:cognigraph_dev@localhost:5432/cognigraph"
        )
        
        # Initialize components
        print("🔧 Initializing cognitive system...")
        
        # OpenAI embeddings
        self.embedder = OpenAIEmbeddingGenerator(model="text-embedding-3-small")
        print(f"   ✓ OpenAI: {self.embedder.model} (dim={self.embedder.get_dimension()})")
        
        # Persistent storage
        self.vector_store = PostgresVectorStore(self.database_url)
        self.graph_store = PostgresGraphStore(self.database_url)
        print(f"   ✓ Postgres: {self.vector_store.count()} existing memories")
        
        # Cognitive engine (in-memory gradient descent)
        self.engine = CognitiveEngine(
            N=N,
            d=d,
            eta1=0.01,  # Slower consolidation for real-time observation
            eta2=0.005,
            eta3=0.01,
            lam=0.01,   # Moderate pruning
            random_seed=42
        )
        print(f"   ✓ Cognitive Engine: N={N}, η₁={self.engine.eta1}, λ={self.engine.lam}")
        
        # Semantic memory manager (bridges storage + engine)
        self.memory_manager = SemanticMemoryManager(
            vector_memory=self.engine.vector_memory,
            graph_memory=self.engine.graph_memory,
            vector_store=self.vector_store,
            graph_store=self.graph_store,
            similarity_threshold=similarity_threshold,
            max_capacity=N
        )
        print(f"   ✓ Memory Manager: threshold={similarity_threshold}")
        
        print()
    
    async def process_text_with_consolidation(
        self,
        text: str,
        consolidation_steps: int = 10
    ):
        """
        Process text through full pipeline:
        1. Generate embedding
        2. Add to memory manager (find similar, allocate node)
        3. Run gradient descent consolidation
        4. Sync updated weights to Postgres
        
        Args:
            text: Input text
            consolidation_steps: Number of gradient descent steps
        """
        start = time.time()
        
        print(f"\n📝 \"{text[:60]}...\"")
        
        # Step 1: Generate embedding
        embed_start = time.time()
        embedding = await self.embedder.embed_single(text)
        embed_time = time.time() - embed_start
        print(f"   ⚡ Embedded: {embed_time*1000:.0f}ms")
        
        # Step 2: Add to memory manager
        mem_start = time.time()
        node_idx, is_new, similar = self.memory_manager.add_or_link_memory(text, embedding)
        mem_time = time.time() - mem_start
        
        if similar:
            print(f"   🔍 Similar: {len(similar)} matches (top: {similar[0]['similarity']:.3f})")
            print(f"   🔗 Linked to nodes: {[m['node_idx'] for m in similar[:3]]}")
        else:
            print(f"   ✨ Novel concept! Node #{node_idx}")
        
        print(f"   💾 Stored: {mem_time*1000:.0f}ms")
        
        # Step 3: Gradient descent consolidation
        if self.memory_manager.next_node_idx > 1:
            cons_start = time.time()
            
            # Create simple cache with just this node
            cache_indices = [node_idx]
            
            # Run consolidation steps
            initial_energy = self.engine.compute_energy()['E_total']
            
            for _ in range(consolidation_steps):
                # Set cache with current node
                cache_input = self.engine.vector_memory.vectors[cache_indices]
                self.engine.set_cache(cache_input, cache_indices)
                self.engine.step(record_history=True)
            
            final_energy = self.engine.compute_energy()['E_total']
            cons_time = time.time() - cons_start
            
            energy_delta = final_energy - initial_energy
            
            print(f"   🧠 Consolidation: {consolidation_steps} steps, {cons_time*1000:.0f}ms")
            print(f"      Energy: {initial_energy:.2f} → {final_energy:.2f} (Δ{energy_delta:.2f})")
        
        # Step 4: Sync to Postgres (periodically)
        if self.memory_manager.next_node_idx % 5 == 0:
            self.memory_manager.sync_to_postgres()
            print(f"   💾 Synced graph to Postgres")
        
        elapsed = time.time() - start
        print(f"   ⏱️  Total: {elapsed*1000:.0f}ms")
    
    async def stream_examples(self):
        """Stream example AI/ML texts."""
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
            "Recurrent neural networks process sequential data with temporal dependencies.",
            "Attention mechanisms allow models to focus on relevant parts of the input.",
            "Generative adversarial networks learn to create realistic synthetic data.",
            "Autoencoders compress data into low-dimensional representations.",
            "Ensemble methods combine multiple models to improve prediction accuracy.",
        ]
        
        print("="*70)
        print("🧠 COGNITIVE ENGINE - STREAMING MEMORY FORMATION")
        print("="*70)
        print(f"Processing {len(examples)} AI/ML concepts with gradient descent...")
        print("="*70)
        
        for i, text in enumerate(examples, 1):
            print(f"\n[{i}/{len(examples)}]", end=" ")
            await self.process_text_with_consolidation(text, consolidation_steps=10)
            await asyncio.sleep(0.1)
        
        self.print_final_stats()
    
    def print_final_stats(self):
        """Print comprehensive final statistics."""
        stats = self.memory_manager.get_statistics()
        engine_state = self.engine.get_state()
        
        print("\n" + "="*70)
        print("📈 FINAL STATISTICS")
        print("="*70)
        
        print("\n💾 Memory Formation:")
        print(f"   New memories:        {stats['memories_created']}")
        print(f"   Linked memories:     {stats['memories_updated']}")
        print(f"   Similar found:       {stats['similar_found']}")
        print(f"   Active nodes:        {stats['active_nodes']}/{self.N}")
        print(f"   Utilization:         {stats['utilization']*100:.1f}%")
        
        print("\n🧠 Cognitive Engine:")
        final_energy = engine_state['energy_history'][-1]
        initial_energy = engine_state['energy_history'][0]
        energy_decrease = initial_energy - final_energy
        
        print(f"   Initial energy:      {initial_energy:.2f}")
        print(f"   Final energy:        {final_energy:.2f}")
        print(f"   Energy decreased:    {energy_decrease:.2f}")
        
        final_metrics = engine_state['metrics_history'][-1]
        print(f"   Graph density:       {final_metrics['density']:.3f}")
        print(f"   Mean degree:         {final_metrics['mean_degree']:.2f}")
        print(f"   Max degree:          {final_metrics['max_degree']}")
        
        print("\n💾 Persistent Storage:")
        print(f"   DB memories:         {stats['db_memories']}")
        print(f"   DB edges:            {stats['db_edges']}")
        print(f"   Sync operations:     {stats['sync_count']}")
        
        print("="*70)
        
        # Export knowledge graph preview
        print("\n🕸️  Knowledge Graph Preview (top 5 nodes by connections):")
        kg = self.memory_manager.export_knowledge_graph()
        kg_sorted = sorted(kg, key=lambda n: len(n['neighbors']), reverse=True)
        
        for node in kg_sorted[:5]:
            print(f"\n   Node #{node['node_idx']}: \"{node['text'][:50]}...\"")
            print(f"      Connections: {len(node['neighbors'])}")
            for neighbor in node['neighbors'][:3]:
                print(f"         → [{neighbor['weight']:.2f}] \"{neighbor['text'][:40]}...\"")
        
        print("\n" + "="*70)
    
    def cleanup(self):
        """Close connections."""
        self.vector_store.close()
        self.graph_store.close()


async def main():
    """Main entry point."""
    demo = FullStreamingDemo(
        N=100,  # Small scale for demo
        d=1536,  # OpenAI embedding size
        similarity_threshold=0.80
    )
    
    try:
        await demo.stream_examples()
    finally:
        demo.cleanup()


if __name__ == '__main__':
    asyncio.run(main())
