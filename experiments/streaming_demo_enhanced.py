"""
Enhanced Streaming Demo with Hierarchical Concepts & Recency-Enhanced Dreaming

Demonstrates the full cognitive pipeline:
1. Stream diverse text from RSS feeds
2. Form memories with gradient descent consolidation
3. Extract abstract concepts via HDBSCAN clustering
4. Run recency-enhanced dream phases with temporal proximity
5. Visualize dual-level (memory + concept) graph evolution

This integrates all features from the hierarchical-concepts-recency-dreaming plan.
"""

import asyncio
import os
import sys
from pathlib import Path
from typing import List, Dict
import time
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / '.env')

from cognigraph import CognitiveEngine
from cognigraph.ingestion.embeddings import OpenAIEmbeddingGenerator
from cognigraph.storage.postgres_store import PostgresVectorStore, PostgresGraphStore
from cognigraph.memory.semantic_memory import SemanticMemoryManager
from cognigraph.knowledge.concepts import ConceptExtractor
from cognigraph.dynamics.dreaming import (
    pre_dream_rumination,
    controlled_dreaming_step_recency,
    compute_combined_gradient,
    detect_creative_bridges
)
from cognigraph.energy.functions import compute_total_energy
from cognigraph.energy.gradients import compute_gradients


# Diverse RSS feed topics for concept diversity
DEMO_TEXTS = {
    'technology': [
        "Artificial intelligence systems are transforming how we process information.",
        "Cloud computing enables scalable data storage and processing globally.",
        "Quantum computers can solve certain problems exponentially faster.",
        "Edge computing brings processing closer to data sources.",
        "Machine learning models learn patterns from large datasets.",
    ],
    'science': [
        "DNA sequencing reveals genetic information encoded in organisms.",
        "Climate change affects global weather patterns and ecosystems.",
        "Neurons transmit signals through electrochemical processes.",
        "Dark matter makes up about 27% of the universe's mass-energy.",
        "Evolution by natural selection shapes species over generations.",
    ],
    'philosophy': [
        "Consciousness remains one of the hardest problems in philosophy.",
        "Free will debates question whether our choices are determined.",
        "Ethics examines moral principles that govern behavior.",
        "Epistemology studies the nature and scope of knowledge.",
        "Existence precedes essence in existentialist philosophy.",
    ],
    'economics': [
        "Supply and demand determine prices in competitive markets.",
        "Inflation erodes purchasing power over time.",
        "Central banks use interest rates to influence economic activity.",
        "GDP measures the total value of goods and services produced.",
        "Behavioral economics studies psychological factors in decisions.",
    ]
}


class EnhancedStreamingDemo:
    """
    Enhanced demo showcasing hierarchical concept extraction
    and recency-enhanced dreaming over streaming data.
    """
    
    def __init__(
        self,
        N: int = 200,
        d: int = 1536,
        database_url: str = None,
        similarity_threshold: float = 0.80
    ):
        """Initialize the enhanced streaming system."""
        self.N = N
        self.d = d
        
        # Database connection
        self.database_url = database_url or os.getenv(
            "DATABASE_URL",
            "postgresql://cognigraph:cognigraph_dev@localhost:5432/cognigraph"
        )
        
        print("=" * 70)
        print("🧠 COGNIGRAPH ENHANCED STREAMING DEMO")
        print("   Hierarchical Concepts + Recency-Enhanced Dreaming")
        print("=" * 70)
        print()
        
        # Initialize components
        print("🔧 Initializing cognitive system...")
        
        # OpenAI embeddings
        self.embedder = OpenAIEmbeddingGenerator(model="text-embedding-3-small")
        print(f"   ✓ OpenAI: {self.embedder.model}")
        
        # Persistent storage
        self.vector_store = PostgresVectorStore(self.database_url)
        self.graph_store = PostgresGraphStore(self.database_url)
        print(f"   ✓ Postgres: {self.vector_store.count()} existing memories")
        
        # Cognitive engine
        self.engine = CognitiveEngine(
            N=N, d=d,
            eta1=0.01, eta2=0.005, eta3=0.01, lam=0.01,
            random_seed=42
        )
        print(f"   ✓ Cognitive Engine: N={N}")
        
        # Semantic memory manager with activation tracking
        self.memory_manager = SemanticMemoryManager(
            vector_memory=self.engine.vector_memory,
            graph_memory=self.engine.graph_memory,
            vector_store=self.vector_store,
            graph_store=self.graph_store,
            similarity_threshold=similarity_threshold,
            max_capacity=N
        )
        print(f"   ✓ Memory Manager: threshold={similarity_threshold}")
        
        # Concept extractor for hierarchical abstraction
        self.concept_extractor = ConceptExtractor(
            min_cluster_size=3,
            min_samples=2
        )
        print(f"   ✓ Concept Extractor: min_cluster_size=3")
        
        # Track concept evolution
        self.concept_snapshots = []
        self.snapshot_timestamps = []
        
        print()
    
    async def process_text_with_activation(self, text: str, topic: str = None):
        """
        Process text through full pipeline with activation tracking.
        
        Args:
            text: Input text
            topic: Optional topic label for logging
        """
        start = time.time()
        topic_str = f"[{topic}] " if topic else ""
        
        print(f"\n📝 {topic_str}\"{text[:50]}...\"")
        
        # Generate embedding
        embedding = await self.embedder.embed_single(text)
        
        # Add to memory manager (triggers activation boost)
        node_idx, is_new, similar = self.memory_manager.add_or_link_memory(text, embedding)
        
        # Boost activation for this and similar memories
        self.memory_manager.access_memory(node_idx, boost_type='cache')
        for sim in similar:
            self.memory_manager.access_memory(sim['node_idx'], boost_type='access')
        
        if similar:
            print(f"   🔗 Linked: {len(similar)} similar (top: {similar[0]['similarity']:.3f})")
        else:
            print(f"   ✨ Novel! Node #{node_idx}, activation={self.memory_manager.nodes[node_idx].activation:.2f}")
        
        # Quick consolidation
        if self.memory_manager.next_node_idx > 1:
            cache_indices = [node_idx] + [s['node_idx'] for s in similar[:2]]
            cache_input = self.engine.vector_memory.vectors[cache_indices]
            self.engine.set_cache(cache_input, cache_indices)
            
            for _ in range(5):
                self.engine.step(record_history=False)
        
        elapsed = time.time() - start
        print(f"   ⏱️  {elapsed*1000:.0f}ms")
    
    def extract_concepts_from_memories(self):
        """
        Extract concepts from current memories using HDBSCAN.
        """
        n_memories = self.memory_manager.next_node_idx
        if n_memories < self.concept_extractor.min_cluster_size:
            print(f"\n⚠️  Not enough memories for concept extraction ({n_memories} < {self.concept_extractor.min_cluster_size})")
            return []
        
        print(f"\n🔬 Extracting concepts from {n_memories} memories...")
        
        # Gather embeddings, texts, and activations
        embeddings = []
        texts = []
        node_indices = []
        activations = []
        
        for idx, node in self.memory_manager.nodes.items():
            embeddings.append(self.memory_manager.vector_memory.vectors[idx])
            texts.append(node.text)
            node_indices.append(idx)
            activations.append(node.activation)
        
        embeddings = np.array(embeddings)
        activations = np.array(activations)
        
        # Extract concepts
        concepts = self.concept_extractor.extract_concepts(
            embeddings, texts, node_indices, activations
        )
        
        print(f"   ✓ Extracted {len(concepts)} concepts:")
        for c in concepts[:5]:  # Show top 5
            print(f"      • {c.name} ({len(c.member_indices)} members, coherence={c.coherence:.2f})")
        
        # Record snapshot
        self.concept_snapshots.append(self.concept_extractor.get_concept_summary())
        self.snapshot_timestamps.append(time.time())
        
        return concepts
    
    def run_recency_enhanced_dream(self, num_steps: int = 50, recency_bias: float = 0.3):
        """
        Run dreaming phase with recency-enhanced consolidation.
        
        Args:
            num_steps: Number of dream steps
            recency_bias: Weight for recency in noise (0=uniform, 1=fully recency-biased)
        """
        print(f"\n💤 Starting recency-enhanced dream phase ({num_steps} steps)...")
        
        # Apply decay to update activations
        self.memory_manager.update_activation_decay()
        
        # Get activation weights
        activations = self.memory_manager.get_activation_weights()
        
        # Pre-dream rumination: boost top-k activated memories
        top_k = min(10, self.memory_manager.next_node_idx)
        activations = pre_dream_rumination(activations, top_k=top_k, rumination_boost=0.2)
        
        print(f"   🔄 Pre-dream rumination: boosted top {top_k} memories")
        
        # Get last_accessed timestamps
        last_accessed = np.zeros(self.N)
        for idx, node in self.memory_manager.nodes.items():
            last_accessed[idx] = node.last_accessed
        
        # Clear cache for dreaming
        self.engine.cache_memory.clear()
        adjacency_before = self.engine.graph_memory.get_adjacency().copy()
        
        initial_energy = self.engine.compute_energy()['E_total']
        
        # Dream loop with recency-enhanced dynamics
        for step in range(num_steps):
            vectors = self.engine.vector_memory.get_vectors()
            adjacency = self.engine.graph_memory.get_adjacency()
            
            # Compute semantic gradients
            cache = np.empty((0, self.d))
            cache_indices = np.array([])
            grads = compute_gradients(vectors, adjacency, cache, cache_indices)
            
            # Combine with temporal gradient
            grad_G_combined = compute_combined_gradient(
                grads['grad_G'], last_accessed, adjacency,
                temporal_weight=0.3, window_seconds=3600.0
            )
            
            # Dream step with recency-weighted noise
            new_vectors, new_adjacency = controlled_dreaming_step_recency(
                vectors, adjacency,
                grads['grad_V'], grad_G_combined,
                activations,
                eta=self.engine.eta2,
                sigma_n=0.002,
                lam=self.engine.lam,
                recency_bias=recency_bias,
                random_state=self.engine.random_state
            )
            
            self.engine.vector_memory.set_vectors(new_vectors)
            self.engine.graph_memory.set_adjacency(new_adjacency)
        
        final_energy = self.engine.compute_energy()['E_total']
        
        # Detect creative bridges
        adjacency_after = self.engine.graph_memory.get_adjacency()
        bridges = detect_creative_bridges(adjacency_before, adjacency_after, threshold=0.05)
        
        print(f"   ✓ Dream complete: Energy {initial_energy:.2f} → {final_energy:.2f}")
        print(f"   🌉 Creative bridges formed: {len(bridges)}")
        
        if bridges:
            # Show top bridges with memory texts
            print("   Top bridges:")
            for i, j, strength in bridges[:3]:
                text_i = self.memory_manager.nodes.get(i, {})
                text_j = self.memory_manager.nodes.get(j, {})
                text_i_str = getattr(text_i, 'text', '?')[:30] if text_i else '?'
                text_j_str = getattr(text_j, 'text', '?')[:30] if text_j else '?'
                print(f"      {i}↔{j} (strength={strength:.3f})")
                print(f"         \"{text_i_str}...\"")
                print(f"         \"{text_j_str}...\"")
        
        return bridges
    
    async def run_demo(self):
        """Run the full enhanced demonstration."""
        print("\n" + "=" * 70)
        print("📡 PHASE 1: STREAMING DATA INGESTION")
        print("=" * 70)
        
        # Process texts from different topics
        memory_count = 0
        for topic, texts in DEMO_TEXTS.items():
            print(f"\n--- Topic: {topic.upper()} ---")
            for text in texts:
                await self.process_text_with_activation(text, topic)
                memory_count += 1
                await asyncio.sleep(0.1)
        
        print(f"\n✓ Ingested {memory_count} memories across {len(DEMO_TEXTS)} topics")
        
        # =========================================
        print("\n" + "=" * 70)
        print("🔬 PHASE 2: CONCEPT EXTRACTION")
        print("=" * 70)
        
        concepts = self.extract_concepts_from_memories()
        
        # Update concept activations
        def get_activation(idx):
            node = self.memory_manager.nodes.get(idx)
            return node.activation if node else 0.0
        
        self.concept_extractor.update_all_activations(get_activation)
        
        # =========================================
        print("\n" + "=" * 70)
        print("💤 PHASE 3: RECENCY-ENHANCED DREAMING")
        print("=" * 70)
        
        bridges = self.run_recency_enhanced_dream(num_steps=100, recency_bias=0.4)
        
        # =========================================
        print("\n" + "=" * 70)
        print("🔬 PHASE 4: POST-DREAM CONCEPT ANALYSIS")
        print("=" * 70)
        
        # Re-extract concepts after dreaming
        print("\nRe-clustering after dream consolidation...")
        self.concept_extractor = ConceptExtractor(min_cluster_size=3, min_samples=2)
        post_dream_concepts = self.extract_concepts_from_memories()
        
        # =========================================
        print("\n" + "=" * 70)
        print("📊 FINAL STATISTICS")
        print("=" * 70)
        
        self.print_final_stats()
        
        # =========================================
        print("\n" + "=" * 70)
        print("🎨 GENERATING VISUALIZATIONS")
        print("=" * 70)
        
        await self.generate_visualizations()
    
    def print_final_stats(self):
        """Print comprehensive final statistics."""
        mem_stats = self.memory_manager.get_statistics()
        concept_summary = self.concept_extractor.get_concept_summary()
        
        print(f"\n📊 Memory Statistics:")
        print(f"   Total memories: {mem_stats['total_memories']}")
        print(f"   Similar found: {mem_stats['similar_found']}")
        print(f"   Sync count: {mem_stats['sync_count']}")
        
        activations = self.memory_manager.get_activation_weights()
        active_memories = np.sum(activations > 0.1)
        print(f"   Active memories (>0.1): {active_memories}")
        print(f"   Mean activation: {np.mean(activations[activations > 0]):.3f}")
        
        print(f"\n🔬 Concept Statistics:")
        print(f"   Total concepts: {concept_summary['total_concepts']}")
        print(f"   Concept edges: {concept_summary['total_edges']}")
        print(f"   Avg coherence: {concept_summary['avg_coherence']:.3f}")
        print(f"   Avg activation: {concept_summary['avg_activation']:.3f}")
        print(f"   Avg members/concept: {concept_summary['avg_members']:.1f}")
        
        print(f"\n⚡ Engine Statistics:")
        state = self.engine.get_state()
        print(f"   Graph density: {self.engine.graph_memory.get_density():.4f}")
        print(f"   Final energy: {self.engine.compute_energy()['E_total']:.3f}")
        print(f"   Time steps: {state['time_step']}")
    
    async def generate_visualizations(self):
        """Generate and save visualizations."""
        try:
            from visualization.concept_viz import (
                plot_concept_graph,
                plot_activation_heatmap,
                plot_dual_level_graph
            )
            
            output_dir = Path(__file__).parent / "output"
            output_dir.mkdir(exist_ok=True)
            
            # 1. Concept graph
            print("   📈 Generating concept graph...")
            fig1 = plot_concept_graph(
                self.concept_extractor,
                title="Extracted Concepts",
                save_path=str(output_dir / "concept_graph.png")
            )
            print(f"      → Saved to {output_dir / 'concept_graph.png'}")
            
            # 2. Activation heatmap
            print("   📈 Generating activation heatmap...")
            activations = self.memory_manager.get_activation_weights()
            fig2 = plot_activation_heatmap(
                activations,
                title="Memory Activation Levels",
                save_path=str(output_dir / "activation_heatmap.png")
            )
            print(f"      → Saved to {output_dir / 'activation_heatmap.png'}")
            
            # 3. Dual-level graph
            print("   📈 Generating dual-level graph...")
            adjacency = self.engine.graph_memory.get_adjacency()
            fig3 = plot_dual_level_graph(
                adjacency,
                self.concept_extractor,
                activations,
                title="Memory + Concept Dual Graph",
                save_path=str(output_dir / "dual_level_graph.png")
            )
            print(f"      → Saved to {output_dir / 'dual_level_graph.png'}")
            
            print(f"\n✓ All visualizations saved to {output_dir}/")
            
        except ImportError as e:
            print(f"   ⚠️  Visualization skipped: {e}")
        except Exception as e:
            print(f"   ⚠️  Visualization error: {e}")


async def main():
    """Main entry point."""
    demo = EnhancedStreamingDemo(N=200, d=1536)
    await demo.run_demo()


if __name__ == "__main__":
    asyncio.run(main())
