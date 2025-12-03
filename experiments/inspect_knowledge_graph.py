"""
Inspect the knowledge graph to see actual semantic relationships.

Shows which concepts are connected, their edge weights from gradient descent,
and similarity scores from embeddings.
"""

import sys
from pathlib import Path
import numpy as np
import json

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / '.env')

import os
from cognigraph.storage.postgres_store import PostgresVectorStore, PostgresGraphStore


def inspect_graph():
    """Inspect the knowledge graph and show relationships."""
    
    database_url = os.getenv(
        "DATABASE_URL",
        "postgresql://cognigraph:cognigraph_dev@localhost:5432/cognigraph"
    )
    
    vector_store = PostgresVectorStore(database_url)
    graph_store = PostgresGraphStore(database_url)
    
    print("="*70)
    print("🕸️  KNOWLEDGE GRAPH INSPECTION")
    print("="*70)
    
    # Get all memories
    total_memories = vector_store.count()
    print(f"\nTotal memories in database: {total_memories}")
    
    # Get all edges
    edges = graph_store.get_all_edges(min_weight=0.01)
    print(f"Total edges: {len(edges)}")
    
    if not edges:
        print("\n⚠️  No edges found! Graph might be empty.")
        return
    
    # Group edges by weight
    print("\n📊 Edge Weight Distribution:")
    weights = [e['weight'] for e in edges]
    print(f"   Min weight: {min(weights):.3f}")
    print(f"   Max weight: {max(weights):.3f}")
    print(f"   Mean weight: {np.mean(weights):.3f}")
    print(f"   Median weight: {np.median(weights):.3f}")
    
    # Show strongest connections
    edges_sorted = sorted(edges, key=lambda e: abs(e['weight']), reverse=True)
    
    print("\n🔗 Top 15 Strongest Connections:")
    print("="*70)
    
    for i, edge in enumerate(edges_sorted[:15], 1):
        source = vector_store.get_by_id(edge['source_id'])
        target = vector_store.get_by_id(edge['target_id'])
        
        if source and target:
            weight = edge['weight']
            
            # Calculate embedding similarity
            source_emb = np.array(json.loads(source['embedding']) if isinstance(source['embedding'], str) else source['embedding'], dtype=np.float32)
            target_emb = np.array(json.loads(target['embedding']) if isinstance(target['embedding'], str) else target['embedding'], dtype=np.float32)
            emb_sim = np.dot(source_emb, target_emb)
            
            print(f"\n{i}. Weight: {weight:.3f} | Embedding Similarity: {emb_sim:.3f}")
            print(f"   [{source['id']}] {source['text'][:60]}...")
            print(f"    ↔")
            print(f"   [{target['id']}] {target['text'][:60]}...")
    
    # Show some weak connections
    print("\n" + "="*70)
    print("🔗 Sample Weak Connections (bottom 5):")
    print("="*70)
    
    for i, edge in enumerate(edges_sorted[-5:], 1):
        source = vector_store.get_by_id(edge['source_id'])
        target = vector_store.get_by_id(edge['target_id'])
        
        if source and target:
            weight = edge['weight']
            source_emb = np.array(json.loads(source['embedding']) if isinstance(source['embedding'], str) else source['embedding'], dtype=np.float32)
            target_emb = np.array(json.loads(target['embedding']) if isinstance(target['embedding'], str) else target['embedding'], dtype=np.float32)
            emb_sim = np.dot(source_emb, target_emb)
            
            print(f"\n{i}. Weight: {weight:.3f} | Embedding Similarity: {emb_sim:.3f}")
            print(f"   [{source['id']}] {source['text'][:60]}...")
            print(f"    ↔")
            print(f"   [{target['id']}] {target['text'][:60]}...")
    
    # Show node connectivity
    print("\n" + "="*70)
    print("📊 Node Connectivity Analysis")
    print("="*70)
    
    # Count connections per node
    node_connections = {}
    for edge in edges:
        node_connections[edge['source_id']] = node_connections.get(edge['source_id'], 0) + 1
        node_connections[edge['target_id']] = node_connections.get(edge['target_id'], 0) + 1
    
    # Get most connected nodes
    sorted_nodes = sorted(node_connections.items(), key=lambda x: x[1], reverse=True)
    
    print("\nTop 10 Most Connected Nodes:")
    for node_id, conn_count in sorted_nodes[:10]:
        memory = vector_store.get_by_id(node_id)
        if memory:
            print(f"   [{node_id}] {conn_count} connections: \"{memory['text'][:55]}...\"")
    
    # Semantic clustering analysis
    print("\n" + "="*70)
    print("🧠 Semantic Relationship Analysis")
    print("="*70)
    
    # Pick a specific concept and show its neighborhood
    if total_memories > 0:
        # Let's examine the first memory's connections
        sample_memory = vector_store.get_by_id(sorted_nodes[0][0])
        
        if sample_memory:
            print(f"\nFocusing on: \"{sample_memory['text'][:60]}...\"")
            print(f"Memory ID: {sample_memory['id']}")
            
            # Get its neighbors
            neighbors = graph_store.get_neighbors(sample_memory['id'], min_weight=0.01)
            
            print(f"\nConnected to {len(neighbors)} other concepts:")
            neighbors_sorted = sorted(neighbors, key=lambda n: n['weight'], reverse=True)
            
            for i, neighbor in enumerate(neighbors_sorted[:10], 1):
                neighbor_memory = vector_store.get_by_id(neighbor['target_id'])
                if neighbor_memory:
                    # Calculate embedding similarity
                    sample_emb = np.array(json.loads(sample_memory['embedding']) if isinstance(sample_memory['embedding'], str) else sample_memory['embedding'], dtype=np.float32)
                    neighbor_emb = np.array(json.loads(neighbor_memory['embedding']) if isinstance(neighbor_memory['embedding'], str) else neighbor_memory['embedding'], dtype=np.float32)
                    emb_sim = np.dot(sample_emb, neighbor_emb)
                    
                    print(f"\n   {i}. Weight: {neighbor['weight']:.3f} | Similarity: {emb_sim:.3f}")
                    print(f"      \"{neighbor_memory['text'][:60]}...\"")
    
    print("\n" + "="*70)
    
    vector_store.close()
    graph_store.close()


if __name__ == '__main__':
    inspect_graph()
