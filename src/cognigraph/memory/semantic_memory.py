"""
Semantic memory manager integrating persistent storage with cognitive engine.

Bridges the gap between Postgres storage and the in-memory cognitive engine,
handling dynamic node allocation, similarity-based lookup, and synchronization.
"""

import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import time

from ..storage.postgres_store import PostgresVectorStore, PostgresGraphStore
from .vector import VectorMemory
from .graph import GraphMemory


@dataclass
class MemoryNode:
    """Metadata for a memory node."""
    node_index: int  # Index in cognitive engine (0 to N-1)
    db_id: Optional[int]  # Database ID (if persisted)
    text: str
    timestamp: float
    access_count: int = 0


class SemanticMemoryManager:
    """
    Manages semantic memories with persistent storage and cognitive processing.
    
    Handles:
    - Dynamic node allocation from streaming inputs
    - Similarity-based lookup to avoid duplicates
    - Synchronization between in-memory engine and Postgres
    - Metadata tracking (text, timestamps, access patterns)
    """
    
    def __init__(
        self,
        vector_memory: VectorMemory,
        graph_memory: GraphMemory,
        vector_store: PostgresVectorStore,
        graph_store: PostgresGraphStore,
        similarity_threshold: float = 0.85,
        max_capacity: int = None
    ):
        """
        Initialize semantic memory manager.
        
        Args:
            vector_memory: In-memory vector storage (from cognitive engine)
            graph_memory: In-memory graph storage (from cognitive engine)
            vector_store: Persistent vector storage (Postgres)
            graph_store: Persistent graph storage (Postgres)
            similarity_threshold: Threshold for considering memories similar
            max_capacity: Maximum number of memories (None = use engine size)
        """
        self.vector_memory = vector_memory
        self.graph_memory = graph_memory
        self.vector_store = vector_store
        self.graph_store = graph_store
        
        self.N = vector_memory.N
        self.d = vector_memory.d
        self.similarity_threshold = similarity_threshold
        self.max_capacity = max_capacity or self.N
        
        # Node metadata tracking
        self.nodes: Dict[int, MemoryNode] = {}  # node_index -> MemoryNode
        self.db_to_node: Dict[int, int] = {}  # db_id -> node_index
        self.next_node_idx = 0
        
        # Statistics
        self.stats = {
            'memories_created': 0,
            'memories_updated': 0,
            'similar_found': 0,
            'sync_count': 0
        }
    
    def allocate_node(self, text: str, embedding: List[float], 
                     db_id: Optional[int] = None) -> int:
        """
        Allocate a new node in the cognitive engine.
        
        Args:
            text: Text content
            embedding: Embedding vector
            db_id: Optional database ID
            
        Returns:
            Node index in cognitive engine
            
        Raises:
            RuntimeError: If at capacity
        """
        if self.next_node_idx >= self.max_capacity:
            raise RuntimeError(f"Memory capacity reached ({self.max_capacity})")
        
        node_idx = self.next_node_idx
        self.next_node_idx += 1
        
        # Update vector memory
        embedding_array = np.array(embedding, dtype=np.float32)
        self.vector_memory.vectors[node_idx] = embedding_array
        self.vector_memory._normalize()
        
        # Create metadata
        self.nodes[node_idx] = MemoryNode(
            node_index=node_idx,
            db_id=db_id,
            text=text,
            timestamp=time.time()
        )
        
        if db_id is not None:
            self.db_to_node[db_id] = node_idx
        
        self.stats['memories_created'] += 1
        
        return node_idx
    
    def find_similar(self, embedding: List[float], 
                    limit: int = 5) -> List[Dict]:
        """
        Find similar existing memories.
        
        Searches both in-memory vectors and Postgres for similarities.
        
        Args:
            embedding: Query embedding
            limit: Maximum results
            
        Returns:
            List of similar memory dictionaries with node_idx, similarity, text
        """
        # Search in Postgres for broader coverage
        db_results = self.vector_store.search_similar(
            embedding,
            limit=limit,
            threshold=self.similarity_threshold
        )
        
        # Convert to node indices
        results = []
        for db_result in db_results:
            db_id = db_result['id']
            if db_id in self.db_to_node:
                node_idx = self.db_to_node[db_id]
                results.append({
                    'node_idx': node_idx,
                    'db_id': db_id,
                    'similarity': db_result['similarity'],
                    'text': db_result['text']
                })
        
        self.stats['similar_found'] += len(results)
        
        return results
    
    def add_or_link_memory(
        self,
        text: str,
        embedding: List[float]
    ) -> Tuple[int, bool, List[Dict]]:
        """
        Add new memory or link to existing similar memories.
        
        Args:
            text: Text content
            embedding: Embedding vector
            
        Returns:
            Tuple of (node_index, is_new, similar_memories)
        """
        # Check for similar memories
        similar = self.find_similar(embedding)
        
        # Store in Postgres first
        db_id = self.vector_store.insert(
            text=text,
            embedding=embedding,
            metadata={'timestamp': time.time()}
        )
        
        # Allocate node in cognitive engine
        node_idx = self.allocate_node(text, embedding, db_id)
        
        # Create graph edges if similar memories exist
        if similar:
            edges = [
                (db_id, match['db_id'], match['similarity'], {})
                for match in similar
            ]
            self.graph_store.upsert_edges_batch(edges)
            
            # Update in-memory graph
            for match in similar:
                other_idx = match['node_idx']
                # Add symmetric edges
                weight = float(match['similarity'])
                self.graph_memory.adjacency[node_idx, other_idx] = weight
                self.graph_memory.adjacency[other_idx, node_idx] = weight
            
            self.graph_memory._enforce_symmetry()
            self.graph_memory._squash_weights()
            
            self.stats['memories_updated'] += 1
            is_new = False
        else:
            is_new = True
        
        return node_idx, is_new, similar
    
    def get_node_text(self, node_idx: int) -> Optional[str]:
        """Get text for a node."""
        if node_idx in self.nodes:
            return self.nodes[node_idx].text
        return None
    
    def get_node_metadata(self, node_idx: int) -> Optional[MemoryNode]:
        """Get full metadata for a node."""
        return self.nodes.get(node_idx)
    
    def sync_to_postgres(self, force: bool = False):
        """
        Sync in-memory state to Postgres.
        
        Updates edge weights that have changed due to gradient descent.
        
        Args:
            force: Force sync even if no significant changes
        """
        # Get current graph state
        adjacency = self.graph_memory.get_adjacency()
        threshold = 0.01
        
        # Find edges with significant weights
        edges_to_sync = []
        
        for i in range(min(self.next_node_idx, self.N)):
            if i not in self.nodes:
                continue
            
            node_i = self.nodes[i]
            if node_i.db_id is None:
                continue
            
            for j in range(i + 1, min(self.next_node_idx, self.N)):
                if j not in self.nodes:
                    continue
                
                node_j = self.nodes[j]
                if node_j.db_id is None:
                    continue
                
                weight = adjacency[i, j]
                if abs(weight) > threshold:
                    edges_to_sync.append((
                        node_i.db_id,
                        node_j.db_id,
                        float(weight),
                        {'updated_by_gradient': True}
                    ))
        
        if edges_to_sync:
            self.graph_store.upsert_edges_batch(edges_to_sync)
            self.stats['sync_count'] += 1
    
    def get_statistics(self) -> Dict:
        """Get current statistics."""
        return {
            **self.stats,
            'active_nodes': self.next_node_idx,
            'capacity': self.max_capacity,
            'utilization': self.next_node_idx / self.max_capacity,
            'db_memories': self.vector_store.count(),
            'db_edges': self.graph_store.count()
        }
    
    def export_knowledge_graph(self) -> List[Dict]:
        """
        Export knowledge graph with text labels.
        
        Returns:
            List of nodes with text, embeddings, and connections
        """
        nodes = []
        
        for idx in range(self.next_node_idx):
            if idx not in self.nodes:
                continue
            
            node = self.nodes[idx]
            embedding = self.vector_memory.vectors[idx].tolist()
            
            # Get neighbors from graph
            neighbors = []
            for j in range(self.next_node_idx):
                if j == idx or j not in self.nodes:
                    continue
                
                weight = self.graph_memory.adjacency[idx, j]
                if abs(weight) > 0.01:
                    neighbors.append({
                        'node_idx': j,
                        'text': self.nodes[j].text,
                        'weight': float(weight)
                    })
            
            nodes.append({
                'node_idx': idx,
                'db_id': node.db_id,
                'text': node.text,
                'embedding': embedding,
                'timestamp': node.timestamp,
                'neighbors': neighbors
            })
        
        return nodes
