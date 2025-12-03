"""
PostgreSQL storage adapter for cognitive engine vectors and graph edges.

Provides persistent storage using pgvector for embeddings and standard tables
for graph structure, with efficient batching and connection pooling.
"""

import psycopg2
from psycopg2.extras import execute_batch, RealDictCursor
from psycopg2.pool import ThreadedConnectionPool
from typing import List, Dict, Tuple, Optional
import numpy as np
import json
from contextlib import contextmanager


class PostgresVectorStore:
    """
    Storage adapter for vector embeddings using pgvector.
    
    Handles embedding storage, retrieval, and similarity search with
    efficient batching and connection pooling.
    """
    
    def __init__(self, database_url: str, pool_size: int = 5):
        """
        Initialize Postgres vector store.
        
        Args:
            database_url: PostgreSQL connection string
            pool_size: Number of connections in pool
        """
        self.database_url = database_url
        self.pool = ThreadedConnectionPool(
            minconn=1,
            maxconn=pool_size,
            dsn=database_url
        )
        self.table_name = "memories"
    
    @contextmanager
    def get_connection(self):
        """Context manager for connection pooling."""
        conn = self.pool.getconn()
        try:
            yield conn
        finally:
            self.pool.putconn(conn)
    
    def insert(self, text: str, embedding: List[float], 
               metadata: Dict = None) -> int:
        """
        Insert single embedding.
        
        Args:
            text: Text content
            embedding: Embedding vector
            metadata: Optional metadata dictionary
            
        Returns:
            ID of inserted row
        """
        with self.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    f"""
                    INSERT INTO {self.table_name} (text, embedding, metadata)
                    VALUES (%s, %s, %s)
                    RETURNING id
                    """,
                    (text, embedding, json.dumps(metadata or {}))
                )
                row_id = cur.fetchone()[0]
                conn.commit()
                return row_id
    
    def insert_batch(self, texts: List[str], embeddings: List[List[float]], 
                     metadata_list: List[Dict] = None) -> List[int]:
        """
        Insert batch of embeddings efficiently.
        
        Args:
            texts: List of text content
            embeddings: List of embedding vectors
            metadata_list: Optional list of metadata dicts
            
        Returns:
            List of inserted row IDs
        """
        if metadata_list is None:
            metadata_list = [{}] * len(texts)
        
        records = [
            (text, embedding, json.dumps(meta))
            for text, embedding, meta in zip(texts, embeddings, metadata_list)
        ]
        
        with self.get_connection() as conn:
            with conn.cursor() as cur:
                # Use execute_values for better performance
                from psycopg2.extras import execute_values
                
                ids = execute_values(
                    cur,
                    f"""
                    INSERT INTO {self.table_name} (text, embedding, metadata)
                    VALUES %s
                    RETURNING id
                    """,
                    records,
                    template="(%s, %s, %s)",
                    fetch=True
                )
                conn.commit()
                return [row[0] for row in ids]
    
    def search_similar(self, query_embedding: List[float], 
                       limit: int = 10,
                       threshold: float = 0.7) -> List[Dict]:
        """
        Search for similar embeddings using cosine distance.
        
        Args:
            query_embedding: Query vector
            limit: Maximum number of results
            threshold: Minimum similarity threshold (0-1)
            
        Returns:
            List of results with id, text, similarity, metadata
        """
        with self.get_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(
                    """
                    SELECT * FROM match_memories(%s::vector, %s, %s)
                    """,
                    (query_embedding, threshold, limit)
                )
                return [dict(row) for row in cur.fetchall()]
    
    def get_by_id(self, memory_id: int) -> Optional[Dict]:
        """
        Get memory by ID.
        
        Args:
            memory_id: Memory ID
            
        Returns:
            Dictionary with memory data or None if not found
        """
        with self.get_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(
                    f"""
                    SELECT id, text, embedding, metadata, created_at, updated_at
                    FROM {self.table_name}
                    WHERE id = %s
                    """,
                    (memory_id,)
                )
                row = cur.fetchone()
                return dict(row) if row else None
    
    def get_batch(self, memory_ids: List[int]) -> List[Dict]:
        """
        Get multiple memories by ID.
        
        Args:
            memory_ids: List of memory IDs
            
        Returns:
            List of memory dictionaries
        """
        with self.get_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(
                    f"""
                    SELECT id, text, embedding, metadata, created_at, updated_at
                    FROM {self.table_name}
                    WHERE id = ANY(%s)
                    """,
                    (memory_ids,)
                )
                return [dict(row) for row in cur.fetchall()]
    
    def count(self) -> int:
        """Get total number of memories."""
        with self.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(f"SELECT COUNT(*) FROM {self.table_name}")
                return cur.fetchone()[0]
    
    def close(self):
        """Close all connections in pool."""
        self.pool.closeall()


class PostgresGraphStore:
    """
    Storage adapter for graph edges.
    
    Handles edge storage, retrieval, and updates with efficient batching.
    """
    
    def __init__(self, database_url: str, pool_size: int = 5):
        """
        Initialize Postgres graph store.
        
        Args:
            database_url: PostgreSQL connection string
            pool_size: Number of connections in pool
        """
        self.database_url = database_url
        self.pool = ThreadedConnectionPool(
            minconn=1,
            maxconn=pool_size,
            dsn=database_url
        )
        self.table_name = "graph_edges"
    
    @contextmanager
    def get_connection(self):
        """Context manager for connection pooling."""
        conn = self.pool.getconn()
        try:
            yield conn
        finally:
            self.pool.putconn(conn)
    
    def upsert_edge(self, source_id: int, target_id: int, weight: float,
                    metadata: Dict = None) -> None:
        """
        Insert or update an edge.
        
        Args:
            source_id: Source memory ID
            target_id: Target memory ID
            weight: Edge weight
            metadata: Optional metadata dictionary
        """
        with self.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    f"""
                    INSERT INTO {self.table_name} (source_id, target_id, weight, metadata)
                    VALUES (%s, %s, %s, %s)
                    ON CONFLICT (source_id, target_id)
                    DO UPDATE SET
                        weight = EXCLUDED.weight,
                        metadata = EXCLUDED.metadata,
                        updated_at = NOW()
                    """,
                    (source_id, target_id, weight, json.dumps(metadata or {}))
                )
                conn.commit()
    
    def upsert_edges_batch(self, edges: List[Tuple[int, int, float, Dict]]) -> None:
        """
        Batch upsert edges.
        
        Args:
            edges: List of (source_id, target_id, weight, metadata) tuples
        """
        if not edges:
            return
        
        records = [
            (src, tgt, weight, json.dumps(meta or {}))
            for src, tgt, weight, meta in edges
        ]
        
        with self.get_connection() as conn:
            with conn.cursor() as cur:
                execute_batch(
                    cur,
                    f"""
                    INSERT INTO {self.table_name} (source_id, target_id, weight, metadata)
                    VALUES (%s, %s, %s, %s)
                    ON CONFLICT (source_id, target_id)
                    DO UPDATE SET
                        weight = EXCLUDED.weight,
                        metadata = EXCLUDED.metadata,
                        updated_at = NOW()
                    """,
                    records
                )
                conn.commit()
    
    def get_neighbors(self, memory_id: int, 
                     min_weight: float = 0.0) -> List[Dict]:
        """
        Get all neighbors of a memory node.
        
        Args:
            memory_id: Memory ID
            min_weight: Minimum edge weight threshold
            
        Returns:
            List of neighbor dictionaries with target_id, weight, metadata
        """
        with self.get_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(
                    f"""
                    SELECT target_id, weight, metadata, updated_at
                    FROM {self.table_name}
                    WHERE source_id = %s AND weight >= %s
                    ORDER BY weight DESC
                    """,
                    (memory_id, min_weight)
                )
                return [dict(row) for row in cur.fetchall()]
    
    def get_edge(self, source_id: int, target_id: int) -> Optional[Dict]:
        """
        Get specific edge.
        
        Args:
            source_id: Source memory ID
            target_id: Target memory ID
            
        Returns:
            Edge dictionary or None if not found
        """
        with self.get_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(
                    f"""
                    SELECT source_id, target_id, weight, metadata, created_at, updated_at
                    FROM {self.table_name}
                    WHERE source_id = %s AND target_id = %s
                    """,
                    (source_id, target_id)
                )
                row = cur.fetchone()
                return dict(row) if row else None
    
    def get_all_edges(self, min_weight: float = 0.0) -> List[Dict]:
        """
        Get all edges above weight threshold.
        
        Args:
            min_weight: Minimum weight threshold
            
        Returns:
            List of edge dictionaries
        """
        with self.get_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(
                    f"""
                    SELECT source_id, target_id, weight, metadata
                    FROM {self.table_name}
                    WHERE weight >= %s
                    ORDER BY weight DESC
                    """,
                    (min_weight,)
                )
                return [dict(row) for row in cur.fetchall()]
    
    def count(self) -> int:
        """Get total number of edges."""
        with self.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(f"SELECT COUNT(*) FROM {self.table_name}")
                return cur.fetchone()[0]
    
    def close(self):
        """Close all connections in pool."""
        self.pool.closeall()
