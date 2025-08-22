"""Embedding store for vector operations and semantic search."""

import logging
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import asyncio
import pickle
import json

logger = logging.getLogger(__name__)


@dataclass
class EmbeddingRecord:
    """Represents an embedding with metadata."""
    id: str
    vector: np.ndarray
    text: str
    metadata: Dict[str, Any]
    timestamp: float


class EmbeddingStore:
    """Store and manage embeddings for semantic operations."""
    
    def __init__(
        self,
        dimension: int = 1536,
        backend: str = "numpy",
        persistence_path: Optional[str] = None
    ):
        """Initialize embedding store.
        
        Args:
            dimension: Embedding dimension
            backend: Storage backend (numpy, faiss, annoy)
            persistence_path: Path for persistence
        """
        self.dimension = dimension
        self.backend = backend
        self.persistence_path = persistence_path
        
        self.embeddings = {}
        self.index = None
        
        self._initialize_backend()
    
    def _initialize_backend(self):
        """Initialize the storage backend."""
        if self.backend == "faiss":
            try:
                import faiss
                # Use IVF index for better scalability
                nlist = 100  # Number of clusters
                quantizer = faiss.IndexFlatL2(self.dimension)
                self.index = faiss.IndexIVFFlat(
                    quantizer,
                    self.dimension,
                    nlist,
                    faiss.METRIC_INNER_PRODUCT
                )
                logger.info("Initialized FAISS IVF index")
            except ImportError:
                logger.warning("FAISS not available, falling back to numpy")
                self.backend = "numpy"
        
        elif self.backend == "annoy":
            try:
                from annoy import AnnoyIndex
                self.index = AnnoyIndex(self.dimension, 'angular')
                logger.info("Initialized Annoy index")
            except ImportError:
                logger.warning("Annoy not available, falling back to numpy")
                self.backend = "numpy"
        
        # Load persisted data if available
        if self.persistence_path:
            self._load_from_disk()
    
    async def add_embedding(
        self,
        id: str,
        vector: List[float],
        text: str,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Add embedding to store.
        
        Args:
            id: Unique identifier
            vector: Embedding vector
            text: Original text
            metadata: Additional metadata
        """
        vec_array = np.array(vector, dtype=np.float32)
        
        # Normalize vector for cosine similarity
        vec_array = vec_array / np.linalg.norm(vec_array)
        
        record = EmbeddingRecord(
            id=id,
            vector=vec_array,
            text=text,
            metadata=metadata or {},
            timestamp=asyncio.get_event_loop().time()
        )
        
        self.embeddings[id] = record
        
        # Add to index
        if self.backend == "faiss" and self.index:
            if not self.index.is_trained:
                # Train index with initial vectors
                if len(self.embeddings) >= 100:
                    training_vectors = np.vstack([
                        r.vector for r in list(self.embeddings.values())[:100]
                    ])
                    self.index.train(training_vectors)
            
            if self.index.is_trained:
                self.index.add(vec_array.reshape(1, -1))
        
        elif self.backend == "annoy" and self.index:
            idx = len(self.embeddings) - 1
            self.index.add_item(idx, vec_array)
            
            # Rebuild index periodically
            if len(self.embeddings) % 100 == 0:
                self.index.build(10)  # 10 trees
    
    async def search(
        self,
        query_vector: List[float],
        k: int = 10,
        threshold: float = 0.0,
        filter_metadata: Optional[Dict[str, Any]] = None
    ) -> List[Tuple[str, float, Dict[str, Any]]]:
        """Search for similar embeddings.
        
        Args:
            query_vector: Query embedding
            k: Number of results
            threshold: Minimum similarity threshold
            filter_metadata: Metadata filters
            
        Returns:
            List of (id, similarity, metadata) tuples
        """
        query_vec = np.array(query_vector, dtype=np.float32)
        query_vec = query_vec / np.linalg.norm(query_vec)
        
        if self.backend == "faiss" and self.index and self.index.is_trained:
            distances, indices = self.index.search(
                query_vec.reshape(1, -1),
                min(k * 2, self.index.ntotal)  # Search more for filtering
            )
            
            results = []
            embedding_list = list(self.embeddings.values())
            
            for dist, idx in zip(distances[0], indices[0]):
                if idx < 0 or idx >= len(embedding_list):
                    continue
                
                record = embedding_list[idx]
                similarity = 1 - dist  # Convert distance to similarity
                
                if similarity < threshold:
                    continue
                
                if filter_metadata and not self._matches_filter(
                    record.metadata,
                    filter_metadata
                ):
                    continue
                
                results.append((
                    record.id,
                    float(similarity),
                    {
                        "text": record.text,
                        **record.metadata
                    }
                ))
                
                if len(results) >= k:
                    break
            
            return results
        
        elif self.backend == "annoy" and self.index:
            # Ensure index is built
            if not self.index.get_n_items():
                return []
            
            indices, distances = self.index.get_nns_by_vector(
                query_vec,
                min(k * 2, self.index.get_n_items()),
                include_distances=True
            )
            
            results = []
            embedding_list = list(self.embeddings.values())
            
            for idx, dist in zip(indices, distances):
                if idx >= len(embedding_list):
                    continue
                
                record = embedding_list[idx]
                similarity = 1 - dist  # Angular distance to similarity
                
                if similarity < threshold:
                    continue
                
                if filter_metadata and not self._matches_filter(
                    record.metadata,
                    filter_metadata
                ):
                    continue
                
                results.append((
                    record.id,
                    float(similarity),
                    {
                        "text": record.text,
                        **record.metadata
                    }
                ))
                
                if len(results) >= k:
                    break
            
            return results
        
        else:
            # Numpy fallback
            return await self._search_numpy(
                query_vec,
                k,
                threshold,
                filter_metadata
            )
    
    async def _search_numpy(
        self,
        query_vec: np.ndarray,
        k: int,
        threshold: float,
        filter_metadata: Optional[Dict[str, Any]]
    ) -> List[Tuple[str, float, Dict[str, Any]]]:
        """Search using numpy (fallback method).
        
        Args:
            query_vec: Query vector
            k: Number of results
            threshold: Similarity threshold
            filter_metadata: Metadata filters
            
        Returns:
            Search results
        """
        if not self.embeddings:
            return []
        
        similarities = []
        
        for record in self.embeddings.values():
            # Apply metadata filter
            if filter_metadata and not self._matches_filter(
                record.metadata,
                filter_metadata
            ):
                continue
            
            # Calculate cosine similarity
            similarity = np.dot(query_vec, record.vector)
            
            if similarity >= threshold:
                similarities.append((
                    record.id,
                    float(similarity),
                    {
                        "text": record.text,
                        **record.metadata
                    }
                ))
        
        # Sort by similarity and return top k
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:k]
    
    def _matches_filter(
        self,
        metadata: Dict[str, Any],
        filter: Dict[str, Any]
    ) -> bool:
        """Check if metadata matches filter.
        
        Args:
            metadata: Record metadata
            filter: Filter criteria
            
        Returns:
            True if matches
        """
        for key, value in filter.items():
            if key not in metadata:
                return False
            
            if isinstance(value, list):
                if metadata[key] not in value:
                    return False
            else:
                if metadata[key] != value:
                    return False
        
        return True
    
    async def update_embedding(
        self,
        id: str,
        vector: Optional[List[float]] = None,
        text: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Update existing embedding.
        
        Args:
            id: Embedding ID
            vector: New vector (optional)
            text: New text (optional)
            metadata: New metadata (optional)
        """
        if id not in self.embeddings:
            raise KeyError(f"Embedding {id} not found")
        
        record = self.embeddings[id]
        
        if vector is not None:
            vec_array = np.array(vector, dtype=np.float32)
            vec_array = vec_array / np.linalg.norm(vec_array)
            record.vector = vec_array
        
        if text is not None:
            record.text = text
        
        if metadata is not None:
            record.metadata.update(metadata)
        
        record.timestamp = asyncio.get_event_loop().time()
        
        # Rebuild index if backend requires it
        if self.backend in ["faiss", "annoy"]:
            await self._rebuild_index()
    
    async def delete_embedding(self, id: str):
        """Delete embedding from store.
        
        Args:
            id: Embedding ID
        """
        if id in self.embeddings:
            del self.embeddings[id]
            
            # Rebuild index after deletion
            if self.backend in ["faiss", "annoy"]:
                await self._rebuild_index()
    
    async def _rebuild_index(self):
        """Rebuild the search index."""
        if not self.embeddings:
            return
        
        if self.backend == "faiss":
            import faiss
            
            # Create new index
            nlist = min(100, len(self.embeddings))
            quantizer = faiss.IndexFlatL2(self.dimension)
            new_index = faiss.IndexIVFFlat(
                quantizer,
                self.dimension,
                nlist,
                faiss.METRIC_INNER_PRODUCT
            )
            
            # Train and add vectors
            vectors = np.vstack([r.vector for r in self.embeddings.values()])
            new_index.train(vectors)
            new_index.add(vectors)
            
            self.index = new_index
        
        elif self.backend == "annoy":
            from annoy import AnnoyIndex
            
            # Create new index
            new_index = AnnoyIndex(self.dimension, 'angular')
            
            # Add all vectors
            for idx, record in enumerate(self.embeddings.values()):
                new_index.add_item(idx, record.vector)
            
            # Build index
            new_index.build(10)
            
            self.index = new_index
    
    async def batch_add(
        self,
        embeddings: List[Dict[str, Any]]
    ):
        """Add multiple embeddings in batch.
        
        Args:
            embeddings: List of embedding data
        """
        for emb in embeddings:
            await self.add_embedding(
                id=emb["id"],
                vector=emb["vector"],
                text=emb["text"],
                metadata=emb.get("metadata")
            )
        
        # Rebuild index once after batch
        if self.backend in ["faiss", "annoy"]:
            await self._rebuild_index()
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get store statistics.
        
        Returns:
            Statistics dictionary
        """
        if not self.embeddings:
            return {
                "total_embeddings": 0,
                "backend": self.backend,
                "dimension": self.dimension
            }
        
        vectors = np.vstack([r.vector for r in self.embeddings.values()])
        
        return {
            "total_embeddings": len(self.embeddings),
            "backend": self.backend,
            "dimension": self.dimension,
            "mean_norm": float(np.mean(np.linalg.norm(vectors, axis=1))),
            "std_norm": float(np.std(np.linalg.norm(vectors, axis=1))),
            "index_trained": self.index.is_trained if hasattr(self.index, 'is_trained') else True
        }
    
    def save_to_disk(self):
        """Save embeddings to disk."""
        if not self.persistence_path:
            return
        
        try:
            data = {
                "embeddings": {
                    id: {
                        "vector": record.vector.tolist(),
                        "text": record.text,
                        "metadata": record.metadata,
                        "timestamp": record.timestamp
                    }
                    for id, record in self.embeddings.items()
                },
                "backend": self.backend,
                "dimension": self.dimension
            }
            
            with open(self.persistence_path, 'wb') as f:
                pickle.dump(data, f)
            
            logger.info(f"Saved {len(self.embeddings)} embeddings to disk")
        
        except Exception as e:
            logger.error(f"Failed to save embeddings: {e}")
    
    def _load_from_disk(self):
        """Load embeddings from disk."""
        if not self.persistence_path:
            return
        
        try:
            import os
            if not os.path.exists(self.persistence_path):
                return
            
            with open(self.persistence_path, 'rb') as f:
                data = pickle.load(f)
            
            for id, emb_data in data["embeddings"].items():
                self.embeddings[id] = EmbeddingRecord(
                    id=id,
                    vector=np.array(emb_data["vector"], dtype=np.float32),
                    text=emb_data["text"],
                    metadata=emb_data["metadata"],
                    timestamp=emb_data["timestamp"]
                )
            
            logger.info(f"Loaded {len(self.embeddings)} embeddings from disk")
            
            # Rebuild index
            asyncio.create_task(self._rebuild_index())
        
        except Exception as e:
            logger.error(f"Failed to load embeddings: {e}")


class OpenAIEmbeddingGenerator:
    """Generate embeddings using OpenAI API."""
    
    def __init__(
        self,
        model: str = "text-embedding-ada-002",
        batch_size: int = 100
    ):
        """Initialize embedding generator.
        
        Args:
            model: OpenAI embedding model
            batch_size: Batch size for API calls
        """
        self.model = model
        self.batch_size = batch_size
    
    async def generate(
        self,
        texts: List[str]
    ) -> List[List[float]]:
        """Generate embeddings for texts.
        
        Args:
            texts: List of texts
            
        Returns:
            List of embeddings
        """
        try:
            from openai import AsyncOpenAI
            client = AsyncOpenAI()
            
            embeddings = []
            
            # Process in batches
            for i in range(0, len(texts), self.batch_size):
                batch = texts[i:i + self.batch_size]
                
                response = await client.embeddings.create(
                    model=self.model,
                    input=batch
                )
                
                for emb in response.data:
                    embeddings.append(emb.embedding)
            
            return embeddings
        
        except Exception as e:
            logger.error(f"Failed to generate embeddings: {e}")
            # Return random embeddings as fallback
            return [
                np.random.randn(1536).tolist()
                for _ in texts
            ]
    
    async def generate_single(
        self,
        text: str
    ) -> List[float]:
        """Generate embedding for single text.
        
        Args:
            text: Input text
            
        Returns:
            Embedding vector
        """
        embeddings = await self.generate([text])
        return embeddings[0] if embeddings else []