from typing import List, Dict, Any, Optional
import numpy as np
import json
import redis
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class EmbeddingStore:
    def __init__(self, redis_client: redis.Redis):
        self.redis_client = redis_client
        self.index_name = "embeddings_idx"
        self.key_prefix = "embedding:"
        
    async def store_embedding(
        self,
        doc_id: str,
        embedding: np.ndarray,
        metadata: Dict[str, Any],
        ttl: Optional[int] = None
    ) -> bool:
        try:
            key = f"{self.key_prefix}{doc_id}"
            
            data = {
                "id": doc_id,
                "embedding": embedding.tolist(),
                "metadata": metadata,
                "created_at": datetime.now().isoformat()
            }
            
            if ttl:
                self.redis_client.setex(key, ttl, json.dumps(data))
            else:
                self.redis_client.set(key, json.dumps(data))
            
            logger.info(f"Stored embedding for document: {doc_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to store embedding: {e}")
            return False
    
    async def get_embedding(self, doc_id: str) -> Optional[Dict[str, Any]]:
        try:
            key = f"{self.key_prefix}{doc_id}"
            data = self.redis_client.get(key)
            
            if data:
                parsed_data = json.loads(data)
                parsed_data["embedding"] = np.array(parsed_data["embedding"])
                return parsed_data
                
            return None
            
        except Exception as e:
            logger.error(f"Failed to retrieve embedding: {e}")
            return None
    
    async def batch_store(
        self,
        embeddings: List[Dict[str, Any]],
        ttl: Optional[int] = None
    ) -> int:
        stored_count = 0
        pipeline = self.redis_client.pipeline()
        
        for item in embeddings:
            try:
                doc_id = item["id"]
                key = f"{self.key_prefix}{doc_id}"
                
                data = {
                    "id": doc_id,
                    "embedding": item["embedding"].tolist() if isinstance(item["embedding"], np.ndarray) else item["embedding"],
                    "metadata": item.get("metadata", {}),
                    "created_at": datetime.now().isoformat()
                }
                
                if ttl:
                    pipeline.setex(key, ttl, json.dumps(data))
                else:
                    pipeline.set(key, json.dumps(data))
                
                stored_count += 1
                
            except Exception as e:
                logger.error(f"Failed to add embedding to batch: {e}")
                continue
        
        pipeline.execute()
        logger.info(f"Stored {stored_count} embeddings in batch")
        return stored_count
    
    async def delete_embedding(self, doc_id: str) -> bool:
        try:
            key = f"{self.key_prefix}{doc_id}"
            result = self.redis_client.delete(key)
            return result > 0
            
        except Exception as e:
            logger.error(f"Failed to delete embedding: {e}")
            return False
    
    async def clear_all(self) -> int:
        count = 0
        for key in self.redis_client.scan_iter(match=f"{self.key_prefix}*"):
            self.redis_client.delete(key)
            count += 1
        
        logger.info(f"Cleared {count} embeddings from store")
        return count
    
    async def get_stats(self) -> Dict[str, Any]:
        total_embeddings = 0
        total_size = 0
        
        for key in self.redis_client.scan_iter(match=f"{self.key_prefix}*"):
            total_embeddings += 1
            try:
                data = self.redis_client.get(key)
                if data:
                    total_size += len(data)
            except:
                continue
        
        return {
            "total_embeddings": total_embeddings,
            "total_size_bytes": total_size,
            "avg_size_bytes": total_size / max(total_embeddings, 1)
        }