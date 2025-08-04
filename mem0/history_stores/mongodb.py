import logging
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

try:
    from pymongo import MongoClient
    from pymongo.collection import Collection
    from pymongo.database import Database
    MONGODB_AVAILABLE = True
except ImportError:
    MONGODB_AVAILABLE = False

logger = logging.getLogger(__name__)


class MongoDBManager:
    """MongoDB-based history storage manager."""
    
    def __init__(self, connection_string: str, collection_name: str = "history", database_name: Optional[str] = None):
        if not MONGODB_AVAILABLE:
            raise ImportError("pymongo is required for MongoDB storage. Install it with: pip install pymongo")
        
        self.connection_string = connection_string
        
        # Parse database name from URI if not explicitly provided
        if database_name is None:
            parsed_uri = urlparse(connection_string)
            database_name = parsed_uri.path.lstrip('/') if parsed_uri.path else "mem0"
        
        self.database_name = database_name
        self.collection_name = collection_name
        
        try:
            self.client = MongoClient(connection_string)
            
            # Use get_database() for better error handling
            self.db: Database = self.client.get_database(self.database_name)
            self.collection: Collection = self.db[self.collection_name]
            
            # Test connection
            self.client.admin.command('ping')
            logger.info(f"Connected to MongoDB: {self.database_name}.{self.collection_name}")
            
            # Create indexes for better performance
            self._create_indexes()
            
        except Exception as e:
            logger.error(f"Failed to connect to MongoDB: {e}")
            raise
    
    def _create_indexes(self) -> None:
        """Create indexes for better query performance."""
        try:
            # Index on memory_id for faster history queries
            self.collection.create_index("memory_id")
            # Compound index for memory_id and created_at for ordered queries
            self.collection.create_index([("memory_id", 1), ("created_at", 1)])
        except Exception as e:
            logger.warning(f"Failed to create indexes: {e}")
    
    def add_history(
        self,
        memory_id: str,
        old_memory: Optional[str],
        new_memory: Optional[str],
        event: str,
        *,
        created_at: Optional[str] = None,
        updated_at: Optional[str] = None,
        is_deleted: int = 0,
        actor_id: Optional[str] = None,
        role: Optional[str] = None,
    ) -> None:
        """Add a history record to MongoDB."""
        try:
            document = {
                "_id": str(uuid.uuid4()),
                "memory_id": memory_id,
                "old_memory": old_memory,
                "new_memory": new_memory,
                "event": event,
                "created_at": created_at or datetime.now(timezone.utc).isoformat(),
                "updated_at": updated_at or datetime.now(timezone.utc).isoformat(),
                "is_deleted": bool(is_deleted),
                "actor_id": actor_id,
                "role": role,
            }
            
            self.collection.insert_one(document)
            logger.debug(f"Added history record for memory_id: {memory_id}")
            
        except Exception as e:
            logger.error(f"Failed to add history record: {e}")
            raise
    
    def get_history(self, memory_id: str) -> List[Dict[str, Any]]:
        """Retrieve history records for a specific memory_id."""
        try:
            cursor = self.collection.find(
                {"memory_id": memory_id}
            ).sort([("created_at", 1), ("updated_at", 1)])
            
            history = []
            for doc in cursor:
                # Convert MongoDB document to the expected format
                record = {
                    "id": doc["_id"],
                    "memory_id": doc["memory_id"],
                    "old_memory": doc["old_memory"],
                    "new_memory": doc["new_memory"],
                    "event": doc["event"],
                    "created_at": doc["created_at"],
                    "updated_at": doc["updated_at"],
                    "is_deleted": doc["is_deleted"],
                    "actor_id": doc["actor_id"],
                    "role": doc["role"],
                }
                history.append(record)
            
            return history
            
        except Exception as e:
            logger.error(f"Failed to get history for memory_id {memory_id}: {e}")
            raise
    
    def reset(self) -> None:
        """Drop all history records."""
        try:
            self.collection.delete_many({})
            logger.info("Reset MongoDB history collection")
        except Exception as e:
            logger.error(f"Failed to reset history collection: {e}")
            raise
    
    def close(self) -> None:
        """Close the MongoDB connection."""
        if hasattr(self, 'client') and self.client:
            self.client.close()
            logger.info("Closed MongoDB connection")
    
    def __del__(self):
        self.close()