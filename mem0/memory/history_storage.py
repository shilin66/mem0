import logging
import os
from typing import Any, Dict, List, Optional, Union

from mem0.history_stores.sqlite import SQLiteManager

logger = logging.getLogger(__name__)

try:
    from mem0.history_stores.mongodb import MongoDBManager
    MONGODB_AVAILABLE = True
except ImportError:
    MONGODB_AVAILABLE = False
    MongoDBManager = None


class HistoryStorageFactory:
    """Factory class for creating history storage instances."""
    
    @staticmethod
    def create_storage(config: Dict[str, Any]) -> Union[SQLiteManager, MongoDBManager]:
        """
        Create a history storage instance based on configuration.
        
        Args:
            config: Configuration dictionary containing storage settings
            
        Returns:
            Storage instance (SQLiteManager or MongoDBManager)
            
        Raises:
            ImportError: If MongoDB is configured but pymongo is not available
            Exception: If MongoDB is configured but connection fails
        """
        # Check for MongoDB configuration first
        mongodb_uri = config.get('mongodb_uri') or os.environ.get('MONGODB_URI')
        
        if mongodb_uri:
            if not MONGODB_AVAILABLE:
                raise ImportError(
                    "MongoDB URI provided but pymongo is not available. "
                    "Install pymongo with: pip install pymongo"
                )
            
            # Get optional overrides for database and collection names
            database_name = config.get('mongodb_database') or os.environ.get('MONGODB_DATABASE')
            collection_name = config.get('mongodb_collection') or os.environ.get('MONGODB_COLLECTION') or 'history'
            
            logger.info(f"Using MongoDB for history storage")
            # Let MongoDB connection errors propagate up
            return MongoDBManager(
                connection_string=mongodb_uri,
                collection_name=collection_name
            )
        
        # Use SQLite only if MongoDB is not configured
        history_db_path = config.get('history_db_path', ':memory:')
        logger.info(f"Using SQLite for history storage: {history_db_path}")
        return SQLiteManager(history_db_path)


class HistoryStorageInterface:
    """
    Interface wrapper for history storage that provides a consistent API
    regardless of the underlying storage backend.
    """
    
    def __init__(self, storage: Union[SQLiteManager, 'MongoDBManager']):
        self.storage = storage
        self.storage_type = type(storage).__name__
    
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
        """Add a history record."""
        return self.storage.add_history(
            memory_id=memory_id,
            old_memory=old_memory,
            new_memory=new_memory,
            event=event,
            created_at=created_at,
            updated_at=updated_at,
            is_deleted=is_deleted,
            actor_id=actor_id,
            role=role,
        )
    
    def get_history(self, memory_id: str) -> List[Dict[str, Any]]:
        """Retrieve history records for a specific memory_id."""
        return self.storage.get_history(memory_id)
    
    def reset(self) -> None:
        """Reset/clear all history records."""
        return self.storage.reset()
    
    def close(self) -> None:
        """Close the storage connection."""
        return self.storage.close()
    
    def get_storage_info(self) -> Dict[str, str]:
        """Get information about the current storage backend."""
        info = {"type": self.storage_type}
        
        if hasattr(self.storage, 'db_path'):
            info["path"] = self.storage.db_path
        elif hasattr(self.storage, 'connection_string'):
            # Don't expose full connection string for security
            info["database"] = f"{self.storage.database_name}.{self.storage.collection_name}"
        
        return info