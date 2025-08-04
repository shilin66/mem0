from typing import Optional

from pydantic import BaseModel, Field


class MongoDBConfig(BaseModel):
    connection_string: str = Field(
        description="MongoDB connection URI",
    )
    collection_name: str = Field(
        description="Collection name for history storage",
        default="history",
    )
    database_name: Optional[str] = Field(
        description="Database name (if not specified, will be auto-parsed from connection_string)",
        default=None,
    )