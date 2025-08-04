import os
from typing import Optional

from pydantic import BaseModel, Field

# Set up the directory path
home_dir = os.path.expanduser("~")
mem0_dir = os.environ.get("MEM0_DIR") or os.path.join(home_dir, ".mem0")


class SQLiteConfig(BaseModel):
    db_path: str = Field(
        description="Path to the SQLite database file",
        default=os.path.join(mem0_dir, "history.db"),
    )