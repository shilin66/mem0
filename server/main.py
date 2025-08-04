import logging
import os
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Depends
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field

from mem0 import AsyncMemory

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Load environment variables
load_dotenv()


POSTGRES_HOST = os.environ.get("POSTGRES_HOST", "postgres")
POSTGRES_PORT = os.environ.get("POSTGRES_PORT", "5432")
POSTGRES_DB = os.environ.get("POSTGRES_DB", "postgres")
POSTGRES_USER = os.environ.get("POSTGRES_USER", "postgres")
POSTGRES_PASSWORD = os.environ.get("POSTGRES_PASSWORD", "postgres")
POSTGRES_COLLECTION_NAME = os.environ.get("POSTGRES_COLLECTION_NAME", "memories")

NEO4J_URI = os.environ.get("NEO4J_URI", "bolt://neo4j:7687")
NEO4J_USERNAME = os.environ.get("NEO4J_USERNAME", "neo4j")
NEO4J_PASSWORD = os.environ.get("NEO4J_PASSWORD", "mem0graph")

MEMGRAPH_URI = os.environ.get("MEMGRAPH_URI", "bolt://localhost:7687")
MEMGRAPH_USERNAME = os.environ.get("MEMGRAPH_USERNAME", "memgraph")
MEMGRAPH_PASSWORD = os.environ.get("MEMGRAPH_PASSWORD", "mem0graph")

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
OPENAI_BASE_URL = os.environ.get("OPENAI_BASE_URL")
HISTORY_DB_PATH = os.environ.get("HISTORY_DB_PATH", "/app/history/history.db")
MONGODB_URI = os.environ.get("MONGODB_URI")
MONGODB_COLLECTION = os.environ.get("MONGODB_COLLECTION", "history")
API_KEY = os.environ.get("API_KEY")

DEFAULT_CONFIG = {
    "version": "v1.1",
    "vector_store": {
        "provider": "pgvector",
        "config": {
            "host": POSTGRES_HOST,
            "port": int(POSTGRES_PORT),
            "dbname": POSTGRES_DB,
            "user": POSTGRES_USER,
            "password": POSTGRES_PASSWORD,
            "collection_name": POSTGRES_COLLECTION_NAME,
        },
    },
    "graph_store": {
        "provider": "neo4j",
        "config": {"url": NEO4J_URI, "username": NEO4J_USERNAME, "password": NEO4J_PASSWORD},
    },
    "llm": {"provider": "openai", "config": {"api_key": OPENAI_API_KEY, "openai_base_url": OPENAI_BASE_URL, "temperature": 0.2, "model": "WebDancer-32B"}},
    "embedder": {"provider": "openai", "config": {"api_key": OPENAI_API_KEY, "openai_base_url": OPENAI_BASE_URL, "model": "bge-m3"}},
    "history_store": {
        "provider": "mongodb",
        "config": {
            "connection_string": MONGODB_URI
        }
    }
}

# 全局异步内存实例，在启动时初始化
ASYNC_MEMORY_INSTANCE = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    global ASYNC_MEMORY_INSTANCE
    # 启动时初始化
    try:
        ASYNC_MEMORY_INSTANCE = await AsyncMemory.from_config(DEFAULT_CONFIG)
        logging.info("AsyncMemory instance initialized successfully")
    except Exception as e:
        logging.error(f"Failed to initialize AsyncMemory instance: {e}")
        raise
    
    yield
    
    # 关闭时清理（如果需要的话）
    # 这里可以添加清理代码


app = FastAPI(
    title="Mem0 REST APIs",
    description="A REST API for managing and searching memories for your AI Agents and Apps.",
    version="1.0.0",
    lifespan=lifespan,
)

# API Key authentication
security = HTTPBearer()

async def verify_api_key(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Verify API key from Authorization header"""
    if not API_KEY:
        # If no API key is configured, skip authentication
        return True
    
    if credentials.credentials != API_KEY:
        raise HTTPException(
            status_code=401,
            detail="Invalid API key",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return True

# Create API router with v1 prefix
from fastapi import APIRouter
api_v1 = APIRouter(prefix="/v1", dependencies=[Depends(verify_api_key)])


class Message(BaseModel):
    role: str = Field(..., description="Role of the message (user or assistant).")
    content: str = Field(..., description="Message content.")


class MemoryCreate(BaseModel):
    messages: List[Message] = Field(..., description="List of messages to store.")
    user_id: Optional[str] = None
    agent_id: Optional[str] = None
    run_id: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    custom_fact_extraction_prompt: Optional[str] = Field(None, description="Custom prompt for fact extraction.")
    infer: Optional[bool] = Field(True, description="Whether to use LLM for fact extraction. Default is True.")
    async_mode: Optional[bool] = Field(False, description="Whether to create memory asynchronously. Default is False.")


class SearchRequest(BaseModel):
    query: str = Field(..., description="Search query.")
    user_id: Optional[str] = None
    run_id: Optional[str] = None
    agent_id: Optional[str] = None
    limit: Optional[int] = Field(100, description="Number of results to return.")
    filters: Optional[Dict[str, Any]] = None


@api_v1.post("/configure", summary="Configure Mem0")
async def set_config(config: Dict[str, Any]):
    """Set memory configuration."""
    global ASYNC_MEMORY_INSTANCE
    ASYNC_MEMORY_INSTANCE = await AsyncMemory.from_config(config)
    return {"message": "Configuration set successfully"}


async def get_memory_instance(custom_fact_extraction_prompt: Optional[str] = None) -> AsyncMemory:
    """获取内存实例，如果需要自定义配置则创建新实例"""
    global ASYNC_MEMORY_INSTANCE
    
    if ASYNC_MEMORY_INSTANCE is None:
        ASYNC_MEMORY_INSTANCE = await AsyncMemory.from_config(DEFAULT_CONFIG)
    
    if custom_fact_extraction_prompt:
        # 创建自定义配置的临时实例
        custom_config = DEFAULT_CONFIG.copy()
        custom_config["custom_fact_extraction_prompt"] = custom_fact_extraction_prompt
        return await AsyncMemory.from_config(custom_config)
    
    return ASYNC_MEMORY_INSTANCE


@api_v1.post("/memories", summary="Create memories")
async def add_memory(memory_create: MemoryCreate):
    """Store new memories."""
    if not any([memory_create.user_id, memory_create.agent_id, memory_create.run_id]):
        raise HTTPException(status_code=400, detail="At least one identifier (user_id, agent_id, run_id) is required.")

    memory_instance = await get_memory_instance(memory_create.custom_fact_extraction_prompt)

    params = {k: v for k, v in memory_create.model_dump().items() 
              if v is not None and k not in ["messages", "custom_fact_extraction_prompt", "async_mode"]}
    
    messages = [m.model_dump() for m in memory_create.messages]
    
    try:
        if memory_create.async_mode:
            # 异步创建memory，不等待结果
            import asyncio
            asyncio.create_task(memory_instance.add(messages=messages, **params))
            return JSONResponse(content={"message": "Memory creation started asynchronously", "status": "pending"})
        else:
            # 同步创建memory，等待结果
            response = await memory_instance.add(messages=messages, **params)
            return JSONResponse(content=response)
    except Exception as e:
        logging.exception("Error in add_memory:")
        raise HTTPException(status_code=500, detail=str(e))


@api_v1.get("/memories", summary="Get memories")
async def get_all_memories(
    user_id: Optional[str] = None,
    run_id: Optional[str] = None,
    agent_id: Optional[str] = None,
):
    """Retrieve stored memories."""
    if not any([user_id, run_id, agent_id]):
        raise HTTPException(status_code=400, detail="At least one identifier is required.")
    try:
        memory_instance = await get_memory_instance()
        params = {
            k: v for k, v in {"user_id": user_id, "run_id": run_id, "agent_id": agent_id}.items() if v is not None
        }
        return await memory_instance.get_all(**params)
    except Exception as e:
        logging.exception("Error in get_all_memories:")
        raise HTTPException(status_code=500, detail=str(e))


@api_v1.get("/memories/{memory_id}", summary="Get a memory")
async def get_memory(memory_id: str):
    """Retrieve a specific memory by ID."""
    try:
        memory_instance = await get_memory_instance()
        return await memory_instance.get(memory_id)
    except Exception as e:
        logging.exception("Error in get_memory:")
        raise HTTPException(status_code=500, detail=str(e))


@api_v1.post("/memories/search", summary="Search memories")
async def search_memories(search_req: SearchRequest):
    """Search for memories based on a query."""
    try:
        memory_instance = await get_memory_instance()
        params = {k: v for k, v in search_req.model_dump().items() if v is not None and k != "query"}
        return await memory_instance.search(query=search_req.query, **params)
    except Exception as e:
        logging.exception("Error in search_memories:")
        raise HTTPException(status_code=500, detail=str(e))


@api_v1.put("/memories/{memory_id}", summary="Update a memory")
async def update_memory(memory_id: str, updated_memory: Dict[str, Any]):
    """Update an existing memory."""
    try:
        memory_instance = await get_memory_instance()
        return await memory_instance.update(memory_id=memory_id, data=updated_memory)
    except Exception as e:
        logging.exception("Error in update_memory:")
        raise HTTPException(status_code=500, detail=str(e))


@api_v1.get("/memories/{memory_id}/history", summary="Get memory history")
async def memory_history(memory_id: str):
    """Retrieve memory history."""
    try:
        memory_instance = await get_memory_instance()
        return await memory_instance.history(memory_id=memory_id)
    except Exception as e:
        logging.exception("Error in memory_history:")
        raise HTTPException(status_code=500, detail=str(e))


@api_v1.delete("/memories/{memory_id}", summary="Delete a memory")
async def delete_memory(memory_id: str):
    """Delete a specific memory by ID."""
    try:
        memory_instance = await get_memory_instance()
        await memory_instance.delete(memory_id=memory_id)
        return {"message": "Memory deleted successfully"}
    except Exception as e:
        logging.exception("Error in delete_memory:")
        raise HTTPException(status_code=500, detail=str(e))


@api_v1.delete("/memories", summary="Delete all memories")
async def delete_all_memories(
    user_id: Optional[str] = None,
    run_id: Optional[str] = None,
    agent_id: Optional[str] = None,
):
    """Delete all memories for a given identifier."""
    if not any([user_id, run_id, agent_id]):
        raise HTTPException(status_code=400, detail="At least one identifier is required.")
    try:
        memory_instance = await get_memory_instance()
        params = {
            k: v for k, v in {"user_id": user_id, "run_id": run_id, "agent_id": agent_id}.items() if v is not None
        }
        await memory_instance.delete_all(**params)
        return {"message": "All relevant memories deleted"}
    except Exception as e:
        logging.exception("Error in delete_all_memories:")
        raise HTTPException(status_code=500, detail=str(e))


@api_v1.post("/reset", summary="Reset all memories")
async def reset_memory():
    """Completely reset stored memories."""
    try:
        memory_instance = await get_memory_instance()
        await memory_instance.reset()
        return {"message": "All memories reset"}
    except Exception as e:
        logging.exception("Error in reset_memory:")
        raise HTTPException(status_code=500, detail=str(e))


@api_v1.get("/storage/info", summary="Get storage backend information")
async def get_storage_info():
    """Get information about the current storage backends."""
    try:
        memory_instance = await get_memory_instance()
        
        # Get history storage info
        storage_type = type(memory_instance.db).__name__
        history_storage_info = {"type": storage_type}
        
        if hasattr(memory_instance.db, 'db_path'):
            history_storage_info["path"] = memory_instance.db.db_path
        elif hasattr(memory_instance.db, 'database_name'):
            history_storage_info["database"] = f"{memory_instance.db.database_name}.{memory_instance.db.collection_name}"
        
        # Get vector store info
        vector_store_info = {
            "type": type(memory_instance.vector_store).__name__,
            "provider": memory_instance.config.vector_store.provider,
        }
        
        # Get graph store info if enabled
        graph_store_info = None
        if hasattr(memory_instance, 'enable_graph') and memory_instance.enable_graph:
            graph_store_info = {
                "type": type(memory_instance.graph).__name__,
                "provider": memory_instance.config.graph_store.provider,
            }
        
        return {
            "history_storage": history_storage_info,
            "vector_store": vector_store_info,
            "graph_store": graph_store_info,
            "api_version": memory_instance.api_version
        }
    except Exception as e:
        logging.exception("Error in get_storage_info:")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/", summary="Redirect to the OpenAPI documentation", include_in_schema=False)
def home():
    """Redirect to the OpenAPI documentation."""
    return RedirectResponse(url="/docs")

@app.get("/health", summary="Health check endpoint", include_in_schema=False)
def health_check():
    """Health check endpoint that doesn't require authentication."""
    global ASYNC_MEMORY_INSTANCE
    
    health_info = {"status": "healthy"}
    
    # Add storage backend information if memory instance is available
    if ASYNC_MEMORY_INSTANCE and hasattr(ASYNC_MEMORY_INSTANCE, 'db'):
        try:
            storage_type = type(ASYNC_MEMORY_INSTANCE.db).__name__
            storage_info = {"type": storage_type}
            
            if hasattr(ASYNC_MEMORY_INSTANCE.db, 'db_path'):
                storage_info["path"] = ASYNC_MEMORY_INSTANCE.db.db_path
            elif hasattr(ASYNC_MEMORY_INSTANCE.db, 'database_name'):
                storage_info["database"] = f"{ASYNC_MEMORY_INSTANCE.db.database_name}.{ASYNC_MEMORY_INSTANCE.db.collection_name}"
            
            health_info["history_storage"] = storage_info
        except Exception as e:
            health_info["history_storage"] = {"error": str(e)}
    
    return health_info

# Include the v1 router
app.include_router(api_v1)

# 在文件末尾添加以下代码
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8889,
        reload=True,
        workers=1
    )
