#!/usr/bin/env python
"""
FastAPI server for RAGAnything - bridges React frontend with RAG query system

This server exposes REST API endpoints that the frontend expects, using the same
RAG setup logic from simple_query.py.
"""

import os
import asyncio
import logging
import uuid
import json
import shutil
from pathlib import Path
from typing import Optional, List, Dict, Any
from datetime import datetime, timedelta
from contextlib import asynccontextmanager
from enum import Enum

from fastapi import FastAPI, HTTPException, Request, BackgroundTasks, UploadFile, File, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from starlette.middleware.base import BaseHTTPMiddleware
from pydantic import BaseModel, Field
from dotenv import load_dotenv
import bcrypt
from jose import JWTError, jwt
import aiosqlite

# Add project root directory to Python path
import sys
sys.path.append(str(Path(__file__).parent))

from lightrag.llm.openai import openai_complete_if_cache, openai_embed
from lightrag.utils import EmbeddingFunc
from raganything import RAGAnything, RAGAnythingConfig

# Load environment variables
load_dotenv(dotenv_path=".env", override=False)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Filter to suppress status endpoint access logs
class StatusEndpointFilter(logging.Filter):
    """Filter out access logs for status endpoints"""
    def filter(self, record):
        # Filter out uvicorn access logs for paths ending with /status
        message = record.getMessage()
        if "/status" in message and "GET" in message:
            return False
        return True

# Apply filter to uvicorn access logger
uvicorn_access_logger = logging.getLogger("uvicorn.access")
uvicorn_access_logger.addFilter(StatusEndpointFilter())

# ============================================================================
# Authentication Configuration
# ============================================================================

# JWT Configuration
SECRET_KEY = os.getenv("JWT_SECRET_KEY", "your-secret-key-change-in-production")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "1440"))  # 24 hours default

# Security scheme
security = HTTPBearer()

# Users file path
USERS_FILE = Path(__file__).parent / "users.json"

# Database file path
DB_FILE = Path(__file__).parent / "chats.db"

# ============================================================================
# Database Setup
# ============================================================================

async def init_database():
    """Initialize SQLite database with tables and indexes"""
    async with aiosqlite.connect(DB_FILE) as db:
        # Create chats table
        await db.execute("""
            CREATE TABLE IF NOT EXISTS chats (
                id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                name TEXT NOT NULL,
                is_temporary BOOLEAN NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                last_message_at TEXT
            )
        """)
        
        # Create messages table
        await db.execute("""
            CREATE TABLE IF NOT EXISTS messages (
                id TEXT PRIMARY KEY,
                chat_id TEXT NOT NULL,
                content TEXT NOT NULL,
                sender TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                created_at TEXT NOT NULL,
                FOREIGN KEY (chat_id) REFERENCES chats(id) ON DELETE CASCADE
            )
        """)
        
        # Create indexes for performance
        await db.execute("CREATE INDEX IF NOT EXISTS idx_chats_user_id ON chats(user_id)")
        await db.execute("CREATE INDEX IF NOT EXISTS idx_chats_user_id_updated ON chats(user_id, updated_at DESC)")
        await db.execute("CREATE INDEX IF NOT EXISTS idx_messages_chat_id ON messages(chat_id)")
        await db.execute("CREATE INDEX IF NOT EXISTS idx_messages_chat_id_timestamp ON messages(chat_id, timestamp)")
        
        await db.commit()
        logger.info("Database initialized successfully")

async def get_db():
    """Get database connection"""
    async with aiosqlite.connect(DB_FILE) as db:
        db.row_factory = aiosqlite.Row
        yield db

# ============================================================================
# Authentication Functions
# ============================================================================

def load_users() -> Dict[str, Dict[str, str]]:
    """Load users from users.json file"""
    if not USERS_FILE.exists():
        logger.warning(f"Users file not found: {USERS_FILE}")
        return {}
    
    try:
        with open(USERS_FILE, 'r') as f:
            data = json.load(f)
            users = {}
            for user in data.get("users", []):
                username = user.get("username")
                password_hash = user.get("password_hash")
                if username and password_hash:
                    users[username] = {"password_hash": password_hash}
            return users
    except Exception as e:
        logger.error(f"Error loading users: {str(e)}")
        return {}

def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a password against a bcrypt hash"""
    try:
        return bcrypt.checkpw(plain_password.encode('utf-8'), hashed_password.encode('utf-8'))
    except Exception as e:
        logger.error(f"Error verifying password: {str(e)}")
        return False

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    """Create a JWT access token"""
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)) -> str:
    """Verify JWT token and return username"""
    token = credentials.credentials
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid authentication credentials",
                headers={"WWW-Authenticate": "Bearer"},
            )
        # Verify user still exists
        users = load_users()
        if username not in users:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="User no longer exists",
                headers={"WWW-Authenticate": "Bearer"},
            )
        return username
    except JWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )

# ============================================================================
# Pydantic Models
# ============================================================================

class LoginRequest(BaseModel):
    username: str = Field(..., min_length=1)
    password: str = Field(..., min_length=1)

class LoginResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    username: str

class ChatMessage(BaseModel):
    id: str
    content: str
    sender: str  # 'user' or 'assistant'
    timestamp: str
    loading: Optional[bool] = False

class ChatSendRequest(BaseModel):
    content: str = Field(..., min_length=1, max_length=10000)

class ChatSendResponse(BaseModel):
    id: str
    content: str
    sender: str
    timestamp: str

class HealthResponse(BaseModel):
    status: str
    system_ready: bool
    message: Optional[str] = None

class Document(BaseModel):
    id: str
    name: str
    path: str
    size: int
    upload_time: str
    processed: bool

class ErrorResponse(BaseModel):
    error: str
    code: str
    details: Optional[Dict[str, Any]] = None

class Chat(BaseModel):
    id: str
    user_id: str
    name: str
    is_temporary: bool
    created_at: str
    updated_at: str
    last_message_at: Optional[str] = None

class ChatListResponse(BaseModel):
    chats: List[Chat]

class ChatCreateRequest(BaseModel):
    name: Optional[str] = None
    is_temporary: bool = False

class ChatNameUpdateRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=100)

class ChatMessageRequest(BaseModel):
    content: str = Field(..., min_length=1, max_length=10000)
    chat_id: Optional[str] = None
    query_mode: Optional[str] = Field(default="auto", description="Query mode: 'all', 'selected', or 'auto'")
    document_names: Optional[List[str]] = Field(default=None, description="List of document names to query (for 'selected' mode)")
    filter_metadata: Optional[Dict[str, Any]] = Field(default=None, description="Metadata filters (e.g., {'file_type': 'pdf'})")

# ============================================================================
# RAG Manager (Singleton)
# ============================================================================

class RAGManager:
    """Singleton manager for RAGAnything instance"""
    
    _instance: Optional['RAGManager'] = None
    _rag: Optional[RAGAnything] = None
    _initialized: bool = False
    _initializing: bool = False
    _init_error: Optional[str] = None
    _processing_semaphore: Optional[asyncio.Semaphore] = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(RAGManager, cls).__new__(cls)
        return cls._instance
    
    def _get_processing_semaphore(self) -> asyncio.Semaphore:
        """Get or create the processing semaphore for concurrency control"""
        if self._processing_semaphore is None:
            # Default to 1 (sequential) to match process_upload_folder.py behavior
            max_concurrent = int(os.getenv("MAX_CONCURRENT_FILES", "1"))
            self._processing_semaphore = asyncio.Semaphore(max_concurrent)
            logger.info(f"Initialized file processing semaphore with max_concurrent={max_concurrent}")
        return self._processing_semaphore
    
    async def initialize(self, api_key: str, base_url: Optional[str] = None, working_dir: str = "./rag_storage"):
        """Initialize RAG instance if not already initialized"""
        if self._initialized:
            return True
        
        if self._initializing:
            # Wait for ongoing initialization
            while self._initializing:
                await asyncio.sleep(0.1)
            return self._initialized
        
        self._initializing = True
        self._init_error = None
        
        try:
            logger.info(f"Initializing RAG database from: {working_dir}")
            
            # Create RAGAnything configuration
            output_dir = os.getenv("OUTPUT_DIR", "./output")
            config = RAGAnythingConfig(
                working_dir=working_dir,
                parser="mineru",  # Not used for querying, but required
                parse_method="auto",
                parser_output_dir=output_dir,  # Set output directory for image path resolution
                enable_image_processing=True,
                enable_table_processing=True,
                enable_equation_processing=True,
            )

            # Define LLM model function
            llm_model_name = os.getenv("LLM_MODEL", "gpt-4o-mini")
            
            def llm_model_func(prompt, system_prompt=None, history_messages=[], **kwargs):
                return openai_complete_if_cache(
                    llm_model_name,
                    prompt,
                    system_prompt=system_prompt,
                    history_messages=history_messages,
                    api_key=api_key,
                    base_url=base_url,
                    **kwargs,
                )

            # Define vision model function for image processing
            def vision_model_func(
                prompt, system_prompt=None, history_messages=[], image_data=None, messages=None, **kwargs
            ):
                vision_model_name = os.getenv("VISION_MODEL", llm_model_name)
                
                # If messages format is provided (for multimodal VLM enhanced query), use it directly
                if messages:
                    return openai_complete_if_cache(
                        vision_model_name,
                        "",
                        system_prompt=None,
                        history_messages=[],
                        messages=messages,
                        api_key=api_key,
                        base_url=base_url,
                        **kwargs,
                    )
                # Traditional single image format
                elif image_data:
                    return openai_complete_if_cache(
                        vision_model_name,
                        "",
                        system_prompt=None,
                        history_messages=[],
                        messages=[
                            {"role": "system", "content": system_prompt}
                            if system_prompt
                            else None,
                            {
                                "role": "user",
                                "content": [
                                    {"type": "text", "text": prompt},
                                    {
                                        "type": "image_url",
                                        "image_url": {
                                            "url": f"data:image/jpeg;base64,{image_data}"
                                        },
                                    },
                                ],
                            }
                            if image_data
                            else {"role": "user", "content": prompt},
                        ],
                        api_key=api_key,
                        base_url=base_url,
                        **kwargs,
                    )
                # Pure text format
                else:
                    return llm_model_func(prompt, system_prompt, history_messages, **kwargs)

            # Define embedding function
            embedding_dim = int(os.getenv("EMBEDDING_DIM", "3072"))
            embedding_model = os.getenv("EMBEDDING_MODEL", "text-embedding-3-large")
            
            embedding_func = EmbeddingFunc(
                embedding_dim=embedding_dim,
                max_token_size=8192,
                func=lambda texts: openai_embed(
                    texts,
                    model=embedding_model,
                    api_key=api_key,
                    base_url=base_url,
                ),
            )

            # Initialize RAGAnything (will load existing database)
            self._rag = RAGAnything(
                config=config,
                llm_model_func=llm_model_func,
                vision_model_func=vision_model_func,
                embedding_func=embedding_func,
            )

            # Initialize LightRAG instance from existing storage
            init_result = await self._rag._ensure_lightrag_initialized()
            if not init_result.get("success", False):
                error_msg = f"Failed to initialize RAG database: {init_result.get('error', 'Unknown error')}"
                logger.error(error_msg)
                self._init_error = error_msg
                self._initializing = False
                return False

            self._initialized = True
            self._initializing = False
            logger.info("RAG database initialized successfully")
            return True
            
        except Exception as e:
            error_msg = f"Error initializing RAG: {str(e)}"
            logger.error(error_msg, exc_info=True)
            self._init_error = error_msg
            self._initializing = False
            return False
    
    def is_ready(self) -> bool:
        """Check if RAG is initialized and ready"""
        return self._initialized and self._rag is not None
    
    def get_init_error(self) -> Optional[str]:
        """Get initialization error if any"""
        return self._init_error
    
    def _detect_multi_document_query(self, query_text: str) -> tuple[bool, List[str]]:
        """
        Detect if query involves multiple documents and extract document names
        
        Args:
            query_text: Query text
            
        Returns:
            Tuple of (is_multi_doc, document_names)
        """
        import re
        
        # Common patterns for multi-document queries
        multi_doc_patterns = [
            r"compare.*?between.*?(?:and|vs|versus)",
            r"difference.*?between.*?(?:and|vs|versus)",
            r"similarity.*?between.*?(?:and|vs|versus)",
            r"compare.*?documents?",
            r"compare.*?files?",
        ]
        
        query_lower = query_text.lower()
        is_multi_doc = any(re.search(pattern, query_lower) for pattern in multi_doc_patterns)
        
        # Try to extract document names from query
        # Look for patterns like "document X and document Y" or file names
        document_names = []
        
        # Pattern 1: "compare X and Y" (only if comparison keywords found)
        if is_multi_doc:
            compare_match = re.search(r"compare\s+(.+?)\s+(?:and|vs|versus)\s+(.+)", query_lower)
            if compare_match:
                doc1 = compare_match.group(1).strip()
                doc2 = compare_match.group(2).strip()
                # Remove common words
                for word in ["document", "file", "the", "a", "an"]:
                    doc1 = doc1.replace(word, "").strip()
                    doc2 = doc2.replace(word, "").strip()
                if doc1:
                    document_names.append(doc1)
                if doc2:
                    document_names.append(doc2)
        
        # Pattern 2: Look for file names (always extract, not just for comparison queries)
        # This works for queries like "what is X in DocumentA and DocumentB"
        file_patterns = [
            r"([A-Za-z0-9_\-]+\.pdf)",
            r"([A-Za-z0-9_\-]+\.docx?)",
            r"([A-Z][a-z]+_[A-Z][a-z]+)",  # Pattern like "Housing_Concrete", "Office_Timber"
            r"([A-Z][a-zA-Z0-9_]+)",  # Pattern for capitalized names like "Office_Timber", "Housing_Concrete"
        ]
        
        for pattern in file_patterns:
            matches = re.findall(pattern, query_text)
            document_names.extend(matches)
        
        # Pattern 3: Look for "in X and in Y" or "in X and Y" patterns
        # This catches queries like "what is X in Office_Timber and in Housing_Concrete"
        in_pattern = re.findall(r"in\s+([A-Z][a-zA-Z0-9_]+)", query_text)
        if in_pattern:
            document_names.extend(in_pattern)
        
        # Remove duplicates and empty strings
        document_names = list(set([name for name in document_names if name]))
        
        # If we found 2+ document names, it's a multi-document query
        if len(document_names) >= 2:
            is_multi_doc = True
        
        return is_multi_doc, document_names

    async def query(
        self, 
        query_text: str, 
        mode: str = "hybrid", 
        timeout: int = 300, 
        document_names: List[str] = None,
        query_mode: str = "auto",
        filter_metadata: Dict[str, Any] = None
    ) -> str:
        """
        Execute a query against the RAG system
        
        Args:
            query_text: The query text
            mode: Retrieval mode ("hybrid", "local", "global")
            timeout: Query timeout in seconds
            document_names: Optional list of document names to query
            query_mode: Query mode - "all" (all documents), "selected" (specified documents), "auto" (detect or use all)
            filter_metadata: Optional metadata filters (e.g., {"file_type": "pdf"})
        """
        if not self.is_ready():
            raise RuntimeError("RAG system is not initialized")
        
        try:
            # System prompt for Czech language - preserves citations in original language
            czech_system_prompt = """Odpovídej vždy v češtině. Citáty a citace z dokumentů ponech vždy v původním jazyce, pokud nejsou v češtině. Nepřekládej citace z jiných jazyků.

Odpovídej stručně a pouze na to, na co se uživatel ptá. Neuváděj žádné dodatečné informace, které uživatel nepožádal."""
            
            # Determine query mode and document selection
            if query_mode == "auto":
                # Auto-detect: try to detect multi-document query if document_names not provided
                if document_names is None:
                    is_multi_doc, detected_doc_names = self._detect_multi_document_query(query_text)
                    logger.info(f"Document detection: is_multi_doc={is_multi_doc}, detected_names={detected_doc_names}")
                    if is_multi_doc and len(detected_doc_names) >= 2:
                        document_names = detected_doc_names
                        query_mode = "selected"
                        logger.info(f"Auto-detected multi-document query with documents: {document_names}")
                    else:
                        # Use all documents if no specific documents detected
                        query_mode = "all"
                        logger.info("Auto-mode: No specific documents detected, querying all documents")
            
            # Execute query based on mode
            if query_mode == "all" or (query_mode == "auto" and not document_names):
                # Query all documents
                logger.info(f"Querying all documents (mode: {query_mode})")
                result = await asyncio.wait_for(
                    self._rag.aquery_all_documents(
                        query_text,
                        mode=mode,
                        system_prompt=czech_system_prompt,
                        filter_metadata=filter_metadata,
                    ),
                    timeout=timeout
                )
            elif query_mode == "selected" or (query_mode == "auto" and document_names):
                # Query selected documents or use multi-document query
                if document_names and len(document_names) >= 2:
                    logger.info(f"Using multi-document query for: {document_names}")
                    # Map document names to IDs first to verify they exist
                    doc_name_to_id = await self._rag._map_document_names_to_ids(document_names)
                    if not doc_name_to_id or len(doc_name_to_id) < 2:
                        logger.warning(f"Could not find all documents. Found: {list(doc_name_to_id.keys()) if doc_name_to_id else []}")
                        logger.info("Falling back to querying all documents")
                        result = await asyncio.wait_for(
                            self._rag.aquery_all_documents(
                                query_text,
                                mode=mode,
                                system_prompt=czech_system_prompt,
                                filter_metadata=filter_metadata,
                            ),
                            timeout=timeout
                        )
                    else:
                        logger.info(f"Successfully mapped documents: {list(doc_name_to_id.keys())}")
                        result = await asyncio.wait_for(
                            self._rag.aquery_multi_document(
                                query_text,
                                document_names=document_names,
                                mode=mode,
                                query_mode="selected",
                                system_prompt=czech_system_prompt,
                                filter_metadata=filter_metadata,
                            ),
                            timeout=timeout
                        )
                else:
                    # Single document query (backwards compatible)
                    logger.info("Using single document query")
                    result = await asyncio.wait_for(
                        self._rag.aquery(query_text, mode=mode, system_prompt=czech_system_prompt),
                        timeout=timeout
                    )
            else:
                # Fallback to standard query
                result = await asyncio.wait_for(
                    self._rag.aquery(query_text, mode=mode, system_prompt=czech_system_prompt),
                    timeout=timeout
                )
            
            if result and isinstance(result, str):
                return result
            else:
                return str(result) if result else "No response generated"
                
        except asyncio.TimeoutError:
            raise RuntimeError(f"Query timed out after {timeout} seconds")
        except Exception as e:
            logger.error(f"Error executing query: {str(e)}", exc_info=True)
            raise
    
    def _is_retryable_error(self, error: Exception) -> bool:
        """
        Determine if an error is retryable (e.g., embedding errors, connection issues)
        
        Args:
            error: The exception that occurred
            
        Returns:
            True if the error should trigger a retry, False otherwise
        """
        error_str = str(error).lower()
        error_type = type(error).__name__
        
        # Retryable errors:
        # - Embedding errors (connection issues, timeouts, EOF)
        # - Connection errors
        # - Timeout errors
        # - Service unavailable errors
        retryable_patterns = [
            "embedding",
            "eof",
            "connection",
            "timeout",
            "llama runner process no longer running",
            "500",
            "503",
            "502",
            "network",
            "socket",
            "refused",
        ]
        
        retryable_types = [
            "ConnectionError",
            "TimeoutError",
            "OSError",
            "IOError",
        ]
        
        # Check error message patterns
        if any(pattern in error_str for pattern in retryable_patterns):
            return True
        
        # Check error types
        if any(retry_type in error_type for retry_type in retryable_types):
            return True
        
        return False

    async def process_uploaded_file(
        self,
        file_path: str,
        filename: str,
        status_key: str,
        output_dir: Optional[str] = None,
        parse_method: Optional[str] = None,
    ) -> None:
        """
        Process an uploaded file using the same logic as process_upload_folder.py
        Uses semaphore to control concurrency (sequential by default, matching process_upload_folder.py)
        Implements retry logic for retryable errors (e.g., embedding errors).
        
        Args:
            file_path: Path to the uploaded file
            filename: Original filename
            status_key: Key for tracking status in upload_statuses
            output_dir: Output directory for parsed files
            parse_method: Parse method to use
        """
        if not self.is_ready():
            raise RuntimeError("RAG system is not initialized")
        
        # Get retry configuration
        max_retries = int(os.getenv("PROCESSING_MAX_RETRIES", "3"))
        retry_delay_base = float(os.getenv("PROCESSING_RETRY_DELAY_BASE", "2.0"))  # Base delay in seconds
        
        # Acquire semaphore to control concurrency (sequential by default)
        semaphore = self._get_processing_semaphore()
        async with semaphore:
            last_error = None
            
            for attempt in range(max_retries):
                try:
                    # Initialize/reset status tracking for each attempt
                    if attempt == 0:
                        upload_statuses[status_key] = {
                            "status": "processing",
                            "progress": 0.0,
                            "message": "Starting file processing...",
                            "steps": [
                                {"id": "upload", "name": "Upload", "description": "File uploaded", "status": "completed", "progress": 1.0},
                                {"id": "extract", "name": "Extract", "description": "Extracting content...", "status": "in_progress", "progress": 0.0},
                                {"id": "chunk", "name": "Chunk", "description": "Chunking document...", "status": "pending", "progress": 0.0},
                                {"id": "embed", "name": "Embed", "description": "Creating vectors...", "status": "pending", "progress": 0.0},
                                {"id": "index", "name": "Index", "description": "Indexing...", "status": "pending", "progress": 0.0},
                            ],
                            "error": None,
                            "retry_count": 0,
                        }
                    else:
                        # Reset status for retry
                        upload_statuses[status_key]["status"] = "processing"
                        upload_statuses[status_key]["error"] = None
                        upload_statuses[status_key]["retry_count"] = attempt
                        upload_statuses[status_key]["message"] = f"Retrying processing (attempt {attempt + 1}/{max_retries})..."
                        # Reset all steps
                        for step in upload_statuses[status_key]["steps"]:
                            if step["id"] != "upload":
                                step["status"] = "pending"
                                step["progress"] = 0.0
                    
                    # Use config defaults if not provided
                    if output_dir is None:
                        output_dir = self._rag.config.parser_output_dir
                    if parse_method is None:
                        parse_method = self._rag.config.parse_method
                    
                    if attempt > 0:
                        logger.info(f"Retrying processing of uploaded file: {filename} (attempt {attempt + 1}/{max_retries})")
                    else:
                        logger.info(f"Processing uploaded file: {filename} (path: {file_path})")
                    
                    # Step 1: Extract (parse document)
                    upload_statuses[status_key]["steps"][1]["status"] = "in_progress"
                    upload_statuses[status_key]["steps"][1]["progress"] = 0.0
                    upload_statuses[status_key]["message"] = "Extracting content from document..." if attempt == 0 else f"Retrying extraction (attempt {attempt + 1}/{max_retries})..."
                    upload_statuses[status_key]["progress"] = 0.1
                    
                    # Process document using the same method as process_upload_folder.py
                    await self._rag.process_document_complete(
                        file_path=file_path,
                        output_dir=output_dir,
                        parse_method=parse_method,
                        display_stats=False,
                        file_name=filename,
                    )
                    
                    # Mark extract as completed
                    upload_statuses[status_key]["steps"][1]["status"] = "completed"
                    upload_statuses[status_key]["steps"][1]["progress"] = 1.0
                    upload_statuses[status_key]["steps"][1]["description"] = "Content extracted"
                    upload_statuses[status_key]["progress"] = 0.3
                    
                    # Steps 2-5 (chunk, embed, index) are handled internally by process_document_complete
                    # We'll mark them as completed since process_document_complete does everything
                    upload_statuses[status_key]["steps"][2]["status"] = "completed"
                    upload_statuses[status_key]["steps"][2]["progress"] = 1.0
                    upload_statuses[status_key]["steps"][2]["description"] = "Document chunked"
                    upload_statuses[status_key]["progress"] = 0.5
                    
                    upload_statuses[status_key]["steps"][3]["status"] = "completed"
                    upload_statuses[status_key]["steps"][3]["progress"] = 1.0
                    upload_statuses[status_key]["steps"][3]["description"] = "Vectors created"
                    upload_statuses[status_key]["progress"] = 0.8
                    
                    upload_statuses[status_key]["steps"][4]["status"] = "completed"
                    upload_statuses[status_key]["steps"][4]["progress"] = 1.0
                    upload_statuses[status_key]["steps"][4]["description"] = "Indexed"
                    upload_statuses[status_key]["progress"] = 1.0
                    
                    # Mark as completed
                    upload_statuses[status_key]["status"] = "completed"
                    upload_statuses[status_key]["message"] = f"Successfully processed {filename}" + (f" (after {attempt} retries)" if attempt > 0 else "")
                    logger.info(f"Successfully processed uploaded file: {filename}" + (f" (after {attempt} retries)" if attempt > 0 else ""))
                    return  # Success, exit retry loop
                    
                except Exception as e:
                    last_error = e
                    error_msg = f"Error processing file: {str(e)}"
                    logger.error(f"Error processing uploaded file {filename} (attempt {attempt + 1}/{max_retries}): {error_msg}", exc_info=True)
                    
                    # Check if error is retryable
                    is_retryable = self._is_retryable_error(e)
                    is_last_attempt = (attempt + 1) >= max_retries
                    
                    if is_retryable and not is_last_attempt:
                        # Calculate exponential backoff delay
                        retry_delay = retry_delay_base * (2 ** attempt)
                        logger.info(f"Retryable error detected. Will retry in {retry_delay:.1f} seconds...")
                        
                        upload_statuses[status_key]["status"] = "processing"
                        upload_statuses[status_key]["error"] = f"Temporary error (attempt {attempt + 1}/{max_retries}): {error_msg}"
                        upload_statuses[status_key]["message"] = f"Retrying in {retry_delay:.1f} seconds... (attempt {attempt + 1}/{max_retries})"
                        
                        # Mark current step as retrying
                        for step in upload_statuses[status_key]["steps"]:
                            if step["status"] == "in_progress":
                                step["status"] = "pending"  # Reset for retry
                                step["progress"] = 0.0
                                break
                        
                        # Wait before retrying
                        await asyncio.sleep(retry_delay)
                        continue  # Retry
                    else:
                        # Non-retryable error or last attempt - fail permanently
                        logger.error(f"Processing failed permanently for {filename}: {error_msg}")
                        
                        # Mark current step as error
                        for step in upload_statuses[status_key]["steps"]:
                            if step["status"] == "in_progress":
                                step["status"] = "error"
                                step["progress"] = 0.0
                                break
                        
                        upload_statuses[status_key]["status"] = "error"
                        upload_statuses[status_key]["error"] = error_msg + (f" (after {attempt + 1} attempts)" if attempt > 0 else "")
                        upload_statuses[status_key]["message"] = f"Processing failed: {error_msg}"
                        
                        raise  # Re-raise the exception

# ============================================================================
# Chat Name Generation Service
# ============================================================================

async def generate_chat_name(first_question: str, rag_manager: RAGManager) -> str:
    """Generate a chat name from the first question using AI"""
    try:
        prompt = f"""Generate a short, descriptive title (maximum 50 characters) for this conversation based on the first question. 
The title should be concise and capture the main topic. Return only the title, nothing else.

First question: {first_question[:200]}

Title:"""
        
        # Use a simple LLM call without RAG query
        llm_model_name = os.getenv("LLM_MODEL", "gpt-4o-mini")
        api_key = os.getenv("OPENAI_API_KEY") or os.getenv("LLM_BINDING_API_KEY")
        base_url = os.getenv("OPENAI_BASE_URL") or os.getenv("LLM_BINDING_HOST")
        
        result = openai_complete_if_cache(
            llm_model_name,
            prompt,
            system_prompt="You are a helpful assistant that generates concise, descriptive titles for conversations.",
            api_key=api_key,
            base_url=base_url,
        )
        
        # Clean up the result
        name = result.strip().strip('"').strip("'")
        if len(name) > 50:
            name = name[:47] + "..."
        
        return name if name else "New Chat"
    except Exception as e:
        logger.error(f"Error generating chat name: {str(e)}")
        # Fallback: use first 50 chars of question
        fallback = first_question[:50].strip()
        return fallback if fallback else "New Chat"

async def should_update_chat_name(
    chat_id: str, 
    old_name: str, 
    recent_messages: List[str],
    rag_manager: RAGManager
) -> Optional[str]:
    """Check if chat name should be updated and return new name if needed"""
    try:
        # Only check every 5 messages to avoid excessive LLM calls
        if len(recent_messages) < 5:
            return None
        
        # Get last 5 messages
        last_messages = recent_messages[-5:]
        messages_text = "\n".join([f"{i+1}. {msg[:100]}" for i, msg in enumerate(last_messages)])
        
        prompt = f"""Analyze if the conversation topic has changed significantly. 
Previous title: "{old_name}"
Recent messages:
{messages_text}

If the topic has changed significantly, generate a new short title (max 50 chars). 
If the topic is still the same, respond with "NO_CHANGE".
Return only the new title or "NO_CHANGE", nothing else."""
        
        llm_model_name = os.getenv("LLM_MODEL", "gpt-4o-mini")
        api_key = os.getenv("OPENAI_API_KEY") or os.getenv("LLM_BINDING_API_KEY")
        base_url = os.getenv("OPENAI_BASE_URL") or os.getenv("LLM_BINDING_HOST")
        
        result = openai_complete_if_cache(
            llm_model_name,
            prompt,
            system_prompt="You are a helpful assistant that analyzes conversation topics.",
            api_key=api_key,
            base_url=base_url,
        )
        
        result = result.strip().strip('"').strip("'")
        
        if result.upper() == "NO_CHANGE" or not result:
            return None
        
        # Clean up the result
        new_name = result.strip()
        if len(new_name) > 50:
            new_name = new_name[:47] + "..."
        
        return new_name if new_name and new_name != old_name else None
    except Exception as e:
        logger.error(f"Error checking chat name update: {str(e)}")
        return None

# ============================================================================
# In-Memory Storage (can be upgraded to database later)
# ============================================================================

# In-memory chat message storage (deprecated - using database now)
chat_messages: List[ChatMessage] = []

# In-memory storage for temporary chats (chat_id -> user_id mapping)
temporary_chats: Dict[str, str] = {}

# Upload processing status tracking
class ProcessingStep(str, Enum):
    UPLOAD = "upload"
    EXTRACT = "extract"
    CHUNK = "chunk"
    EMBED = "embed"
    INDEX = "index"

class StepStatus(str, Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    ERROR = "error"

# In-memory storage for upload processing status
upload_statuses: Dict[str, Dict[str, Any]] = {}

# Temporary upload directory
UPLOAD_TEMP_DIR = Path(__file__).parent / "temp_uploads"
UPLOAD_TEMP_DIR.mkdir(exist_ok=True)

# ============================================================================
# FastAPI App
# ============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup/shutdown"""
    # Startup
    logger.info("Starting API server...")
    
    # Initialize database
    await init_database()
    
    # Initialize RAG manager
    api_key = os.getenv("OPENAI_API_KEY") or os.getenv("LLM_BINDING_API_KEY")
    base_url = os.getenv("OPENAI_BASE_URL") or os.getenv("LLM_BINDING_HOST")
    working_dir = os.getenv("RAG_WORKING_DIR", "./rag_storage")
    
    if api_key:
        rag_manager = RAGManager()
        # Initialize in background to not block startup
        asyncio.create_task(rag_manager.initialize(api_key, base_url, working_dir))
    else:
        logger.warning("No API key found. RAG system will not be initialized.")
    
    yield
    
    # Shutdown
    logger.info("Shutting down API server...")

app = FastAPI(
    title="RAGAnything API",
    description="API server for RAGAnything query system",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware
# Note: When allow_credentials=True, you cannot use allow_origins=["*"]
# Must specify exact origins
cors_origins_env = os.getenv("CORS_ORIGINS", "")
if cors_origins_env:
    # Parse comma-separated origins from environment variable
    cors_origins = [origin.strip() for origin in cors_origins_env.split(",") if origin.strip()]
else:
    # Default to common development origins
    cors_origins = [
        "http://localhost:3000",
        "http://localhost:5173",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:5173",
    ]

logger.info(f"CORS enabled for origins: {cors_origins}")

# Add request logging middleware for debugging
class LoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        origin = request.headers.get("origin")
        method = request.method
        path = request.url.path
        
        # Skip logging status polling requests to reduce log noise
        if not path.endswith("/status"):
            logger.info(f"Incoming {method} request to {path} from origin: {origin}")
        
        response = await call_next(request)
        
        # Log CORS headers in response (only for non-status requests)
        if not path.endswith("/status"):
            cors_headers = {k: v for k, v in response.headers.items() if k.lower().startswith("access-control")}
            if cors_headers:
                logger.debug(f"CORS headers for {path}: {cors_headers}")
        
        return response

app.add_middleware(LoggingMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS", "PATCH"],
    allow_headers=["*"],
    expose_headers=["*"],
)

# ============================================================================
# Exception Handlers
# ============================================================================

@app.exception_handler(RuntimeError)
async def runtime_error_handler(request: Request, exc: RuntimeError):
    """Handle runtime errors"""
    logger.error(f"Runtime error: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={
            "error": str(exc),
            "code": "RUNTIME_ERROR",
            "details": None
        }
    )

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle general exceptions"""
    logger.error(f"Unhandled exception: {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "code": "INTERNAL_ERROR",
            "details": {"message": str(exc)} if os.getenv("DEBUG", "false").lower() == "true" else None
        }
    )

# ============================================================================
# API Endpoints
# ============================================================================

@app.post("/auth/login", response_model=LoginResponse)
async def login(request: LoginRequest):
    """Authenticate user and return JWT token"""
    logger.info(f"Login request received for username: {request.username}")
    users = load_users()
    
    if request.username not in users:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    user = users[request.username]
    if not verify_password(request.password, user["password_hash"]):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    # Create access token
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": request.username}, expires_delta=access_token_expires
    )
    
    logger.info(f"User {request.username} logged in successfully")
    return LoginResponse(
        access_token=access_token,
        token_type="bearer",
        username=request.username
    )

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint with system readiness status"""
    rag_manager = RAGManager()
    
    if rag_manager.is_ready():
        return HealthResponse(
            status="healthy",
            system_ready=True,
            message="RAG system is ready"
        )
    elif rag_manager._initializing:
        return HealthResponse(
            status="healthy",
            system_ready=False,
            message="RAG system is initializing..."
        )
    else:
        error = rag_manager.get_init_error()
        return HealthResponse(
            status="healthy",
            system_ready=False,
            message=f"RAG system not ready: {error or 'Not initialized'}"
        )

# ============================================================================
# Chat Management Endpoints
# ============================================================================

@app.post("/chats", response_model=Chat)
async def create_chat(request: ChatCreateRequest, current_user: str = Depends(get_current_user)):
    """Create a new chat"""
    chat_id = str(uuid.uuid4())
    now = datetime.now().isoformat()
    name = request.name or "New Chat"
    
    # Only save to database if not temporary
    if not request.is_temporary:
        async with aiosqlite.connect(DB_FILE) as db:
            await db.execute("""
                INSERT INTO chats (id, user_id, name, is_temporary, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (chat_id, current_user, name, request.is_temporary, now, now))
            await db.commit()
    else:
        # Store temporary chat in memory
        temporary_chats[chat_id] = current_user
        logger.info(f"Temporary chat created (in-memory): {chat_id} for user: {current_user}")
    
    logger.info(f"Chat created: {chat_id} for user: {current_user} (temporary: {request.is_temporary})")
    return Chat(
        id=chat_id,
        user_id=current_user,
        name=name,
        is_temporary=request.is_temporary,
        created_at=now,
        updated_at=now,
        last_message_at=None
    )

@app.get("/chats", response_model=ChatListResponse)
async def list_chats(current_user: str = Depends(get_current_user)):
    """List all chats for current user (only non-temporary chats from database)"""
    async with aiosqlite.connect(DB_FILE) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute("""
            SELECT id, user_id, name, is_temporary, created_at, updated_at, last_message_at
            FROM chats
            WHERE user_id = ? AND is_temporary = 0
            ORDER BY updated_at DESC
        """, (current_user,)) as cursor:
            rows = await cursor.fetchall()
            chats = [
                Chat(
                    id=row["id"],
                    user_id=row["user_id"],
                    name=row["name"],
                    is_temporary=bool(row["is_temporary"]),
                    created_at=row["created_at"],
                    updated_at=row["updated_at"],
                    last_message_at=row["last_message_at"]
                )
                for row in rows
            ]
    
    return ChatListResponse(chats=chats)

@app.get("/chats/{chat_id}", response_model=Chat)
async def get_chat(chat_id: str, current_user: str = Depends(get_current_user)):
    """Get chat details"""
    # Check if it's a temporary chat
    if chat_id in temporary_chats:
        if temporary_chats[chat_id] != current_user:
            raise HTTPException(
                status_code=404,
                detail="Chat not found"
            )
        # Return temporary chat info (not from database)
        now = datetime.now().isoformat()
        return Chat(
            id=chat_id,
            user_id=current_user,
            name="Temporary Chat",
            is_temporary=True,
            created_at=now,
            updated_at=now,
            last_message_at=None
        )
    
    # Get chat from database
    async with aiosqlite.connect(DB_FILE) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute("""
            SELECT id, user_id, name, is_temporary, created_at, updated_at, last_message_at
            FROM chats
            WHERE id = ? AND user_id = ?
        """, (chat_id, current_user)) as cursor:
            row = await cursor.fetchone()
            if not row:
                raise HTTPException(
                    status_code=404,
                    detail="Chat not found"
                )
            return Chat(
                id=row["id"],
                user_id=row["user_id"],
                name=row["name"],
                is_temporary=bool(row["is_temporary"]),
                created_at=row["created_at"],
                updated_at=row["updated_at"],
                last_message_at=row["last_message_at"]
            )

@app.get("/chats/{chat_id}/messages", response_model=List[ChatMessage])
async def get_chat_messages(chat_id: str, current_user: str = Depends(get_current_user)):
    """Get messages for a specific chat"""
    # Check if it's a temporary chat
    if chat_id in temporary_chats:
        if temporary_chats[chat_id] != current_user:
            raise HTTPException(
                status_code=404,
                detail="Chat not found"
            )
        # Temporary chats don't have messages in database, return empty list
        return []
    
    # Verify chat belongs to user and get messages from database
    async with aiosqlite.connect(DB_FILE) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute("""
            SELECT id FROM chats WHERE id = ? AND user_id = ?
        """, (chat_id, current_user)) as cursor:
            if not await cursor.fetchone():
                raise HTTPException(
                    status_code=404,
                    detail="Chat not found"
                )
        
        # Get messages
        db.row_factory = aiosqlite.Row
        async with db.execute("""
            SELECT id, content, sender, timestamp
            FROM messages
            WHERE chat_id = ?
            ORDER BY timestamp ASC
        """, (chat_id,)) as cursor:
            rows = await cursor.fetchall()
            messages = [
                ChatMessage(
                    id=row["id"],
                    content=row["content"],
                    sender=row["sender"],
                    timestamp=row["timestamp"],
                    loading=False
                )
                for row in rows
            ]
    
    return messages

@app.put("/chats/{chat_id}/name", response_model=Chat)
async def update_chat_name(
    chat_id: str,
    request: ChatNameUpdateRequest,
    current_user: str = Depends(get_current_user)
):
    """Manually update chat name"""
    # Temporary chats don't support name updates (not stored in database)
    if chat_id in temporary_chats:
        if temporary_chats[chat_id] != current_user:
            raise HTTPException(
                status_code=404,
                detail="Chat not found"
            )
        raise HTTPException(
            status_code=400,
            detail="Cannot update name for temporary chats"
        )
    
    now = datetime.now().isoformat()
    
    async with aiosqlite.connect(DB_FILE) as db:
        # Verify chat belongs to user
        db.row_factory = aiosqlite.Row
        async with db.execute("""
            SELECT id FROM chats WHERE id = ? AND user_id = ?
        """, (chat_id, current_user)) as cursor:
            if not await cursor.fetchone():
                raise HTTPException(
                    status_code=404,
                    detail="Chat not found"
                )
        
        # Update name
        await db.execute("""
            UPDATE chats
            SET name = ?, updated_at = ?
            WHERE id = ? AND user_id = ?
        """, (request.name, now, chat_id, current_user))
        await db.commit()
        
        # Get updated chat
        db.row_factory = aiosqlite.Row
        async with db.execute("""
            SELECT id, user_id, name, is_temporary, created_at, updated_at, last_message_at
            FROM chats
            WHERE id = ? AND user_id = ?
        """, (chat_id, current_user)) as cursor:
            row = await cursor.fetchone()
            return Chat(
                id=row["id"],
                user_id=row["user_id"],
                name=row["name"],
                is_temporary=bool(row["is_temporary"]),
                created_at=row["created_at"],
                updated_at=row["updated_at"],
                last_message_at=row["last_message_at"]
            )

@app.delete("/chats/{chat_id}")
async def delete_chat(chat_id: str, current_user: str = Depends(get_current_user)):
    """Delete a chat and all its messages"""
    # Check if it's a temporary chat
    if chat_id in temporary_chats:
        if temporary_chats[chat_id] != current_user:
            raise HTTPException(
                status_code=404,
                detail="Chat not found"
            )
        # Remove from in-memory storage
        del temporary_chats[chat_id]
        logger.info(f"Temporary chat deleted: {chat_id} by user: {current_user}")
        return {"status": "success", "message": "Chat deleted"}
    
    # Delete from database
    async with aiosqlite.connect(DB_FILE) as db:
        # Verify chat belongs to user
        db.row_factory = aiosqlite.Row
        async with db.execute("""
            SELECT id FROM chats WHERE id = ? AND user_id = ?
        """, (chat_id, current_user)) as cursor:
            if not await cursor.fetchone():
                raise HTTPException(
                    status_code=404,
                    detail="Chat not found"
                )
        
        # Delete chat (messages will be deleted via CASCADE)
        await db.execute("DELETE FROM chats WHERE id = ? AND user_id = ?", (chat_id, current_user))
        await db.commit()
    
    logger.info(f"Chat deleted: {chat_id} by user: {current_user}")
    return {"status": "success", "message": "Chat deleted"}

@app.post("/chats/{chat_id}/clear")
async def clear_chat_messages(chat_id: str, current_user: str = Depends(get_current_user)):
    """Clear all messages in a chat (keep the chat)"""
    # Temporary chats don't have messages in database, so nothing to clear
    if chat_id in temporary_chats:
        if temporary_chats[chat_id] != current_user:
            raise HTTPException(
                status_code=404,
                detail="Chat not found"
            )
        logger.info(f"Temporary chat messages cleared (no-op): {chat_id} by user: {current_user}")
        return {"status": "success", "message": "Chat messages cleared"}
    
    async with aiosqlite.connect(DB_FILE) as db:
        # Verify chat belongs to user
        db.row_factory = aiosqlite.Row
        async with db.execute("""
            SELECT id FROM chats WHERE id = ? AND user_id = ?
        """, (chat_id, current_user)) as cursor:
            if not await cursor.fetchone():
                raise HTTPException(
                    status_code=404,
                    detail="Chat not found"
                )
        
        # Delete messages
        await db.execute("DELETE FROM messages WHERE chat_id = ?", (chat_id,))
        await db.commit()
    
    logger.info(f"Chat messages cleared: {chat_id} by user: {current_user}")
    return {"status": "success", "message": "Chat messages cleared"}

# ============================================================================
# Legacy Chat Endpoints (for backward compatibility)
# ============================================================================

@app.get("/chat/messages", response_model=List[ChatMessage])
async def get_chat_messages(current_user: str = Depends(get_current_user)):
    """Get chat message history"""
    return chat_messages

@app.post("/chats/{chat_id}/messages", response_model=ChatSendResponse)
async def send_chat_message(
    chat_id: str,
    request: ChatMessageRequest,
    current_user: str = Depends(get_current_user)
):
    """Send a message to a specific chat and get AI response"""
    rag_manager = RAGManager()
    
    if not rag_manager.is_ready():
        raise HTTPException(
            status_code=503,
            detail={
                "error": "RAG system is not ready",
                "code": "SERVICE_UNAVAILABLE",
                "details": {"message": rag_manager.get_init_error() or "System is initializing"}
            }
        )
    
    # Verify chat belongs to user (check both database and temporary chats)
    is_temporary = False
    chat_name = None
    
    # First check if it's a temporary chat
    if chat_id in temporary_chats:
        if temporary_chats[chat_id] != current_user:
            raise HTTPException(
                status_code=404,
                detail="Chat not found"
            )
        is_temporary = True
        chat_name = "Temporary Chat"
    else:
        # Check database
        async with aiosqlite.connect(DB_FILE) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute("""
                SELECT id, name, is_temporary FROM chats WHERE id = ? AND user_id = ?
            """, (chat_id, current_user)) as cursor:
                chat_row = await cursor.fetchone()
                if not chat_row:
                    raise HTTPException(
                        status_code=404,
                        detail="Chat not found"
                    )
                is_temporary = bool(chat_row["is_temporary"])
                chat_name = chat_row["name"]
    
    try:
        # Execute query with optional parameters
        query_timeout = int(os.getenv("QUERY_TIMEOUT", "300"))  # Default 5 minutes
        result = await rag_manager.query(
            request.content, 
            mode="hybrid", 
            timeout=query_timeout,
            document_names=request.document_names,
            query_mode=request.query_mode or "auto",
            filter_metadata=request.filter_metadata
        )
        
        now = datetime.now().isoformat()
        
        # Create message IDs
        user_message_id = f"user-{int(datetime.now().timestamp() * 1000)}-{uuid.uuid4().hex[:9]}"
        ai_message_id = f"ai-{int(datetime.now().timestamp() * 1000)}-{uuid.uuid4().hex[:9]}"
        
        # Only store messages in database if chat is not temporary
        if not is_temporary:
            async with aiosqlite.connect(DB_FILE) as db:
                # Insert user message
                await db.execute("""
                    INSERT INTO messages (id, chat_id, content, sender, timestamp, created_at)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (user_message_id, chat_id, request.content, "user", now, now))
                
                # Insert assistant message
                await db.execute("""
                    INSERT INTO messages (id, chat_id, content, sender, timestamp, created_at)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (ai_message_id, chat_id, result, "assistant", now, now))
                
                # Update chat's last_message_at and updated_at
                await db.execute("""
                    UPDATE chats
                    SET last_message_at = ?, updated_at = ?
                    WHERE id = ?
                """, (now, now, chat_id))
                
                await db.commit()
            
            # Check if we need to generate/update chat name
            async with aiosqlite.connect(DB_FILE) as db:
                db.row_factory = aiosqlite.Row
                # Get message count
                async with db.execute("""
                    SELECT COUNT(*) as count FROM messages WHERE chat_id = ?
                """, (chat_id,)) as cursor:
                    msg_count_row = await cursor.fetchone()
                    msg_count = msg_count_row["count"] if msg_count_row else 0
                
                # Get first user message for name generation
                async with db.execute("""
                    SELECT content FROM messages 
                    WHERE chat_id = ? AND sender = 'user' 
                    ORDER BY timestamp ASC LIMIT 1
                """, (chat_id,)) as cursor:
                    first_msg_row = await cursor.fetchone()
                    first_question = first_msg_row["content"] if first_msg_row else None
                
                # Get chat name
                async with db.execute("""
                    SELECT name FROM chats WHERE id = ?
                """, (chat_id,)) as cursor:
                    chat_name_row = await cursor.fetchone()
                    current_name = chat_name_row["name"] if chat_name_row else "New Chat"
                
                # Generate name after first exchange (2 messages: user + assistant)
                if msg_count == 2 and first_question and current_name == "New Chat":
                    new_name = await generate_chat_name(first_question, rag_manager)
                    await db.execute("""
                        UPDATE chats SET name = ? WHERE id = ?
                    """, (new_name, chat_id))
                    await db.commit()
                    logger.info(f"Generated chat name: {new_name} for chat: {chat_id}")
                
                # Check for name update every 5 messages
                elif msg_count > 0 and msg_count % 5 == 0:
                    # Get recent messages
                    async with db.execute("""
                        SELECT content FROM messages 
                        WHERE chat_id = ? 
                        ORDER BY timestamp DESC LIMIT 5
                    """, (chat_id,)) as cursor:
                        recent_rows = await cursor.fetchall()
                        recent_messages = [row["content"] for row in recent_rows]
                    
                    new_name = await should_update_chat_name(chat_id, current_name, recent_messages, rag_manager)
                    if new_name:
                        await db.execute("""
                            UPDATE chats SET name = ? WHERE id = ?
                        """, (new_name, chat_id))
                        await db.commit()
                        logger.info(f"Updated chat name: {new_name} for chat: {chat_id}")
        else:
            logger.info(f"Temporary chat message not saved to database: {chat_id}")
        
        # Create response message
        response_message = ChatSendResponse(
            id=ai_message_id,
            content=result,
            sender="assistant",
            timestamp=now
        )
        
        logger.info(f"Query processed successfully: {request.content[:50]}...")
        return response_message
        
    except asyncio.TimeoutError:
        raise HTTPException(
            status_code=504,
            detail={
                "error": "Query timed out",
                "code": "TIMEOUT_ERROR",
                "details": {"timeout": query_timeout}
            }
        )
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail={
                "error": f"Error processing query: {str(e)}",
                "code": "QUERY_ERROR",
                "details": None
            }
        )

@app.post("/chat/send", response_model=ChatSendResponse)
async def send_chat_message_legacy(request: ChatMessageRequest, current_user: str = Depends(get_current_user)):
    """Legacy endpoint: Send a query and get AI response (creates default chat if needed)"""
    # If no chat_id provided, create or get default chat
    async with aiosqlite.connect(DB_FILE) as db:
        db.row_factory = aiosqlite.Row
        # Try to find a default chat for this user
        async with db.execute("""
            SELECT id FROM chats 
            WHERE user_id = ? AND is_temporary = 0 
            ORDER BY updated_at DESC LIMIT 1
        """, (current_user,)) as cursor:
            row = await cursor.fetchone()
            if row:
                chat_id = row["id"]
            else:
                # Create new chat
                chat_id = str(uuid.uuid4())
                now = datetime.now().isoformat()
                await db.execute("""
                    INSERT INTO chats (id, user_id, name, is_temporary, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (chat_id, current_user, "New Chat", False, now, now))
                await db.commit()
    
    # Forward to new endpoint
    return await send_chat_message(chat_id, request, current_user)

@app.post("/chat/clear")
async def clear_chat_legacy(current_user: str = Depends(get_current_user)):
    """Legacy endpoint: Clear chat message history (uses default chat)"""
    # Find default chat for user
    async with aiosqlite.connect(DB_FILE) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute("""
            SELECT id FROM chats 
            WHERE user_id = ? AND is_temporary = 0 
            ORDER BY updated_at DESC LIMIT 1
        """, (current_user,)) as cursor:
            row = await cursor.fetchone()
            if row:
                chat_id = row["id"]
                # Clear messages
                await db.execute("DELETE FROM messages WHERE chat_id = ?", (chat_id,))
                await db.commit()
                logger.info(f"Chat history cleared by user: {current_user}")
                return {"status": "success", "message": "Chat history cleared"}
    
    # Fallback: clear in-memory storage
    global chat_messages
    chat_messages.clear()
    logger.info(f"Chat history cleared by user: {current_user}")
    return {"status": "success", "message": "Chat history cleared"}

@app.get("/documents", response_model=List[Document])
async def get_documents(
    current_user: str = Depends(get_current_user),
    file_type: Optional[str] = None,
    status: Optional[str] = None,
    sort_by: Optional[str] = "name"
):
    """
    Get list of processed documents from RAG storage with optional filtering
    
    Args:
        file_type: Filter by file type (e.g., "pdf", "docx")
        status: Filter by status (e.g., "PROCESSED")
        sort_by: Sort field ("name", "date", "size")
    """
    rag_manager = RAGManager()
    
    if not rag_manager.is_ready():
        raise HTTPException(
            status_code=503,
            detail={
                "error": "RAG system is not ready",
                "code": "SERVICE_UNAVAILABLE"
            }
        )
    
    try:
        rag = rag_manager._rag
        
        # Build filter metadata
        filter_metadata = {}
        if file_type:
            filter_metadata["file_type"] = file_type
        if status:
            filter_metadata["status"] = status
        
        # Use the new get_all_documents method
        doc_list = await rag.get_all_documents(
            filter_metadata=filter_metadata if filter_metadata else None,
            sort_by=sort_by or "name"
        )
        
        # Convert to Document model format
        documents = []
        for doc_info in doc_list:
            file_path = doc_info.get("file_path", "")
            file_size = 0
            if file_path and Path(file_path).exists():
                try:
                    file_size = Path(file_path).stat().st_size
                except:
                    pass
            
            upload_time = doc_info.get("created_at", doc_info.get("updated_at", datetime.now().isoformat()))
            processed = doc_info.get("status") == "PROCESSED"
            
            documents.append(Document(
                id=doc_info["doc_id"],
                name=doc_info["name"],
                path=file_path,
                size=file_size,
                upload_time=upload_time,
                processed=processed
            ))
        
        return documents
        
    except Exception as e:
        logger.error(f"Error getting documents: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail={
                "error": f"Error getting documents: {str(e)}",
                "code": "DOCUMENT_ERROR",
                "details": {"message": str(e)}
            }
        )

@app.delete("/documents/{document_id}")
async def delete_document(document_id: str, current_user: str = Depends(get_current_user)):
    """Delete a document (placeholder - not fully implemented)"""
    # This would require integration with RAG storage to actually delete
    # For now, just return success
    logger.info(f"Delete document requested: {document_id} by user: {current_user}")
    return {"status": "success", "message": f"Document {document_id} deletion requested"}

@app.post("/upload")
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    current_user: str = Depends(get_current_user)
):
    """
    Upload and process a document using the same logic as process_upload_folder.py
    """
    rag_manager = RAGManager()
    
    if not rag_manager.is_ready():
        raise HTTPException(
            status_code=503,
            detail={
                "error": "RAG system is not ready",
                "code": "SERVICE_UNAVAILABLE",
                "details": {"message": rag_manager.get_init_error() or "System is initializing"}
            }
        )
    
    try:
        # Validate file
        if not file.filename:
            raise HTTPException(
                status_code=400,
                detail={
                    "error": "No filename provided",
                    "code": "INVALID_FILE"
                }
            )
        
        filename = file.filename
        logger.info(f"File upload received: {filename} from user: {current_user}")
        
        # Create unique file path in temp directory
        file_extension = Path(filename).suffix
        unique_filename = f"{uuid.uuid4().hex}{file_extension}"
        temp_file_path = UPLOAD_TEMP_DIR / unique_filename
        
        # Save uploaded file to temporary location
        try:
            with open(temp_file_path, "wb") as f:
                shutil.copyfileobj(file.file, f)
            logger.info(f"File saved to temporary location: {temp_file_path}")
        except Exception as e:
            logger.error(f"Error saving uploaded file: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail={
                    "error": f"Failed to save uploaded file: {str(e)}",
                    "code": "FILE_SAVE_ERROR"
                }
            )
        
        # Use filename as status key (frontend polls using filename)
        status_key = filename
        
        # Get configuration
        output_dir = os.getenv("OUTPUT_DIR", "./output")
        parse_method = os.getenv("PARSE_METHOD", "auto")
        
        # Start background processing
        background_tasks.add_task(
            rag_manager.process_uploaded_file,
            file_path=str(temp_file_path),
            filename=filename,
            status_key=status_key,
            output_dir=output_dir,
            parse_method=parse_method,
        )
        
        # Cleanup task: remove temp file after processing (with delay to ensure processing is done)
        async def cleanup_temp_file():
            await asyncio.sleep(300)  # Wait 5 minutes before cleanup
            try:
                if temp_file_path.exists():
                    temp_file_path.unlink()
                    logger.info(f"Cleaned up temporary file: {temp_file_path}")
            except Exception as e:
                logger.warning(f"Error cleaning up temp file {temp_file_path}: {str(e)}")
        
        background_tasks.add_task(cleanup_temp_file)
        
        logger.info(f"File upload accepted, processing started in background: {filename}")
        return {
            "status": "accepted",
            "message": f"File {filename} uploaded and processing started",
            "document_id": unique_filename,
            "filename": filename
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error handling upload: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail={
                "error": f"Error processing upload: {str(e)}",
                "code": "UPLOAD_ERROR"
            }
        )

@app.get("/{filename}/status")
async def get_upload_status(filename: str, current_user: str = Depends(get_current_user)):
    """Get processing status for an uploaded file"""
    # Check if we have status for this file
    if filename not in upload_statuses:
        # File might not be uploaded yet, or status was cleared
        # Return a default "processing" status to keep frontend polling
        return {
            "status": "processing",
            "progress": 0.0,
            "message": "Processing status not available yet",
            "steps": [
                {
                    "id": "upload",
                    "name": "Upload",
                    "description": "File uploaded",
                    "status": "completed",
                    "progress": 1.0
                },
                {
                    "id": "extract",
                    "name": "Extract",
                    "description": "Extracting content...",
                    "status": "in_progress",
                    "progress": 0.0
                },
                {
                    "id": "chunk",
                    "name": "Chunk",
                    "description": "Chunking document...",
                    "status": "pending",
                    "progress": 0.0
                },
                {
                    "id": "embed",
                    "name": "Embed",
                    "description": "Creating vectors...",
                    "status": "pending",
                    "progress": 0.0
                },
                {
                    "id": "index",
                    "name": "Index",
                    "description": "Indexing...",
                    "status": "pending",
                    "progress": 0.0
                }
            ]
        }
    
    status_data = upload_statuses[filename]
    
    # Return status with error if present
    response = {
        "status": status_data["status"],
        "progress": status_data["progress"],
        "message": status_data["message"],
        "steps": status_data["steps"]
    }
    
    if status_data.get("error"):
        response["error"] = status_data["error"]
    
    return response

# ============================================================================
# Main Entry Point
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    
    port = int(os.getenv("API_PORT", "8000"))
    host = os.getenv("API_HOST", "0.0.0.0")
    
    logger.info(f"Starting API server on {host}:{port}")
    uvicorn.run(
        "api_server:app",
        host=host,
        port=port,
        reload=os.getenv("DEBUG", "false").lower() == "true",
        log_level="info"
    )

