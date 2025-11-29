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
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(RAGManager, cls).__new__(cls)
        return cls._instance
    
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
            config = RAGAnythingConfig(
                working_dir=working_dir,
                parser="mineru",  # Not used for querying, but required
                parse_method="auto",
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
    
    async def query(self, query_text: str, mode: str = "hybrid", timeout: int = 300) -> str:
        """Execute a query against the RAG system"""
        if not self.is_ready():
            raise RuntimeError("RAG system is not initialized")
        
        try:
            # Execute query with timeout
            result = await asyncio.wait_for(
                self._rag.aquery(query_text, mode=mode),
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
        
        Args:
            file_path: Path to the uploaded file
            filename: Original filename
            status_key: Key for tracking status in upload_statuses
            output_dir: Output directory for parsed files
            parse_method: Parse method to use
        """
        if not self.is_ready():
            raise RuntimeError("RAG system is not initialized")
        
        try:
            # Initialize status tracking
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
            }
            
            # Use config defaults if not provided
            if output_dir is None:
                output_dir = self._rag.config.parser_output_dir
            if parse_method is None:
                parse_method = self._rag.config.parse_method
            
            logger.info(f"Processing uploaded file: {filename} (path: {file_path})")
            
            # Step 1: Extract (parse document)
            upload_statuses[status_key]["steps"][1]["status"] = "in_progress"
            upload_statuses[status_key]["steps"][1]["progress"] = 0.0
            upload_statuses[status_key]["message"] = "Extracting content from document..."
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
            upload_statuses[status_key]["message"] = f"Successfully processed {filename}"
            logger.info(f"Successfully processed uploaded file: {filename}")
            
        except Exception as e:
            error_msg = f"Error processing file: {str(e)}"
            logger.error(f"Error processing uploaded file {filename}: {error_msg}", exc_info=True)
            
            # Mark current step as error
            for step in upload_statuses[status_key]["steps"]:
                if step["status"] == "in_progress":
                    step["status"] = "error"
                    step["progress"] = 0.0
                    break
            
            upload_statuses[status_key]["status"] = "error"
            upload_statuses[status_key]["error"] = error_msg
            upload_statuses[status_key]["message"] = f"Processing failed: {error_msg}"
            
            raise

# ============================================================================
# In-Memory Storage (can be upgraded to database later)
# ============================================================================

# In-memory chat message storage
chat_messages: List[ChatMessage] = []

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

@app.get("/chat/messages", response_model=List[ChatMessage])
async def get_chat_messages(current_user: str = Depends(get_current_user)):
    """Get chat message history"""
    return chat_messages

@app.post("/chat/send", response_model=ChatSendResponse)
async def send_chat_message(request: ChatSendRequest, current_user: str = Depends(get_current_user)):
    """Send a query and get AI response"""
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
        # Execute query
        query_timeout = int(os.getenv("QUERY_TIMEOUT", "300"))  # Default 5 minutes
        result = await rag_manager.query(request.content, mode="hybrid", timeout=query_timeout)
        
        # Create response message
        response_message = ChatSendResponse(
            id=f"ai-{int(datetime.now().timestamp() * 1000)}-{uuid.uuid4().hex[:9]}",
            content=result,
            sender="assistant",
            timestamp=datetime.now().isoformat()
        )
        
        # Store in chat history
        user_message = ChatMessage(
            id=f"user-{int(datetime.now().timestamp() * 1000)}-{uuid.uuid4().hex[:9]}",
            content=request.content,
            sender="user",
            timestamp=datetime.now().isoformat()
        )
        chat_messages.append(user_message)
        chat_messages.append(ChatMessage(**response_message.dict()))
        
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

@app.post("/chat/clear")
async def clear_chat(current_user: str = Depends(get_current_user)):
    """Clear chat message history"""
    global chat_messages
    chat_messages.clear()
    logger.info(f"Chat history cleared by user: {current_user}")
    return {"status": "success", "message": "Chat history cleared"}

@app.get("/documents", response_model=List[Document])
async def get_documents(current_user: str = Depends(get_current_user)):
    """Get list of processed documents from RAG storage"""
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
        documents = []
        rag = rag_manager._rag
        
        # Try to get documents from doc_status storage
        if rag and rag.lightrag and rag.lightrag.doc_status:
            try:
                # Access the underlying storage to get all document IDs
                # LightRAG uses KV storage for doc_status
                # We need to get all keys from the doc_status storage
                doc_status_storage = rag.lightrag.doc_status
                
                # Try to get all document IDs from storage
                # This depends on the storage implementation
                if hasattr(doc_status_storage, 'get_all_ids'):
                    doc_ids = await doc_status_storage.get_all_ids()
                elif hasattr(doc_status_storage, 'list_all'):
                    doc_ids = await doc_status_storage.list_all()
                else:
                    # Fallback: try to access the underlying storage
                    # For file-based storage, we can scan the directory
                    working_dir = Path(rag.config.working_dir)
                    doc_status_file = working_dir / "kv_store_doc_status.json"
                    
                    if doc_status_file.exists():
                        with open(doc_status_file, 'r') as f:
                            doc_status_data = json.load(f)
                            doc_ids = list(doc_status_data.keys()) if isinstance(doc_status_data, dict) else []
                    else:
                        doc_ids = []
                
                # Get document details for each ID
                for doc_id in doc_ids:
                    try:
                        doc_status = await doc_status_storage.get_by_id(doc_id)
                        if doc_status:
                            # Extract document information
                            file_path = doc_status.get("file_path", "")
                            file_name = Path(file_path).name if file_path else doc_id
                            upload_time = doc_status.get("created_at", doc_status.get("updated_at", datetime.now().isoformat()))
                            processed = doc_status.get("status") == "processed"
                            
                            # Try to get file size if file exists
                            file_size = 0
                            if file_path and Path(file_path).exists():
                                file_size = Path(file_path).stat().st_size
                            
                            documents.append(Document(
                                id=doc_id,
                                name=file_name,
                                path=file_path,
                                size=file_size,
                                upload_time=upload_time,
                                processed=processed
                            ))
                    except Exception as e:
                        logger.warning(f"Error getting document {doc_id}: {str(e)}")
                        continue
                        
            except Exception as e:
                logger.warning(f"Error accessing doc_status storage: {str(e)}")
                # Fallback: scan output directory if available
                output_dir = Path(os.getenv("OUTPUT_DIR", "./output"))
                if output_dir.exists():
                    for doc_dir in output_dir.iterdir():
                        if doc_dir.is_dir():
                            # Look for markdown or other processed files
                            md_files = list(doc_dir.glob("**/*.md"))
                            if md_files:
                                documents.append(Document(
                                    id=doc_dir.name,
                                    name=doc_dir.name,
                                    path=str(doc_dir),
                                    size=sum(f.stat().st_size for f in md_files),
                                    upload_time=datetime.fromtimestamp(doc_dir.stat().st_mtime).isoformat(),
                                    processed=True
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

