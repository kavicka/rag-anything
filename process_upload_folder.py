#!/usr/bin/env python
"""
Process all files in docs/upload folder into RAG-Anything database

This script processes all supported files in the docs/upload folder,
parsing them and inserting them into the RAG-Anything knowledge base.
"""

import os
import sys
import asyncio
import logging
import logging.handlers
from pathlib import Path
from dotenv import load_dotenv

# Try to import httpx for health checks, but make it optional
try:
    import httpx
    HAS_HTTPX = True
except ImportError:
    HAS_HTTPX = False

# Add project root directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

from lightrag.llm.openai import openai_complete_if_cache, openai_embed
from lightrag.utils import EmbeddingFunc, logger, set_verbose_debug
from raganything import RAGAnything, RAGAnythingConfig

# Load environment variables
load_dotenv(dotenv_path=".env", override=False)


def configure_logging():
    """Configure logging for the application"""
    log_dir = os.getenv("LOG_DIR", os.getcwd())
    log_file_path = os.path.abspath(os.path.join(log_dir, "process_upload.log"))

    print(f"\nProcessing log file: {log_file_path}\n")
    os.makedirs(log_dir, exist_ok=True)

    log_max_bytes = int(os.getenv("LOG_MAX_BYTES", 10485760))  # Default 10MB
    log_backup_count = int(os.getenv("LOG_BACKUP_COUNT", 5))  # Default 5 backups

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.handlers.RotatingFileHandler(
                log_file_path,
                maxBytes=log_max_bytes,
                backupCount=log_backup_count,
                encoding="utf-8",
            ),
        ],
    )

    logger.setLevel(logging.INFO)
    set_verbose_debug(os.getenv("VERBOSE", "false").lower() == "true")


async def check_ollama_embedding_available(embedding_base_url: str, embedding_model: str) -> bool:
    """
    Check if Ollama embedding service and model are available.
    
    Args:
        embedding_base_url: Base URL for Ollama (with /v1)
        embedding_model: Name of the embedding model to check
        
    Returns:
        True if available, False otherwise
    """
    try:
        # Remove /v1 from base_url to get Ollama base URL
        ollama_base = embedding_base_url.replace("/v1", "").rstrip("/")
        
        # Check 1: Verify Ollama service is accessible (if httpx is available)
        if HAS_HTTPX:
            logger.info(f"Checking Ollama service at {ollama_base}...")
            try:
                async with httpx.AsyncClient(timeout=10.0) as client:
                    try:
                        response = await client.get(f"{ollama_base}/api/tags")
                        if response.status_code != 200:
                            logger.error(f"Ollama service returned status {response.status_code}")
                            return False
                    except httpx.RequestError as e:
                        logger.error(f"Cannot connect to Ollama service at {ollama_base}: {str(e)}")
                        logger.error("Please ensure Ollama is running: ollama serve")
                        return False
                    
                    # Check 2: List available models
                    try:
                        response = await client.get(f"{ollama_base}/api/tags")
                        models_data = response.json()
                        available_models = [model.get("name", "") for model in models_data.get("models", [])]
                        
                        # Check if embedding model is in the list
                        # Handle both "model:tag" and "model" formats
                        model_found = False
                        for model_name in available_models:
                            # Check exact match or if model name starts with embedding_model
                            if model_name == embedding_model or model_name.startswith(embedding_model + ":"):
                                model_found = True
                                logger.info(f"✅ Found embedding model: {model_name}")
                                break
                        
                        if not model_found:
                            logger.error(f"❌ Embedding model '{embedding_model}' not found in Ollama")
                            logger.error(f"Available models: {', '.join(available_models) if available_models else 'None'}")
                            logger.error(f"Please pull the model: ollama pull {embedding_model}")
                            return False
                        
                    except Exception as e:
                        logger.warning(f"Could not list models, will try test embedding: {str(e)}")
                        # Continue to test embedding as fallback
            except Exception as e:
                logger.warning(f"Could not check Ollama service (httpx error), will try test embedding: {str(e)}")
        else:
            logger.info("Skipping service check (httpx not available), will test embedding directly...")
        
        # Check 3: Test embedding with a small text
        logger.info(f"Testing embedding with model '{embedding_model}'...")
        try:
            from lightrag.llm.ollama import ollama_embed
            test_texts = ["test"]
            result = await asyncio.wait_for(
                ollama_embed(
                    test_texts,
                    embedding_model,
                    host=ollama_base,
                ),
                timeout=30.0
            )
            # Handle both list and numpy array results
            if result is None:
                logger.error("❌ Embedding test returned None")
                return False
            
            # Convert to list if it's a numpy array (handle numpy import gracefully)
            try:
                import numpy as np
                if isinstance(result, np.ndarray):
                    result = result.tolist()
            except ImportError:
                pass  # numpy not available, assume it's already a list
            
            # Check if we have at least one embedding
            try:
                result_len = len(result)
                if result_len == 0:
                    logger.error("❌ Embedding test returned empty result")
                    return False
                
                # Check the first embedding dimension
                first_embedding = result[0]
                try:
                    import numpy as np
                    if isinstance(first_embedding, np.ndarray):
                        first_embedding = first_embedding.tolist()
                except ImportError:
                    pass
                
                embedding_dim = len(first_embedding)
                if embedding_dim == 0:
                    logger.error("❌ Embedding test returned empty embedding vector")
                    return False
                
                logger.info(f"✅ Embedding test successful (dimension: {embedding_dim})")
                return True
            except (TypeError, ValueError) as e:
                logger.error(f"❌ Embedding test returned invalid format: {str(e)}")
                return False
        except asyncio.TimeoutError:
            logger.error("❌ Embedding test timed out")
            logger.error("The embedding model may be too slow or not responding")
            return False
        except Exception as e:
            logger.error(f"❌ Embedding test failed: {str(e)}")
            logger.error(f"Please ensure the model '{embedding_model}' is pulled: ollama pull {embedding_model}")
            return False
            
    except Exception as e:
        logger.error(f"Error checking Ollama embedding availability: {str(e)}")
        return False


async def process_upload_folder():
    """
    Process all files in docs/upload folder into RAG-Anything database
    REQUIRES: Ollama must be configured. Script will exit if Ollama is not set.
    """
    # Get configuration from environment variables - NO FALLBACKS, Ollama only
    base_url = os.getenv("LLM_BINDING_HOST")
    llm_binding = os.getenv("LLM_BINDING", "").lower()
    
    # Embedding-specific configuration (separate from LLM)
    embedding_base_url = os.getenv("EMBEDDING_BINDING_HOST")
    embedding_binding = os.getenv("EMBEDDING_BINDING", "").lower()
    
    # Validate Ollama configuration - EXIT if not properly configured
    is_ollama_llm = False
    is_ollama_embedding = False
    
    # Check LLM binding
    if llm_binding == "ollama":
        is_ollama_llm = True
    elif base_url:
        # Check if base_url points to Ollama
        ollama_indicators = ["localhost:11434", "127.0.0.1:11434", ":11434"]
        is_ollama_llm = any(indicator in base_url for indicator in ollama_indicators)
    
    # Check embedding binding
    if embedding_binding == "ollama":
        is_ollama_embedding = True
    elif embedding_base_url:
        # Check if embedding_base_url points to Ollama
        ollama_indicators = ["localhost:11434", "127.0.0.1:11434", ":11434"]
        is_ollama_embedding = any(indicator in embedding_base_url for indicator in ollama_indicators)
    
    # If embedding_base_url not set, use LLM base_url
    if not embedding_base_url:
        embedding_base_url = base_url
        is_ollama_embedding = is_ollama_llm
    
    # EXIT if Ollama is not configured for LLM
    if not is_ollama_llm:
        logger.error("ERROR: Ollama is required for LLM but not configured.")
        logger.error("Please set LLM_BINDING=ollama or LLM_BINDING_HOST to an Ollama endpoint (e.g., http://localhost:11434)")
        logger.error("Script will now exit.")
        sys.exit(1)
    
    # EXIT if Ollama is not configured for embeddings
    if not is_ollama_embedding:
        logger.error("ERROR: Ollama is required for embeddings but not configured.")
        logger.error("Please set EMBEDDING_BINDING=ollama or EMBEDDING_BINDING_HOST to an Ollama endpoint (e.g., http://localhost:11434)")
        logger.error("Script will now exit.")
        sys.exit(1)
    
    # Ensure base URL is set (required for Ollama)
    if not base_url:
        logger.error("ERROR: LLM_BINDING_HOST must be set for Ollama (e.g., http://localhost:11434)")
        logger.error("Script will now exit.")
        sys.exit(1)
    
    # Ensure Ollama LLM endpoint uses /v1 for OpenAI-compatible API
    base_url = base_url.rstrip('/')
    if not base_url.endswith('/v1'):
        base_url = f"{base_url}/v1"
    
    # Ensure Ollama embedding endpoint uses /v1 for OpenAI-compatible API
    embedding_base_url = embedding_base_url.rstrip('/')
    if not embedding_base_url.endswith('/v1'):
        embedding_base_url = f"{embedding_base_url}/v1"
    
    # Set dummy OPENAI_API_KEY for Ollama compatibility
    # The lightrag library's openai_complete_if_cache function tries to read
    # OPENAI_API_KEY from environment even when api_key=None is passed.
    # Ollama doesn't use this key, but we need to set it to avoid KeyError.
    if "OPENAI_API_KEY" not in os.environ:
        os.environ["OPENAI_API_KEY"] = "ollama"  # Dummy value, not used by Ollama
    
    working_dir = os.getenv("WORKING_DIR", "./rag_storage")
    output_dir = os.getenv("OUTPUT_DIR", "./output")
    parser = os.getenv("PARSER", "mineru")
    parse_method = os.getenv("PARSE_METHOD", "auto")
    
    default_llm_model = os.getenv("LLM_MODEL", "llama3.1:8b")
    default_vision_model = os.getenv("VISION_MODEL", default_llm_model)  # Use same model for vision
    default_embedding_model = os.getenv("EMBEDDING_MODEL", "nomic-embed-text")
    default_embedding_dim = int(os.getenv("EMBEDDING_DIM", "768"))
    
    # Check if Ollama embedding is available before proceeding
    logger.info("\n" + "="*60)
    logger.info("Checking Ollama Embedding Availability")
    logger.info("="*60)
    embedding_available = await check_ollama_embedding_available(
        embedding_base_url, 
        default_embedding_model
    )
    if not embedding_available:
        logger.error("\n" + "="*60)
        logger.error("ERROR: Ollama embedding is not available!")
        logger.error("="*60)
        logger.error("Please ensure:")
        logger.error(f"  1. Ollama is running: ollama serve")
        logger.error(f"  2. Embedding model is pulled: ollama pull {default_embedding_model}")
        logger.error(f"  3. Ollama is accessible at: {embedding_base_url.replace('/v1', '')}")
        logger.error("\nScript will now exit.")
        sys.exit(1)
    logger.info("="*60 + "\n")
    
    # Embedding timeout configuration (Ollama embeddings can be slower)
    embedding_timeout = os.getenv("EMBEDDING_TIMEOUT")
    if embedding_timeout:
        embedding_timeout = int(embedding_timeout)
    else:
        # Default to 300 seconds (5 minutes) for Ollama embeddings
        embedding_timeout = 300
    
    # Embedding batch and concurrency configuration for Ollama
    # Use smaller batches to avoid timeouts
    default_batch_num = 8  # Smaller batches for Ollama
    default_max_async = 4  # Lower concurrency for Ollama
    
    embedding_batch_num = int(os.getenv("EMBEDDING_BATCH_NUM", str(default_batch_num)))
    embedding_func_max_async = int(os.getenv("EMBEDDING_FUNC_MAX_ASYNC", str(default_max_async)))

    # Define paths
    upload_folder = Path(__file__).parent / "docs" / "upload"
    
    if not upload_folder.exists():
        logger.error(f"Upload folder not found: {upload_folder}")
        return False

    # Check if folder has any files
    files_in_folder = list(upload_folder.iterdir())
    if not files_in_folder:
        logger.warning(f"No files found in {upload_folder}")
        return False

    logger.info(f"Found {len(files_in_folder)} items in upload folder")
    for item in files_in_folder:
        logger.info(f"  - {item.name} ({'file' if item.is_file() else 'directory'})")

    # Create RAGAnything configuration
    config = RAGAnythingConfig(
        working_dir=working_dir,
        parser=parser,
        parse_method=parse_method,
        enable_image_processing=True,
        enable_table_processing=True,
        enable_equation_processing=True,
    )

    # Define LLM model function (Ollama only)
    def llm_model_func(prompt, system_prompt=None, history_messages=[], **kwargs):
        return openai_complete_if_cache(
            default_llm_model,
            prompt,
            system_prompt=system_prompt,
            history_messages=history_messages,
            api_key="ollama",  # Dummy value for Ollama (not actually used)
            base_url=base_url,
            **kwargs,
        )

    # Define vision model function for image processing (Ollama only)
    def vision_model_func(
        prompt, system_prompt=None, history_messages=[], image_data=None, messages=None, **kwargs
    ):
        # If messages format is provided (for multimodal VLM enhanced query), use it directly
        if messages:
            return openai_complete_if_cache(
                default_vision_model,
                "",
                system_prompt=None,
                history_messages=[],
                messages=messages,
                api_key="ollama",  # Dummy value for Ollama (not actually used)
                base_url=base_url,
                **kwargs,
            )
        # Traditional single image format
        elif image_data:
            return openai_complete_if_cache(
                default_vision_model,
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
                api_key="ollama",  # Dummy value for Ollama (not actually used)
                base_url=base_url,
                **kwargs,
            )
        # Pure text format
        else:
            return llm_model_func(prompt, system_prompt, history_messages, **kwargs)

    # Define embedding function with timeout handling for Ollama
    # Uses embedding-specific base_url (Ollama only)
    async def embedding_func_with_timeout(texts):
        """Wrapper for embedding function with timeout handling"""
        import asyncio
        try:
            # Use asyncio.wait_for to add timeout protection
            from lightrag.llm.ollama import ollama_embed
            result = await asyncio.wait_for(
                ollama_embed(
                    texts,
                    default_embedding_model,  # embed_model as positional argument
                    host=embedding_base_url.replace("/v1", ""),  # remove /v1 for ollama
                ),
                timeout=embedding_timeout
            )
            return result
        except asyncio.TimeoutError:
            logger.error(f"Embedding request timed out after {embedding_timeout} seconds")
            raise
        except Exception as e:
            logger.error(f"Embedding error: {str(e)}")
            raise
    
    embedding_func = EmbeddingFunc(
        embedding_dim=default_embedding_dim,
        max_token_size=8192,
        func=embedding_func_with_timeout,
    )

    # Initialize RAGAnything
    logger.info("Initializing RAG-Anything with Ollama...")
    logger.info(f"  Working directory: {working_dir}")
    logger.info(f"  Output directory: {output_dir}")
    logger.info(f"  Parser: {parser}")
    logger.info(f"  Parse method: {parse_method}")
    logger.info(f"  LLM Model: {default_llm_model}")
    logger.info(f"  Vision Model: {default_vision_model}")
    logger.info(f"  Embedding Model: {default_embedding_model} (dim: {default_embedding_dim})")
    logger.info(f"  Embedding Timeout: {embedding_timeout}s")
    logger.info(f"  Embedding Batch Size: {embedding_batch_num}")
    logger.info(f"  Embedding Max Async: {embedding_func_max_async}")
    logger.info(f"  LLM Base URL: {base_url} (Ollama)")
    logger.info(f"  Embedding Base URL: {embedding_base_url} (Ollama)")
    
    # Configure lightrag_kwargs for embedding timeout and concurrency
    lightrag_kwargs = {
        "embedding_batch_num": embedding_batch_num,
        "embedding_func_max_async": embedding_func_max_async,
    }
    
    # Add timeout configuration if supported by lightrag
    # Note: The actual timeout is handled in the embedding function wrapper above
    # but we can also configure it here if lightrag supports it
    
    rag = RAGAnything(
        config=config,
        llm_model_func=llm_model_func,
        vision_model_func=vision_model_func,
        embedding_func=embedding_func,
        lightrag_kwargs=lightrag_kwargs,
    )

    # Helper function to check if error is retryable
    def is_retryable_error(error: Exception) -> bool:
        """Determine if an error is retryable (e.g., embedding errors, connection issues)"""
        error_str = str(error).lower()
        error_type = type(error).__name__
        
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
        
        if any(pattern in error_str for pattern in retryable_patterns):
            return True
        
        if any(retry_type in error_type for retry_type in retryable_types):
            return True
        
        return False

    # Process all files in the upload folder with retry logic
    logger.info(f"\n{'='*60}")
    logger.info(f"Processing files in: {upload_folder}")
    logger.info(f"{'='*60}\n")

    # Get retry configuration
    max_retries = int(os.getenv("PROCESSING_MAX_RETRIES", "3"))
    retry_delay_base = float(os.getenv("PROCESSING_RETRY_DELAY_BASE", "2.0"))  # Base delay in seconds
    
    last_error = None
    
    try:
        for attempt in range(max_retries):
            try:
                if attempt > 0:
                    logger.info(f"\n{'='*60}")
                    logger.info(f"RETRY ATTEMPT {attempt + 1}/{max_retries}")
                    logger.info(f"{'='*60}\n")
                    # Wait before retrying with exponential backoff
                    retry_delay = retry_delay_base * (2 ** (attempt - 1))
                    logger.info(f"Waiting {retry_delay:.1f} seconds before retry...")
                    await asyncio.sleep(retry_delay)
                
                await rag.process_folder_complete(
                    folder_path=str(upload_folder),
                    output_dir=output_dir,
                    parse_method=parse_method,
                    display_stats=True,
                    recursive=True,  # Process subdirectories if any
                    max_workers=int(os.getenv("MAX_CONCURRENT_FILES", "1")),  # Process files concurrently
                )

                logger.info(f"\n{'='*60}")
                logger.info("✅ Successfully processed all files!" + (f" (after {attempt} retries)" if attempt > 0 else ""))
                logger.info(f"{'='*60}\n")

                # Test query to verify processing
                logger.info("Testing database with a sample query...")
                test_query = "What is the main content of the documents?"
                try:
                    result = await rag.aquery(test_query, mode="hybrid")
                    logger.info(f"Query: {test_query}")
                    logger.info(f"Answer: {result}\n")
                except Exception as query_error:
                    logger.warning(f"Test query failed (this is okay if API is not fully configured): {str(query_error)}")
                    logger.info("Document parsing completed successfully. You can query the database later once API is configured.")

                return True

            except Exception as e:
                last_error = e
                error_msg = str(e)
                logger.error(f"\n{'='*60}")
                logger.error(f"Error processing files (attempt {attempt + 1}/{max_retries}): {error_msg}")
                logger.error(f"{'='*60}")
                import traceback
                logger.error(traceback.format_exc())
                
                # Check if error is retryable
                is_retryable = is_retryable_error(e)
                is_last_attempt = (attempt + 1) >= max_retries
                
                if is_retryable and not is_last_attempt:
                    logger.warning(f"Retryable error detected. Will retry processing from the beginning...")
                    continue  # Retry
                else:
                    # Non-retryable error or last attempt - fail permanently
                    logger.error(f"\n{'='*60}")
                    logger.error("❌ Processing failed permanently after all retry attempts")
                    logger.error(f"{'='*60}\n")
                    return False
        
        # Should not reach here, but just in case
        if last_error:
            logger.error(f"Processing failed after {max_retries} attempts: {str(last_error)}")
        return False
    finally:
        # Finalize storages before event loop closes to avoid warnings
        try:
            await rag.finalize_storages()
            # Unregister atexit handler to prevent duplicate finalization attempts
            # after event loop closes
            import atexit
            try:
                atexit.unregister(rag.close)
            except (ValueError, AttributeError):
                # Handler might not be registered or unregister might not be available
                pass
        except Exception as finalize_error:
            # Log but don't fail if finalization has issues
            logger.warning(f"Warning during storage finalization: {str(finalize_error)}")


def main():
    """Main function"""
    configure_logging()

    print("=" * 60)
    print("RAG-Anything: Process Upload Folder")
    print("=" * 60)
    print("This script processes all files in docs/upload folder")
    print("and inserts them into the RAG-Anything knowledge base.")
    print("=" * 60)
    print()

    success = asyncio.run(process_upload_folder())

    if success:
        print("\n✅ Processing completed successfully!")
        print(f"📁 Files processed from: docs/upload/")
        print(f"💾 Database location: {os.getenv('WORKING_DIR', './rag_storage')}")
        print(f"📤 Output location: {os.getenv('OUTPUT_DIR', './output')}")
    else:
        print("\n❌ Processing failed. Check the logs for details.")
        sys.exit(1)


if __name__ == "__main__":
    main()

