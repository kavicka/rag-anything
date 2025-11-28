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


async def process_upload_folder():
    """
    Process all files in docs/upload folder into RAG-Anything database
    """
    # Get configuration from environment variables
    api_key = os.getenv("LLM_BINDING_API_KEY") or os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("LLM_BINDING_HOST") or os.getenv("OPENAI_BASE_URL")
    working_dir = os.getenv("WORKING_DIR", "./rag_storage")
    output_dir = os.getenv("OUTPUT_DIR", "./output")
    parser = os.getenv("PARSER", "mineru")
    parse_method = os.getenv("PARSE_METHOD", "auto")
    
    # Check if API key is provided
    if not api_key:
        logger.error("Error: API key is required")
        logger.error("Please set LLM_BINDING_API_KEY or OPENAI_API_KEY environment variable")
        logger.error("Or create a .env file with your API key")
        return False

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

    # Define LLM model function
    def llm_model_func(prompt, system_prompt=None, history_messages=[], **kwargs):
        return openai_complete_if_cache(
            os.getenv("LLM_MODEL", "gpt-4o-mini"),
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
        # If messages format is provided (for multimodal VLM enhanced query), use it directly
        if messages:
            return openai_complete_if_cache(
                os.getenv("VISION_MODEL", "gpt-4o"),
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
                os.getenv("VISION_MODEL", "gpt-4o"),
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

    # Initialize RAGAnything
    logger.info("Initializing RAG-Anything...")
    logger.info(f"  Working directory: {working_dir}")
    logger.info(f"  Output directory: {output_dir}")
    logger.info(f"  Parser: {parser}")
    logger.info(f"  Parse method: {parse_method}")
    
    rag = RAGAnything(
        config=config,
        llm_model_func=llm_model_func,
        vision_model_func=vision_model_func,
        embedding_func=embedding_func,
    )

    # Process all files in the upload folder
    logger.info(f"\n{'='*60}")
    logger.info(f"Processing files in: {upload_folder}")
    logger.info(f"{'='*60}\n")

    try:
        await rag.process_folder_complete(
            folder_path=str(upload_folder),
            output_dir=output_dir,
            parse_method=parse_method,
            display_stats=True,
            recursive=True,  # Process subdirectories if any
            max_workers=int(os.getenv("MAX_CONCURRENT_FILES", "1")),  # Process files concurrently
        )

        logger.info(f"\n{'='*60}")
        logger.info("✅ Successfully processed all files!")
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
        logger.error(f"Error processing files: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return False


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

