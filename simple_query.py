#!/usr/bin/env python
"""
Simple query script for RAGAnything

This script loads an existing RAG database and runs a query.
Usage: python simple_query.py "Your question here"
"""

import os
import asyncio
import argparse
from pathlib import Path
from dotenv import load_dotenv

# Add project root directory to Python path
import sys
sys.path.append(str(Path(__file__).parent))

from lightrag.llm.openai import openai_complete_if_cache, openai_embed
from lightrag.utils import EmbeddingFunc
from raganything import RAGAnything, RAGAnythingConfig

load_dotenv(dotenv_path=".env", override=False)


async def setup_rag(api_key: str, base_url: str = None, working_dir: str = "./rag_storage"):
    """
    Set up and initialize RAGAnything instance for querying existing database
    
    Args:
        api_key: API key for LLM services
        base_url: Optional base URL for API
        working_dir: Working directory where RAG storage exists
        
    Returns:
        RAGAnything instance
    """
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
    rag = RAGAnything(
        config=config,
        llm_model_func=llm_model_func,
        vision_model_func=vision_model_func,
        embedding_func=embedding_func,
    )

    # Initialize LightRAG instance from existing storage
    print("Initializing RAG database...")
    init_result = await rag._ensure_lightrag_initialized()
    if not init_result.get("success", False):
        raise RuntimeError(f"Failed to initialize RAG database: {init_result.get('error', 'Unknown error')}")

    return rag


async def main():
    """Main function to run query"""
    parser = argparse.ArgumentParser(
        description="Simple query script for RAGAnything - queries existing database"
    )
    parser.add_argument(
        "--query",
        type=str,
        default="What is the design phase of project Housing_Concrete?",
        help="Query to run (default: What is the design phase of project Housing_Concrete?)",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=os.getenv("OPENAI_API_KEY") or os.getenv("LLM_BINDING_API_KEY"),
        help="API key for LLM services (defaults to OPENAI_API_KEY or LLM_BINDING_API_KEY env var)",
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default=os.getenv("OPENAI_BASE_URL") or os.getenv("LLM_BINDING_HOST"),
        help="Base URL for API (defaults to OPENAI_BASE_URL or LLM_BINDING_HOST env var)",
    )
    parser.add_argument(
        "--working-dir",
        type=str,
        default="./rag_storage",
        help="Working directory where RAG storage exists (default: ./rag_storage)",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="hybrid",
        choices=["hybrid", "local", "global", "naive"],
        help="Query mode (default: hybrid)",
    )
    
    args = parser.parse_args()
    
    if not args.api_key:
        print("Error: API key is required. Set OPENAI_API_KEY or LLM_BINDING_API_KEY env var, or use --api-key")
        return
    
    # Set up RAG instance (loads existing database)
    print(f"Loading RAG database from: {args.working_dir}")
    try:
        rag = await setup_rag(
            api_key=args.api_key,
            base_url=args.base_url,
            working_dir=args.working_dir
        )
        print("Database loaded successfully!\n")
    except Exception as e:
        print(f"Error loading database: {str(e)}")
        print("\nMake sure:")
        print("1. You have processed documents into the database first")
        print("2. The working directory path is correct")
        print("3. The database files exist in the working directory")
        return
    
    # Run query
    print("="*80)
    print("QUERY")
    print("="*80)
    print(f"\nQuery: {args.query}")
    print(f"Mode: {args.mode}")
    print("-" * 80)
    try:
        result = await rag.aquery(args.query, mode=args.mode)
        if result and isinstance(result, str):
            print(f"\nAnswer:\n{result}\n")
        else:
            print(f"\nAnswer: {result}\n")
    except Exception as e:
        error_msg = str(e) if e else "Unknown error"
        print(f"Error: {error_msg}")
        print("\nMake sure you have processed documents into the database first.")
        print("Also check your API key and base URL configuration.")
        import traceback
        print(f"\nFull error details:\n{traceback.format_exc()}")


if __name__ == "__main__":
    asyncio.run(main())

