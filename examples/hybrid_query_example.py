#!/usr/bin/env python
"""
Example script demonstrating hybrid query mode, multi-query, and multi-hop querying

This example shows how to:
1. Load existing RAGAnything database
2. Use hybrid mode for comparison queries
3. Extract specific data from tables using hybrid mode
4. Perform both text and multimodal queries with hybrid mode
5. Run multiple queries sequentially (multi-query)
6. Chain queries together where later queries use results from earlier ones (multi-hop)
7. Decompose complex queries into simpler sub-queries
8. Query across multiple documents/files in the database

Note: This script only queries existing data - no document processing/parsing.
Make sure you have already processed documents into the database.
"""

import os
import argparse
import asyncio
from pathlib import Path
from dotenv import load_dotenv

# Add project root directory to Python path
import sys
sys.path.append(str(Path(__file__).parent.parent))

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
        parser="mineru",  # or "docling" (not used for querying, but required)
        parse_method="auto",
        enable_image_processing=True,
        enable_table_processing=True,
        enable_equation_processing=True,
    )

    # Define LLM model function
    # Use environment variable for model name, default to llama3.1:8b (Ollama)
    llm_model_name = os.getenv("LLM_MODEL", "llama3.1:8b")
    
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
    # Use llama3.1:8b for vision as well (Ollama models can handle text prompts)
    def vision_model_func(
        prompt, system_prompt=None, history_messages=[], image_data=None, messages=None, **kwargs
    ):
        # Use same LLM model for vision (Ollama models)
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
    # Use Ollama nomic-embed-text model (768 dimensions)
    # This matches the existing database dimension
    embedding_dim = int(os.getenv("EMBEDDING_DIM", "768"))
    embedding_model = os.getenv("EMBEDDING_MODEL", "nomic-embed-text")
    
    # Use Ollama embedding endpoint if base_url is provided and points to Ollama
    # Ollama embeddings use /api/embeddings endpoint (OpenAI-compatible)
    embedding_base_url = base_url
    if base_url and "localhost:11434" in base_url or "127.0.0.1:11434" in base_url:
        # Ensure we're using the correct endpoint for Ollama embeddings
        # Ollama uses OpenAI-compatible API, so base_url should work
        pass
    
    embedding_func = EmbeddingFunc(
        embedding_dim=embedding_dim,
        max_token_size=8192,
        func=lambda texts: openai_embed(
            texts,
            model=embedding_model,
            api_key=api_key,
            base_url=embedding_base_url,
        ),
    )

    # Initialize RAGAnything (will load existing database)
    rag = RAGAnything(
        config=config,
        llm_model_func=llm_model_func,
        vision_model_func=vision_model_func,
        embedding_func=embedding_func,
    )

    # IMPORTANT: Initialize LightRAG instance from existing storage
    print("Initializing LightRAG from existing storage...")
    init_result = await rag._ensure_lightrag_initialized()
    if not init_result.get("success", False):
        raise RuntimeError(f"Failed to initialize LightRAG: {init_result.get('error', 'Unknown error')}")

    return rag


async def example_comparison_queries(rag: RAGAnything):
    """
    Demonstrate comparison queries using hybrid mode
    
    Hybrid mode is ideal for comparisons because it:
    - Retrieves entity-specific information (for details)
    - Retrieves relationships between entities (for comparisons)
    """
    print("\n" + "="*80)
    print("COMPARISON QUERIES WITH HYBRID MODE")
    print("="*80)
    
    comparison_queries = [
        "Compare the performance metrics between different methods mentioned in the document",
        "What are the differences between the approaches discussed in the tables?",
        "Compare the accuracy and speed metrics across all methods in the performance tables",
        "What are the key differences between the techniques described in the document?",
    ]
    
    for i, query in enumerate(comparison_queries, 1):
        print(f"\n[Query {i}]: {query}")
        print("-" * 80)
        try:
            result = await rag.aquery(query, mode="hybrid")
            if result and isinstance(result, str):
                print(f"Answer:\n{result}\n")
            else:
                print(f"Answer: {result}\n")
        except Exception as e:
            error_msg = str(e) if e else "Unknown error"
            print(f"Error: {error_msg}\n")
            import traceback
            print(f"Traceback: {traceback.format_exc()}\n")


async def example_table_data_extraction(rag: RAGAnything):
    """
    Demonstrate extracting specific data from tables using hybrid mode
    
    Hybrid mode works well for table data extraction because it:
    - Gets precise entity information (specific table values)
    - Understands relationships (how data points relate to each other)
    """
    print("\n" + "="*80)
    print("TABLE DATA EXTRACTION WITH HYBRID MODE")
    print("="*80)
    
    extraction_queries = [
        "What are the exact accuracy values for each method in the performance tables?",
        "Extract all the numerical values from the comparison tables in the document",
        "What are the specific metrics (accuracy, F1-score, speed) for each method mentioned in tables?",
        "List all the data points from the tables with their corresponding values",
    ]
    
    for i, query in enumerate(extraction_queries, 1):
        print(f"\n[Query {i}]: {query}")
        print("-" * 80)
        try:
            result = await rag.aquery(query, mode="hybrid")
            print(f"Answer:\n{result}\n")
        except Exception as e:
            print(f"Error: {str(e)}\n")


async def example_multimodal_table_query(rag: RAGAnything):
    """
    Demonstrate querying with specific table content using hybrid mode
    
    This shows how to provide table data directly in the query
    """
    print("\n" + "="*80)
    print("MULTIMODAL TABLE QUERIES WITH HYBRID MODE")
    print("="*80)
    
    # Example table data
    table_content = {
        "type": "table",
        "table_data": """Method,Accuracy,F1-Score,Speed
RAGAnything,95.2%,0.94,120ms
Traditional RAG,87.3%,0.85,180ms
Baseline,82.1%,0.78,200ms""",
        "table_caption": "Performance Comparison"
    }
    
    multimodal_queries = [
        "Compare the performance metrics in this table and extract the specific values",
        "What are the differences between RAGAnything and Traditional RAG based on this table?",
        "Analyze the data trends and relationships in this performance table",
    ]
    
    for i, query in enumerate(multimodal_queries, 1):
        print(f"\n[Query {i}]: {query}")
        print("-" * 80)
        try:
            result = await rag.aquery_with_multimodal(
                query,
                multimodal_content=[table_content],
                mode="hybrid"
            )
            print(f"Answer:\n{result}\n")
        except Exception as e:
            print(f"Error: {str(e)}\n")


async def example_combined_comparison_and_extraction(rag: RAGAnything):
    """
    Demonstrate queries that combine comparison and data extraction
    """
    print("\n" + "="*80)
    print("COMBINED COMPARISON AND DATA EXTRACTION QUERIES")
    print("="*80)
    
    combined_queries = [
        "Compare all methods in the tables and extract their exact performance metrics",
        "What are the differences between the top-performing methods, and what are their specific values?",
        "Extract and compare the accuracy, F1-score, and speed metrics for all methods mentioned in tables",
    ]
    
    for i, query in enumerate(combined_queries, 1):
        print(f"\n[Query {i}]: {query}")
        print("-" * 80)
        try:
            result = await rag.aquery(query, mode="hybrid")
            print(f"Answer:\n{result}\n")
        except Exception as e:
            print(f"Error: {str(e)}\n")


async def example_multi_query_sequential(rag: RAGAnything):
    """
    Demonstrate running multiple queries sequentially
    
    Multi-query allows you to:
    - Break down complex questions into simpler sub-queries
    - Query different aspects of the database
    - Gather comprehensive information across multiple topics
    """
    print("\n" + "="*80)
    print("MULTI-QUERY: SEQUENTIAL QUERIES")
    print("="*80)
    
    # Define a sequence of related queries
    query_sequence = [
        "What are the main topics or subjects discussed across all documents in the database?",
        "What methods or approaches are mentioned in the documents?",
        "What are the key findings or results presented?",
        "What tables or performance metrics are included in the documents?",
        "What conclusions or recommendations are made?",
    ]
    
    results = {}
    
    for i, query in enumerate(query_sequence, 1):
        print(f"\n[Query {i}/{len(query_sequence)}]: {query}")
        print("-" * 80)
        try:
            result = await rag.aquery(query, mode="hybrid")
            results[f"query_{i}"] = {
                "query": query,
                "result": result
            }
            print(f"Answer:\n{result}\n")
        except Exception as e:
            print(f"Error: {str(e)}\n")
            results[f"query_{i}"] = {
                "query": query,
                "error": str(e)
            }
    
    # Summary of all queries
    print("\n" + "="*80)
    print("MULTI-QUERY SUMMARY")
    print("="*80)
    print(f"Completed {len([r for r in results.values() if 'result' in r])}/{len(query_sequence)} queries successfully")
    return results


async def example_multi_hop_query_chaining(rag: RAGAnything):
    """
    Demonstrate multi-hop query chaining where later queries use results from earlier queries
    
    Multi-hop querying allows you to:
    - Use information from one query to inform the next
    - Build up understanding progressively
    - Answer complex questions that require multiple steps
    """
    print("\n" + "="*80)
    print("MULTI-HOP QUERY CHAINING")
    print("="*80)
    
    # Step 1: Identify main entities/topics
    print("\n[Hop 1]: Identifying main entities and topics...")
    print("-" * 80)
    try:
        hop1_result = await rag.aquery(
            "What are the main entities, methods, or techniques mentioned across all documents? List them clearly.",
            mode="hybrid"
        )
        print(f"Result:\n{hop1_result}\n")
        
        # Step 2: Use the entities from hop1 to ask more specific questions
        print("\n[Hop 2]: Getting detailed information about the identified entities...")
        print("-" * 80)
        hop2_query = f"Based on the entities and methods identified ({hop1_result[:200]}...), what are the specific details, performance metrics, or characteristics of each?"
        hop2_result = await rag.aquery(hop2_query, mode="hybrid")
        print(f"Result:\n{hop2_result}\n")
        
        # Step 3: Compare or analyze relationships
        print("\n[Hop 3]: Analyzing relationships and comparisons...")
        print("-" * 80)
        hop3_query = f"Based on the information gathered about the entities ({hop2_result[:200]}...), what are the relationships, comparisons, or differences between them?"
        hop3_result = await rag.aquery(hop3_query, mode="hybrid")
        print(f"Result:\n{hop3_result}\n")
        
        # Step 4: Synthesize final answer
        print("\n[Hop 4]: Synthesizing comprehensive answer...")
        print("-" * 80)
        hop4_query = f"Based on all the information gathered:\n1. Entities: {hop1_result[:300]}...\n2. Details: {hop2_result[:300]}...\n3. Relationships: {hop3_result[:300]}...\n\nProvide a comprehensive summary that ties everything together."
        hop4_result = await rag.aquery(hop4_query, mode="hybrid")
        print(f"Final Synthesis:\n{hop4_result}\n")
        
        return {
            "hop1": hop1_result,
            "hop2": hop2_result,
            "hop3": hop3_result,
            "hop4": hop4_result
        }
        
    except Exception as e:
        print(f"Error in multi-hop query: {str(e)}\n")
        return None


async def example_multi_query_decomposition(rag: RAGAnything):
    """
    Demonstrate breaking down a complex query into multiple simpler queries
    
    Query decomposition allows you to:
    - Answer complex questions by breaking them into parts
    - Query different aspects systematically
    - Combine results for comprehensive answers
    """
    print("\n" + "="*80)
    print("MULTI-QUERY DECOMPOSITION")
    print("="*80)
    
    # Complex question that needs decomposition
    complex_question = "What are all the performance metrics, methods, and results mentioned in the documents, and how do they compare?"
    
    print(f"\nComplex Question: {complex_question}")
    print("\nDecomposing into sub-queries...")
    
    # Decompose into sub-queries
    sub_queries = [
        "What methods or approaches are mentioned in the documents?",
        "What performance metrics or evaluation criteria are used?",
        "What are the specific numerical results or values for these metrics?",
        "What comparisons are made between different methods?",
        "What are the key differences or advantages mentioned?",
    ]
    
    sub_results = {}
    
    for i, sub_query in enumerate(sub_queries, 1):
        print(f"\n[Sub-query {i}]: {sub_query}")
        print("-" * 80)
        try:
            result = await rag.aquery(sub_query, mode="hybrid")
            sub_results[f"sub_query_{i}"] = {
                "query": sub_query,
                "result": result
            }
            print(f"Answer:\n{result[:500]}...\n")
        except Exception as e:
            print(f"Error: {str(e)}\n")
            sub_results[f"sub_query_{i}"] = {
                "query": sub_query,
                "error": str(e)
            }
    
    # Now query to synthesize the results
    print("\n[Synthesis Query]: Combining all sub-query results...")
    print("-" * 80)
    
    # Build synthesis prompt parts separately to avoid f-string backslash issues
    query_parts = []
    for i, (k, r) in enumerate(sub_results.items()):
        query_text = r['query']
        answer_text = r.get('result', r.get('error', 'N/A'))[:300]
        query_parts.append(f"Query {i+1}: {query_text}\nAnswer: {answer_text}...")
    
    query_summary = "\n\n".join(query_parts)
    synthesis_prompt = f"""Based on the following information gathered from multiple queries:

{query_summary}

Please provide a comprehensive answer to the original complex question: "{complex_question}"
"""
    
    try:
        synthesis_result = await rag.aquery(synthesis_prompt, mode="hybrid")
        print(f"\nComprehensive Answer:\n{synthesis_result}\n")
        return {
            "sub_queries": sub_results,
            "synthesis": synthesis_result
        }
    except Exception as e:
        print(f"Error in synthesis: {str(e)}\n")
        return {"sub_queries": sub_results}


async def example_cross_document_querying(rag: RAGAnything):
    """
    Demonstrate querying across multiple documents/files in the database
    
    Cross-document querying allows you to:
    - Query information across all documents in rag_storage
    - Find relationships between different documents
    - Aggregate information from multiple sources
    """
    print("\n" + "="*80)
    print("CROSS-DOCUMENT MULTI-QUERY")
    print("="*80)
    
    cross_document_queries = [
        "What documents or files are stored in this database? List their main topics or subjects.",
        "What are the common themes or topics across all documents?",
        "Are there any contradictions or conflicting information between different documents?",
        "What information appears in multiple documents? Summarize the overlapping content.",
        "What unique information does each document contribute?",
    ]
    
    for i, query in enumerate(cross_document_queries, 1):
        print(f"\n[Cross-Document Query {i}]: {query}")
        print("-" * 80)
        try:
            result = await rag.aquery(query, mode="hybrid")
            print(f"Answer:\n{result}\n")
        except Exception as e:
            print(f"Error: {str(e)}\n")


async def main():
    """Main function to run query examples"""
    parser = argparse.ArgumentParser(
        description="Hybrid query mode examples - queries existing database only (no document processing)"
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
        "--query",
        type=str,
        help="Single query to run (optional - if not provided, runs example queries)",
    )
    parser.add_argument(
        "--multi-query",
        action="store_true",
        help="Run multi-query examples (sequential queries)",
    )
    parser.add_argument(
        "--multi-hop",
        action="store_true",
        help="Run multi-hop query chaining examples",
    )
    parser.add_argument(
        "--decomposition",
        action="store_true",
        help="Run query decomposition examples",
    )
    parser.add_argument(
        "--cross-document",
        action="store_true",
        help="Run cross-document querying examples",
    )
    
    args = parser.parse_args()
    
    if not args.api_key:
        print("Error: API key is required. Set OPENAI_API_KEY or LLM_BINDING_API_KEY env var, or use --api-key")
        return
    
    # Set up RAG instance (loads existing database)
    print("Loading RAGAnything database...")
    print(f"Working directory: {args.working_dir}")
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
    
    # Run single query if provided
    if args.query:
        print("="*80)
        print("RUNNING SINGLE QUERY")
        print("="*80)
        print(f"\nQuery: {args.query}")
        print("-" * 80)
        try:
            result = await rag.aquery(args.query, mode="hybrid", vlm_enhanced=False)
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
    # Run specific example types if requested
    elif args.multi_query or args.multi_hop or args.decomposition or args.cross_document:
        try:
            if args.multi_query:
                await example_multi_query_sequential(rag)
            if args.multi_hop:
                await example_multi_hop_query_chaining(rag)
            if args.decomposition:
                await example_multi_query_decomposition(rag)
            if args.cross_document:
                await example_cross_document_querying(rag)
            
            print("\n" + "="*80)
            print("Selected examples completed!")
            print("="*80)
        except Exception as e:
            print(f"\nError running queries: {str(e)}")
            print("\nMake sure you have processed documents into the database first.")
            print("Use raganything_example.py or another script to process documents.")
    else:
        # Run all query examples
        try:
            # Basic hybrid query examples
            await example_comparison_queries(rag)
            await example_table_data_extraction(rag)
            await example_multimodal_table_query(rag)
            await example_combined_comparison_and_extraction(rag)
            
            # Multi-query and multi-hop examples
            await example_multi_query_sequential(rag)
            await example_multi_hop_query_chaining(rag)
            await example_multi_query_decomposition(rag)
            await example_cross_document_querying(rag)
            
            print("\n" + "="*80)
            print("All examples completed!")
            print("="*80)
            print("\nKey Takeaways:")
            print("- Hybrid mode is ideal for comparison queries (combines entity + relationship retrieval)")
            print("- Hybrid mode works well for table data extraction (gets specific values + relationships)")
            print("- Use aquery() for text queries, aquery_with_multimodal() for queries with specific table content")
            print("- Multi-query allows running multiple sequential queries to gather comprehensive information")
            print("- Multi-hop querying chains queries together, using results from earlier queries to inform later ones")
            print("- Query decomposition breaks complex questions into simpler sub-queries for better answers")
            print("- Cross-document querying enables querying across all files in the rag_storage database")
            print("\nUsage examples:")
            print("  # Run a single query:")
            print("  python examples/hybrid_query_example.py --query 'Your question here'")
            print("  # Run only multi-query examples:")
            print("  python examples/hybrid_query_example.py --multi-query")
            print("  # Run only multi-hop examples:")
            print("  python examples/hybrid_query_example.py --multi-hop")
            print("  # Run only decomposition examples:")
            print("  python examples/hybrid_query_example.py --decomposition")
            print("  # Run only cross-document examples:")
            print("  python examples/hybrid_query_example.py --cross-document")
            print("="*80)
            
        except Exception as e:
            print(f"\nError running queries: {str(e)}")
            print("\nMake sure you have processed documents into the database first.")
            print("Use raganything_example.py or another script to process documents.")


if __name__ == "__main__":
    asyncio.run(main())

