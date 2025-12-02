#!/usr/bin/env python
"""
Diagnostic script to check RAG storage status

This script helps diagnose why queries return "No relevant content found"
by checking:
1. Storage directory path and existence
2. Storage files presence and size
3. Document count in storage
4. Configuration issues
"""

import os
import json
import sys
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv(dotenv_path=".env", override=False)

def check_storage():
    """Check storage status and diagnose issues"""
    print("=" * 80)
    print("RAG Storage Diagnostic Tool")
    print("=" * 80)
    print()
    
    # 1. Check working directory configuration
    working_dir = os.getenv("RAG_WORKING_DIR", "./rag_storage")
    working_dir_abs = os.path.abspath(working_dir)
    
    print("1. Storage Directory Configuration")
    print("-" * 80)
    print(f"   RAG_WORKING_DIR env var: {os.getenv('RAG_WORKING_DIR', '(not set, using default)')}")
    print(f"   Working directory: {working_dir}")
    print(f"   Absolute path: {working_dir_abs}")
    print(f"   Directory exists: {os.path.exists(working_dir_abs)}")
    print()
    
    if not os.path.exists(working_dir_abs):
        print("   ❌ ERROR: Storage directory does not exist!")
        print(f"   Please create it or set RAG_WORKING_DIR to the correct path.")
        print()
        return False
    
    # 2. Check storage files
    storage_path = Path(working_dir_abs)
    expected_files = {
        "kv_store_doc_status.json": "Document status (required)",
        "kv_store_text_chunks.json": "Text chunks (required)",
        "vdb_chunks.json": "Vector database chunks (required)",
        "kv_store_full_docs.json": "Full documents",
        "kv_store_entity_chunks.json": "Entity chunks",
        "kv_store_relation_chunks.json": "Relation chunks",
        "vdb_entities.json": "Entity vectors",
        "vdb_relationships.json": "Relationship vectors",
        "kv_store_full_entities.json": "Full entities",
        "kv_store_full_relations.json": "Full relations",
        "kv_store_llm_response_cache.json": "LLM cache",
        "kv_store_parse_cache.json": "Parse cache",
        "graph_chunk_entity_relation.graphml": "Knowledge graph"
    }
    
    print("2. Storage Files")
    print("-" * 80)
    found_files = []
    missing_files = []
    empty_files = []
    
    for file_name, description in expected_files.items():
        file_path = storage_path / file_name
        exists = file_path.exists()
        if exists:
            size = file_path.stat().st_size
            found_files.append(file_name)
            if size == 0:
                empty_files.append(file_name)
                print(f"   ⚠️  {file_name}: exists but EMPTY ({size} bytes) - {description}")
            else:
                print(f"   ✅ {file_name}: exists ({size:,} bytes) - {description}")
        else:
            missing_files.append(file_name)
            required = "required" in description.lower()
            symbol = "❌" if required else "⚠️"
            print(f"   {symbol} {file_name}: MISSING - {description}")
    
    print()
    print(f"   Summary: {len(found_files)}/{len(expected_files)} files found")
    if missing_files:
        required_missing = [f for f in missing_files if "required" in expected_files[f].lower()]
        if required_missing:
            print(f"   ❌ Missing required files: {', '.join(required_missing)}")
    if empty_files:
        print(f"   ⚠️  Empty files: {', '.join(empty_files)}")
    print()
    
    # 3. Check document status
    print("3. Document Status")
    print("-" * 80)
    doc_status_file = storage_path / "kv_store_doc_status.json"
    
    if doc_status_file.exists():
        try:
            with open(doc_status_file, 'r', encoding='utf-8') as f:
                doc_status_data = json.load(f)
            
            if isinstance(doc_status_data, dict):
                doc_count = len(doc_status_data)
                print(f"   ✅ Found {doc_count} documents in storage")
                
                if doc_count > 0:
                    print()
                    print("   Document list:")
                    for i, (doc_id, doc_info) in enumerate(list(doc_status_data.items())[:10], 1):
                        file_path = doc_info.get("file_path", "unknown")
                        file_name = Path(file_path).name if file_path != "unknown" else "unknown"
                        status = doc_info.get("status", "unknown")
                        chunks = doc_info.get("chunks_count", 0)
                        print(f"      {i}. {file_name}")
                        print(f"         ID: {doc_id[:16]}...")
                        print(f"         Status: {status}, Chunks: {chunks}")
                    
                    if doc_count > 10:
                        print(f"      ... and {doc_count - 10} more documents")
                else:
                    print("   ❌ Storage file exists but contains no documents!")
            else:
                print("   ⚠️  Document status file exists but has unexpected format")
        except Exception as e:
            print(f"   ❌ Error reading document status: {e}")
    else:
        print("   ❌ Document status file not found!")
    print()
    
    # 4. Check text chunks
    print("4. Text Chunks")
    print("-" * 80)
    text_chunks_file = storage_path / "kv_store_text_chunks.json"
    
    if text_chunks_file.exists():
        try:
            with open(text_chunks_file, 'r', encoding='utf-8') as f:
                chunks_data = json.load(f)
            
            if isinstance(chunks_data, dict):
                chunk_count = len(chunks_data)
                print(f"   ✅ Found {chunk_count:,} text chunks in storage")
                
                if chunk_count == 0:
                    print("   ❌ Storage file exists but contains no chunks!")
            else:
                print("   ⚠️  Text chunks file exists but has unexpected format")
        except Exception as e:
            print(f"   ❌ Error reading text chunks: {e}")
    else:
        print("   ❌ Text chunks file not found!")
    print()
    
    # 5. Check vector database
    print("5. Vector Database")
    print("-" * 80)
    vdb_chunks_file = storage_path / "vdb_chunks.json"
    
    if vdb_chunks_file.exists():
        try:
            with open(vdb_chunks_file, 'r', encoding='utf-8') as f:
                vdb_data = json.load(f)
            
            if isinstance(vdb_data, dict) and "data" in vdb_data:
                vector_count = len(vdb_data["data"])
                print(f"   ✅ Found {vector_count:,} vectors in database")
                
                if vector_count == 0:
                    print("   ❌ Vector database exists but contains no vectors!")
            else:
                print("   ⚠️  Vector database file exists but has unexpected format")
        except Exception as e:
            print(f"   ❌ Error reading vector database: {e}")
    else:
        print("   ❌ Vector database file not found!")
    print()
    
    # 6. Summary and recommendations
    print("=" * 80)
    print("Summary and Recommendations")
    print("=" * 80)
    print()
    
    has_issues = False
    
    if not os.path.exists(working_dir_abs):
        print("❌ CRITICAL: Storage directory does not exist!")
        print(f"   Fix: Create the directory or set RAG_WORKING_DIR correctly")
        print(f"   mkdir -p {working_dir_abs}")
        has_issues = True
    
    required_files = ["kv_store_doc_status.json", "kv_store_text_chunks.json", "vdb_chunks.json"]
    for req_file in required_files:
        if req_file in missing_files:
            print(f"❌ CRITICAL: Required file missing: {req_file}")
            has_issues = True
    
    if doc_status_file.exists():
        try:
            with open(doc_status_file, 'r') as f:
                doc_data = json.load(f)
                if isinstance(doc_data, dict) and len(doc_data) == 0:
                    print("❌ CRITICAL: Storage exists but contains no documents!")
                    print("   Fix: Upload and process documents first")
                    has_issues = True
        except:
            pass
    
    if not has_issues:
        print("✅ Storage appears to be properly configured and contains data.")
        print("   If queries still fail, check:")
        print("   1. Embedding model configuration matches the stored embeddings")
        print("   2. API keys and endpoints are correct")
        print("   3. Server logs for detailed error messages")
    else:
        print()
        print("Common fixes:")
        print("1. If storage directory is wrong:")
        print(f"   export RAG_WORKING_DIR=/path/to/rag_storage")
        print()
        print("2. If storage files are missing:")
        print("   - Copy storage files from development environment")
        print("   - Or re-process documents on the server")
        print()
        print("3. If storage is empty:")
        print("   - Upload and process documents using the /upload endpoint")
        print("   - Or run process_upload_folder.py on the server")
    
    # 7. Check embedding configuration
    print("6. Embedding Configuration")
    print("-" * 80)
    embedding_dim = os.getenv("EMBEDDING_DIM", "3072")
    embedding_model = os.getenv("EMBEDDING_MODEL", "text-embedding-3-large")
    embedding_binding = os.getenv("EMBEDDING_BINDING", "openai")
    
    print(f"   EMBEDDING_DIM: {embedding_dim}")
    print(f"   EMBEDDING_MODEL: {embedding_model}")
    print(f"   EMBEDDING_BINDING: {embedding_binding}")
    print()
    
    # Try to detect stored embedding dimension from vectors
    vdb_chunks_file = storage_path / "vdb_chunks.json"
    if vdb_chunks_file.exists():
        try:
            with open(vdb_chunks_file, 'r', encoding='utf-8') as f:
                vdb_data = json.load(f)
            
            stored_dim = None
            first_vector = None
            
            # Handle different vector database storage formats
            if isinstance(vdb_data, dict):
                if "data" in vdb_data:
                    data = vdb_data["data"]
                    # Format 1: data is a dict with vector IDs as keys
                    if isinstance(data, dict) and len(data) > 0:
                        first_vector_id = list(data.keys())[0]
                        first_vector = data[first_vector_id]
                    # Format 2: data is a list of vectors
                    elif isinstance(data, list) and len(data) > 0:
                        first_vector = data[0]
                        # If it's a list of dicts, get the vector from the dict
                        if isinstance(first_vector, dict):
                            # Try common keys for vector data
                            for key in ["vector", "embedding", "data", "values"]:
                                if key in first_vector and isinstance(first_vector[key], list):
                                    first_vector = first_vector[key]
                                    break
                # Format 3: vectors stored directly as dict values
                elif len(vdb_data) > 0:
                    first_vector_id = list(vdb_data.keys())[0]
                    first_vector = vdb_data[first_vector_id]
                    # If value is a dict, extract the vector
                    if isinstance(first_vector, dict):
                        for key in ["vector", "embedding", "data", "values"]:
                            if key in first_vector and isinstance(first_vector[key], list):
                                first_vector = first_vector[key]
                                break
            elif isinstance(vdb_data, list) and len(vdb_data) > 0:
                # Format 4: data is a list at root level
                first_vector = vdb_data[0]
                if isinstance(first_vector, dict):
                    for key in ["vector", "embedding", "data", "values"]:
                        if key in first_vector and isinstance(first_vector[key], list):
                            first_vector = first_vector[key]
                            break
            
            # Extract dimension from the vector
            if isinstance(first_vector, list) and len(first_vector) > 0:
                stored_dim = len(first_vector)
                print(f"   ✅ Detected stored embedding dimension: {stored_dim}")
                
                config_dim = int(embedding_dim)
                if stored_dim != config_dim:
                    print(f"   ❌ CRITICAL: Embedding dimension mismatch!")
                    print(f"      Stored vectors: {stored_dim} dimensions")
                    print(f"      Current config: {config_dim} dimensions")
                    print(f"      This will cause queries to fail!")
                    print()
                    print(f"   Fix: Set EMBEDDING_DIM={stored_dim} in your .env file")
                    has_issues = True
                else:
                    print(f"   ✅ Embedding dimension matches stored vectors")
            else:
                # Try alternative: check text chunks for embedding dimension
                print(f"   ⚠️  Could not extract vector dimension from vdb format")
                print(f"      Trying alternative method: checking text chunks...")
                
                try:
                    text_chunks_file = storage_path / "kv_store_text_chunks.json"
                    if text_chunks_file.exists():
                        with open(text_chunks_file, 'r', encoding='utf-8') as f:
                            chunks_data = json.load(f)
                        
                        if isinstance(chunks_data, dict) and len(chunks_data) > 0:
                            first_chunk_id = list(chunks_data.keys())[0]
                            first_chunk = chunks_data[first_chunk_id]
                            
                            # Check if chunk has embedding field
                            if isinstance(first_chunk, dict) and "embedding" in first_chunk:
                                chunk_embedding = first_chunk["embedding"]
                                if isinstance(chunk_embedding, list) and len(chunk_embedding) > 0:
                                    stored_dim = len(chunk_embedding)
                                    print(f"   ✅ Detected stored embedding dimension from chunks: {stored_dim}")
                                    
                                    config_dim = int(embedding_dim)
                                    if stored_dim != config_dim:
                                        print(f"   ❌ CRITICAL: Embedding dimension mismatch!")
                                        print(f"      Stored vectors: {stored_dim} dimensions")
                                        print(f"      Current config: {config_dim} dimensions")
                                        print(f"      This will cause queries to fail!")
                                        print()
                                        print(f"   Fix: Set EMBEDDING_DIM={stored_dim} in your .env file")
                                        has_issues = True
                                    else:
                                        print(f"   ✅ Embedding dimension matches stored vectors")
                except Exception as e2:
                    print(f"   ⚠️  Could not detect dimension from chunks either: {e2}")
                    print(f"      First vector type: {type(first_vector)}")
                    if isinstance(first_vector, dict):
                        print(f"      Available keys: {list(first_vector.keys())[:5]}")
        except Exception as e:
            print(f"   ⚠️  Could not detect stored embedding dimension: {e}")
            import traceback
            print(f"      Error details: {traceback.format_exc()}")
    print()
    
    print("=" * 80)
    if has_issues:
        print("❌ Issues found - see recommendations above")
    else:
        print("✅ Storage appears to be properly configured and contains data.")
        print("   If queries still fail, check:")
        print("   1. Embedding model configuration matches the stored embeddings")
        print("   2. API keys and endpoints are correct")
        print("   3. Server logs for detailed error messages")
    print()
    return not has_issues

if __name__ == "__main__":
    success = check_storage()
    sys.exit(0 if success else 1)

