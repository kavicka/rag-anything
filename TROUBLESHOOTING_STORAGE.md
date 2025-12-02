# Troubleshooting: "No relevant content found in the documents"

## Problem

When deploying on a server, queries instantly return:
```
No relevant content found in the documents.
```

## Root Causes

This issue typically occurs when:

1. **Embedding dimension mismatch** - The embedding model/dimension used for queries doesn't match the one used to create stored vectors (MOST COMMON)
2. **Storage path is incorrect** - The `RAG_WORKING_DIR` environment variable points to a different location than where the storage files actually are
3. **Storage files not copied** - The storage files from your development environment weren't copied to the server
4. **Storage is empty** - The storage directory exists but contains no documents
5. **Relative path issues** - Using relative paths (like `./rag_storage`) which resolve differently on the server

## Diagnosis

### Method 1: Use the Diagnostic Script

Run the diagnostic script on your server:

```bash
python check_storage.py
```

This will check:
- Storage directory path and existence
- All storage files and their sizes
- Document count
- Chunk count
- Vector database status
- **Embedding dimension** - Detects stored vector dimensions and compares with configuration

### Method 2: Use the API Endpoint

Call the storage status endpoint:

```bash
curl -H "Authorization: Bearer YOUR_TOKEN" http://your-server:8000/storage/status
```

This returns detailed information about:
- Working directory (absolute path)
- Document count
- Chunk count
- Storage file status
- Whether storage has content

### Method 3: Check Server Logs

Look for these log messages during server startup:

```
Initializing RAG database from: /path/to/rag_storage
Found X documents in storage
```

If you see:
```
WARNING: Storage initialized but no documents found
```

This indicates the storage path is wrong or the storage is empty.

## Solutions

### Solution 1: Fix Embedding Dimension Mismatch (MOST COMMON)

**If storage has content but queries return "No relevant content found", this is almost always the cause.**

1. **Check what dimension your stored vectors use:**
   ```bash
   python check_storage.py
   ```
   
   Look for the "Detected stored embedding dimension" line.

2. **Check your current configuration:**
   ```bash
   echo $EMBEDDING_DIM
   echo $EMBEDDING_MODEL
   ```

3. **Set the correct dimension in your `.env` file:**
   ```
   EMBEDDING_DIM=1024  # or whatever dimension your stored vectors use
   EMBEDDING_MODEL=bge-m3:latest  # or whatever model was used to create vectors
   ```

4. **Restart the server**

**Example:**
- If documents were processed with `bge-m3:latest` (1024 dimensions)
- But queries use `text-embedding-3-large` (3072 dimensions)
- This mismatch causes similarity search to fail completely

**Important:** The embedding model AND dimension must match what was used to create the stored vectors.

### Solution 2: Fix Storage Path

1. **Find where your storage files actually are on the server:**
   ```bash
   find / -name "kv_store_doc_status.json" 2>/dev/null
   ```

2. **Set the correct path in your environment:**
   ```bash
   export RAG_WORKING_DIR=/absolute/path/to/rag_storage
   ```

   Or in your `.env` file:
   ```
   RAG_WORKING_DIR=/absolute/path/to/rag_storage
   ```

3. **Restart the server**

### Solution 3: Copy Storage Files to Server

If your storage files are on your development machine:

1. **On your development machine, identify the storage directory:**
   ```bash
   echo $RAG_WORKING_DIR  # or check .env file
   ```

2. **Copy the entire storage directory to the server:**
   ```bash
   scp -r ./rag_storage user@server:/path/to/deployment/rag_storage
   ```

3. **Set RAG_WORKING_DIR on the server:**
   ```bash
   export RAG_WORKING_DIR=/path/to/deployment/rag_storage
   ```

4. **Restart the server**

### Solution 4: Re-process Documents on Server

If storage is empty or corrupted:

1. **Upload documents using the API:**
   ```bash
   curl -X POST \
     -H "Authorization: Bearer YOUR_TOKEN" \
     -F "file=@document.pdf" \
     http://your-server:8000/upload
   ```

2. **Wait for processing to complete** (check status endpoint)

3. **Verify documents are processed:**
   ```bash
   curl -H "Authorization: Bearer YOUR_TOKEN" \
     http://your-server:8000/documents
   ```

### Solution 5: Use Absolute Paths

Always use absolute paths in production:

**In `.env` file:**
```
RAG_WORKING_DIR=/var/www/rag-anything/rag_storage
```

**Not:**
```
RAG_WORKING_DIR=./rag_storage  # ❌ Relative paths can cause issues
```

## Verification

After fixing the issue, verify:

1. **Check storage status:**
   ```bash
   python check_storage.py
   ```

2. **Check health endpoint:**
   ```bash
   curl http://your-server:8000/health
   ```
   
   Should return:
   ```json
   {
     "status": "healthy",
     "system_ready": true,
     "message": "RAG system is ready (X documents)"
   }
   ```

3. **Try a query:**
   ```bash
   curl -X POST \
     -H "Authorization: Bearer YOUR_TOKEN" \
     -H "Content-Type: application/json" \
     -d '{"content": "test query"}' \
     http://your-server:8000/chats/{chat_id}/messages
   ```

## Prevention

To avoid this issue in the future:

1. **Always use absolute paths in production**
2. **Document where storage files should be located**
3. **Include storage directory in deployment scripts**
4. **Set RAG_WORKING_DIR in your deployment configuration**
5. **Verify storage after deployment**

## Additional Debugging

If the issue persists:

1. **Check embedding dimension mismatch** - This is the #1 cause when storage has content
   - Run `python check_storage.py` and look for dimension mismatch warnings
   - Check server logs for "CRITICAL: Embedding dimension mismatch" errors
   
2. **Check server logs** for detailed error messages:
   ```bash
   # If using systemd
   journalctl -u your-service -f
   
   # If using PM2
   pm2 logs your-app
   ```
   
3. **Verify embedding API is working:**
   - Test embedding API endpoint directly
   - Check if embedding requests are succeeding
   - Look for embedding errors in logs
   
4. **Verify file permissions** - the server process needs read/write access
5. **Check disk space** - ensure there's enough space for storage
6. **Check API keys** - ensure LLM and embedding API keys are correct
7. **Test query embedding generation:**
   - Check if query embeddings are being generated successfully
   - Verify embedding dimension matches stored vectors

## Related Files

- `api_server.py` - Server initialization and storage path handling
- `check_storage.py` - Diagnostic script
- `raganything/query.py` - Query execution and error handling
- `.env` - Environment configuration

