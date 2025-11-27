# RAG-Anything Setup Instructions

Complete setup guide for RAG-Anything with Ollama and Neo4j.

## Prerequisites

- Python 3.10+
- Docker (for Neo4j)
- Ollama installed
- Homebrew (macOS) or appropriate package manager

## Step 1: Install Dependencies

```bash
# Install RAG-Anything with all optional dependencies
pip install 'raganything[all]'

# Install additional required packages
pip install requests neo4j
```

## Step 2: Install LibreOffice (for Office document support)

**macOS:**
```bash
brew install --cask libreoffice
```

**Ubuntu/Debian:**
```bash
sudo apt-get install libreoffice
```

**Windows:**
Download from [LibreOffice official website](https://www.libreoffice.org/download/download/)

## Step 3: Setup Ollama

### Install Ollama (if not already installed)

**macOS:**
```bash
brew install ollama
```

**Linux/Windows:**
Download from [Ollama website](https://ollama.ai)

### Start Ollama Service

```bash
# Start Ollama (runs in background)
ollama serve
```

### Pull Required Models

```bash
# Pull LLM model
ollama pull llama3.1:8b

# Pull embedding model
ollama pull nomic-embed-text
```

### Verify Ollama is Running

```bash
# Check if Ollama is accessible
curl http://localhost:11434/api/tags

# List installed models
ollama list
```

## Step 4: Setup Neo4j

### Start Neo4j with Docker

```bash
# Start Neo4j container
docker run -d \
  --name neo4j-rag \
  -p 7474:7474 \
  -p 7687:7687 \
  -e NEO4J_AUTH=neo4j/ragpassword123 \
  neo4j:latest
```

### Verify Neo4j is Running

```bash
# Check container status
docker ps | grep neo4j

# Access Neo4j Browser (optional)
# Open http://localhost:7474 in your browser
# Login with: neo4j / ragpassword123
```

## Step 5: Configure Environment

The `.env` file has been created with the following configuration:

- **LLM**: Ollama with `llama3.1:8b`
- **Embeddings**: Ollama with `nomic-embed-text`
- **Graph Storage**: Neo4j at `neo4j://localhost:7687`
- **Multimodal Processing**: Enabled for images, tables, and equations

### Key Configuration Values

```env
LLM_BINDING=ollama
LLM_MODEL=llama3.1:8b
LLM_BINDING_HOST=http://localhost:11434
LLM_BINDING_API_KEY=ollama

EMBEDDING_BINDING=ollama
EMBEDDING_MODEL=nomic-embed-text
EMBEDDING_DIM=768
EMBEDDING_BINDING_HOST=http://localhost:11434

LIGHTRAG_GRAPH_STORAGE=Neo4JStorage
NEO4J_URI=neo4j://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD='ragpassword123'
```

## Step 6: Verify Setup

Run the demo script to verify everything is working:

```bash
# Run all demos
python examples/setup_demo.py

# Run specific demos
python examples/setup_demo.py --demo single
python examples/setup_demo.py --demo parallel
python examples/setup_demo.py --demo multimodal

# Process a document
python examples/setup_demo.py --file path/to/document.pdf
```

## Step 7: Usage Examples

### Basic Usage

```python
import asyncio
from raganything import RAGAnything, RAGAnythingConfig
from lightrag.llm.openai import openai_complete_if_cache, openai_embed
from lightrag.utils import EmbeddingFunc

async def main():
    # Setup functions (see setup_demo.py for full example)
    rag = RAGAnything(...)
    
    # Process a document
    await rag.process_document_complete("document.pdf", "./output")
    
    # Single query
    result = await rag.aquery("What is this document about?", mode="hybrid")
    
    # Parallel multi-query
    queries = ["What are the main topics?", "What are the conclusions?"]
    results = await rag.parallel_query(queries, mode="hybrid", merge_strategy="union")
    
    print(results["merged_result"])

asyncio.run(main())
```

## Troubleshooting

### Ollama Connection Issues

- Ensure Ollama is running: `ollama serve`
- Check if port 11434 is accessible: `curl http://localhost:11434/api/tags`
- Verify models are installed: `ollama list`

### Neo4j Connection Issues

- Check Docker is running: `docker ps`
- Verify Neo4j container: `docker ps | grep neo4j`
- Check Neo4j logs: `docker logs neo4j-rag`
- Test connection: `python -c "from neo4j import GraphDatabase; driver = GraphDatabase.driver('neo4j://localhost:7687', auth=('neo4j', 'ragpassword123')); driver.verify_connectivity(); print('OK')"`

### Document Processing Issues

- Verify LibreOffice is installed: `which soffice` or `soffice --version`
- Check file permissions
- Ensure file format is supported (PDF, DOCX, images, etc.)

### Import Errors

- Ensure all dependencies are installed: `pip install 'raganything[all]'`
- Check Python version: `python --version` (should be 3.10+)

## Next Steps

1. Process your documents: Use `rag.process_document_complete()` to add documents to the knowledge base
2. Query your data: Use `rag.aquery()` for single queries or `rag.parallel_query()` for multiple queries
3. Explore multimodal queries: Use `rag.aquery_with_multimodal()` for queries with images, tables, or equations

## Additional Resources

- [RAG-Anything Documentation](https://github.com/HKUDS/RAG-Anything)
- [Ollama Documentation](https://ollama.ai/docs)
- [Neo4j Documentation](https://neo4j.com/docs/)

