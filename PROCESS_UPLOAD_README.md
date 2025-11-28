# Process Upload Folder

This script processes all files in the `docs/upload` folder and inserts them into the RAG-Anything knowledge base.

## Quick Start

1. **Set up environment variables** (create a `.env` file or export them):

```bash
# Required: API Key for LLM
LLM_BINDING_API_KEY=your_api_key_here
# or
OPENAI_API_KEY=your_api_key_here

# Optional: Base URL (if using a different API endpoint)
LLM_BINDING_HOST=https://api.openai.com/v1
# or
OPENAI_BASE_URL=https://api.openai.com/v1

# Optional: Configuration
WORKING_DIR=./rag_storage          # Database storage location
OUTPUT_DIR=./output                # Parsed document output location
PARSER=mineru                      # Parser: mineru or docling
PARSE_METHOD=auto                  # Parse method: auto, ocr, or txt
MAX_CONCURRENT_FILES=1             # Number of files to process concurrently
```

2. **Place your files** in the `docs/upload/` folder

3. **Run the script**:

```bash
python process_upload_folder.py
```

Or make it executable and run directly:

```bash
chmod +x process_upload_folder.py
./process_upload_folder.py
```

## What It Does

1. **Scans** the `docs/upload` folder for supported files
2. **Parses** each file using MinerU or Docling parser
3. **Extracts** text, images, tables, and equations
4. **Processes** multimodal content (images, tables, equations)
5. **Inserts** everything into the RAG-Anything knowledge base
6. **Tests** the database with a sample query

## Supported File Types

- **PDFs**: `.pdf`
- **Office Documents**: `.doc`, `.docx`, `.ppt`, `.pptx`, `.xls`, `.xlsx` (requires LibreOffice)
- **Images**: `.jpg`, `.jpeg`, `.png`, `.bmp`, `.tiff`, `.gif`, `.webp`
- **Text Files**: `.txt`, `.md`

## Output

- **Database**: Stored in `./rag_storage` (or `WORKING_DIR`)
- **Parsed Documents**: Stored in `./output` (or `OUTPUT_DIR`)
- **Logs**: Written to `process_upload.log` in the current directory

## Example Usage

```bash
# Set your API key
export LLM_BINDING_API_KEY="sk-..."

# Process all files in docs/upload
python process_upload_folder.py
```

## Troubleshooting

1. **API Key Error**: Make sure you've set `LLM_BINDING_API_KEY` or `OPENAI_API_KEY`
2. **Parser Not Found**: Install MinerU: `pip install mineru` or use Docling
3. **Office Documents**: Install LibreOffice for Office document support
4. **Memory Issues**: Reduce `MAX_CONCURRENT_FILES` if processing large files

## Next Steps

After processing, you can query the database using the RAG-Anything query methods:

```python
from raganything import RAGAnything

# Load existing RAG instance
rag = RAGAnything(...)

# Query the database
result = await rag.aquery("Your question here", mode="hybrid")
```

