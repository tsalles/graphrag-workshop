# GraphRAG

A toolkit for building and querying knowledge graphs from document collections using [Microsoft GraphRAG](https://microsoft.github.io/graphrag/) and Azure OpenAI. Includes an MCP server for integration with AI assistants, an in-memory vector RAG pipeline for comparison, and a PDF-to-text converter for document ingestion.

## Components

| File | Description |
|---|---|
| `graphrag_mcp.py` | [FastMCP](https://github.com/jlowin/fastmcp) server exposing GraphRAG queries as tools (local, global, drift, basic) |
| `rag_in_memory.py` | Standalone vector RAG pipeline using FAISS and Azure OpenAI for indexing and Q&A |
| `pdf2txt.py` | Batch PDF-to-text converter for preparing input documents |
| `settings.yaml` | GraphRAG configuration (models, chunking, storage) |

## Prerequisites

- Python 3.10+
- An Azure OpenAI resource with `gpt-4o` and `text-embedding-3-large` deployments
- Authentication via **Azure Managed Identity** (default) or an API key

## Setup

```bash
# Create and activate a virtual environment
python -m venv .venv
.venv/Scripts/Activate.ps1   # Windows PowerShell
# source .venv/bin/activate  # Linux / macOS

# Install dependencies
pip install -r requirements.txt
```

### Environment variables

Create a `.env` file (already git-ignored) for any secrets:

```dotenv
# Only needed if not using Azure Managed Identity
GRAPHRAG_API_KEY=<your-key>
AZURE_OPENAI_API_KEY=<your-key>
```

Other optional variables for `rag_in_memory.py`:

| Variable | Default |
|---|---|
| `AZURE_OPENAI_ENDPOINT` | `https://aifoundrylearningenv.openai.azure.com` |
| `AZURE_OPENAI_API_VERSION` | `2024-12-01-preview` |
| `AZURE_OPENAI_CHAT_DEPLOYMENT` | `gpt-4o` |
| `AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT` | `text-embedding-3-large` |
| `GRAPHRAG_QUERY_TIMEOUT_SECONDS` | `600` |

## Usage

### 1. Prepare input documents

Place `.txt` or `.md` files in the `input/` directory. To convert PDFs:

```bash
python pdf2txt.py pdfs/ input/
# Use -r for recursive scanning
python pdf2txt.py -r pdfs/ input/
```

### 2. Build the GraphRAG knowledge graph

```bash
python -m graphrag index
```

This reads from `input/`, builds the graph, and writes artifacts to `output/`, `cache/`, and `logs/`.

### 3. Query with GraphRAG (CLI)

```bash
python -m graphrag query --method local "What are the main topics?"
python -m graphrag query --method global "Summarize the dataset"
python -m graphrag query --method drift "Explain concept X in detail"
python -m graphrag query --method basic "Simple vector search question"
```

### 4. Query with the MCP server

Start the FastMCP server to expose GraphRAG as tools for AI assistants:

```bash
python graphrag_mcp.py
```

The server provides:
- **Tool** — `graphrag_query(question, mode)` — runs a query in the selected mode
- **Prompt** — `graphrag_query_prompt` — helps the caller choose the right query mode

### 5. In-memory vector RAG (FAISS)

An alternative RAG pipeline for comparison with GraphRAG:

```bash
# Index documents
python rag_in_memory.py index --input-dir input/

# Query
python rag_in_memory.py query "What are the key findings?"
python rag_in_memory.py query --top-k 8 "Explain the methodology"
```

## Project structure

```
├── graphrag_mcp.py          # MCP server
├── rag_in_memory.py         # FAISS vector RAG
├── pdf2txt.py               # PDF converter
├── settings.yaml            # GraphRAG config
├── requirements.txt         # Python dependencies
├── input/                   # Source documents (text)
├── pdfs/                    # Source PDFs (git-ignored)
├── output/                  # GraphRAG output (git-ignored)
├── cache/                   # GraphRAG cache (git-ignored)
├── logs/                    # GraphRAG logs (git-ignored)
├── prompts/                 # GraphRAG prompt templates
└── tuned_prompts/           # Custom-tuned prompts
```

