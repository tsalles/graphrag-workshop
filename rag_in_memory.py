import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import faiss
import numpy as np
from openai import AzureOpenAI

try:
    from azure.identity import DefaultAzureCredential, get_bearer_token_provider
except ImportError:
    DefaultAzureCredential = None
    get_bearer_token_provider = None


DEFAULT_ENDPOINT = "https://aifoundrylearningenv.openai.azure.com"
DEFAULT_API_VERSION = "2024-12-01-preview"
DEFAULT_CHAT_DEPLOYMENT = "gpt-4o"
DEFAULT_EMBEDDINGS_DEPLOYMENT = "text-embedding-3-large"


@dataclass
class Chunk:
    source: str
    index: int
    text: str
    embedding: List[float]


def build_azure_client(endpoint: str, api_version: str) -> AzureOpenAI:
    api_key = os.environ.get("AZURE_OPENAI_API_KEY")

    if api_key:
        return AzureOpenAI(azure_endpoint=endpoint, api_version=api_version, api_key=api_key)

    if DefaultAzureCredential is None or get_bearer_token_provider is None:
        raise ValueError("Install azure-identity or set AZURE_OPENAI_API_KEY")

    token_provider = get_bearer_token_provider(
        DefaultAzureCredential(), "https://cognitiveservices.azure.com/.default"
    )
    return AzureOpenAI(
        azure_endpoint=endpoint, api_version=api_version, azure_ad_token_provider=token_provider
    )


def read_input_texts(input_dir: Path) -> List[Tuple[str, str]]:
    texts: List[Tuple[str, str]] = []
    for path in sorted(input_dir.glob("**/*")):
        if path.is_file() and path.suffix.lower() in {".txt", ".md"}:
            content = path.read_text(encoding="utf-8", errors="ignore")
            if content.strip():
                texts.append((path.name, content))
    return texts


def chunk_text(text: str, chunk_size: int, overlap: int) -> List[str]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    chunks: List[str] = []
    start = 0
    step = chunk_size - overlap
    while start < len(text):
        end = min(len(text), start + chunk_size)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start += step
    return chunks


def embed_chunks(
    client: AzureOpenAI,
    embedding_deployment: str,
    items: List[Tuple[str, str]],
    chunk_size: int,
    overlap: int,
) -> List[Chunk]:
    chunks: List[Chunk] = []
    for source, text in items:
        parts = chunk_text(text, chunk_size=chunk_size, overlap=overlap)
        for idx, part in enumerate(parts):
            response = client.embeddings.create(model=embedding_deployment, input=part)
            vector = response.data[0].embedding
            chunks.append(Chunk(source=source, index=idx, text=part, embedding=vector))
    return chunks


def build_faiss_index(chunks: List[Chunk]) -> faiss.IndexFlatIP:
    vectors = np.array([chunk.embedding for chunk in chunks], dtype="float32")
    faiss.normalize_L2(vectors)
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)  # type: ignore[arg-type]
    return index


def save_chunks(chunks: List[Chunk], path: Path) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for chunk in chunks:
            handle.write(
                json.dumps(
                    {"source": chunk.source, "index": chunk.index, "text": chunk.text},
                    ensure_ascii=True,
                )
                + "\n"
            )


def load_chunks(path: Path) -> List[Chunk]:
    chunks: List[Chunk] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            data = json.loads(line)
            chunks.append(
                Chunk(
                    source=data["source"],
                    index=int(data["index"]),
                    text=data["text"],
                    embedding=[],
                )
            )
    return chunks


def search_top_k(
    index: faiss.IndexFlatIP,
    chunks: List[Chunk],
    query_vector: List[float],
    k: int,
) -> List[Tuple[Chunk, float]]:
    query = np.array([query_vector], dtype="float32")
    faiss.normalize_L2(query)
    scores, indices = index.search(query, k)  # type: ignore[call-arg]
    results: List[Tuple[Chunk, float]] = []
    for idx, score in zip(indices[0], scores[0]):
        if idx == -1:
            continue
        results.append((chunks[int(idx)], float(score)))
    return results


def answer_question(
    client: AzureOpenAI,
    chat_deployment: str,
    question: str,
    context_chunks: List[Chunk],
) -> str:
    context = "\n\n".join(
        f"Source: {c.source} | Chunk: {c.index}\n{c.text}" for c in context_chunks
    )
    system_prompt = (
        "You are a helpful assistant. Answer the question using only the provided context. "
        "If the context does not contain the answer, say you do not know."
    )
    user_prompt = f"Context:\n{context}\n\nQuestion: {question}"

    response = client.chat.completions.create(
        model=chat_deployment,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.2,
    )
    content = response.choices[0].message.content
    return content.strip() if content else ""


def add_common_azure_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--endpoint", default=None, help="Azure OpenAI endpoint")
    parser.add_argument("--api-version", default=None, help="Azure OpenAI API version")
    parser.add_argument(
        "--chat-deployment",
        default=None,
        help="Chat deployment name (e.g., gpt-4o)",
    )
    parser.add_argument(
        "--embeddings-deployment",
        default=None,
        help="Embeddings deployment name (e.g., text-embedding-3-large)",
    )


def resolve_azure_settings(args: argparse.Namespace) -> Tuple[str, str, str, str]:
    endpoint = args.endpoint or os.environ.get("AZURE_OPENAI_ENDPOINT") or DEFAULT_ENDPOINT
    api_version = (
        args.api_version or os.environ.get("AZURE_OPENAI_API_VERSION") or DEFAULT_API_VERSION
    )
    embedding_deployment = (
        args.embeddings_deployment
        or os.environ.get("AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT")
        or DEFAULT_EMBEDDINGS_DEPLOYMENT
    )
    chat_deployment = (
        args.chat_deployment
        or os.environ.get("AZURE_OPENAI_CHAT_DEPLOYMENT")
        or DEFAULT_CHAT_DEPLOYMENT
    )
    return endpoint, api_version, embedding_deployment, chat_deployment


def index_documents(args: argparse.Namespace) -> None:
    endpoint, api_version, embedding_deployment, _ = resolve_azure_settings(args)
    client = build_azure_client(endpoint=endpoint, api_version=api_version)

    base_dir = Path(__file__).resolve().parent
    input_dir = (base_dir / args.input_dir).resolve()
    if not input_dir.exists():
        raise FileNotFoundError(f"Input folder not found: {input_dir}")

    items = read_input_texts(input_dir)
    if not items:
        raise ValueError("No .txt or .md files found in input folder")

    print(f"Indexing {len(items)} file(s) from {input_dir}...")
    chunks = embed_chunks(
        client,
        embedding_deployment=embedding_deployment,
        items=items,
        chunk_size=args.chunk_size,
        overlap=args.overlap,
    )
    print(f"Created {len(chunks)} chunk(s).")

    index = build_faiss_index(chunks)

    store_dir = (base_dir / args.store_dir).resolve()
    store_dir.mkdir(parents=True, exist_ok=True)
    index_path = store_dir / "faiss.index"
    chunks_path = store_dir / "chunks.jsonl"
    info_path = store_dir / "store_info.json"

    faiss.write_index(index, str(index_path))
    save_chunks(chunks, chunks_path)
    info_path.write_text(
        json.dumps(
            {
                "endpoint": endpoint,
                "api_version": api_version,
                "embedding_deployment": embedding_deployment,
                "chunk_size": args.chunk_size,
                "overlap": args.overlap,
                "input_dir": str(input_dir),
            },
            ensure_ascii=True,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"Saved FAISS index to {index_path}")
    print(f"Saved chunk metadata to {chunks_path}")


def query_documents(args: argparse.Namespace) -> None:
    question = args.question or input("Question: ").strip()
    if not question:
        raise ValueError("Question is required")

    endpoint, api_version, embedding_deployment, chat_deployment = resolve_azure_settings(args)
    client = build_azure_client(endpoint=endpoint, api_version=api_version)

    base_dir = Path(__file__).resolve().parent
    store_dir = (base_dir / args.store_dir).resolve()
    index_path = store_dir / "faiss.index"
    chunks_path = store_dir / "chunks.jsonl"

    if not index_path.exists() or not chunks_path.exists():
        raise FileNotFoundError("Run 'rag_in_memory.py index' before querying")

    index = faiss.read_index(str(index_path))
    chunks = load_chunks(chunks_path)
    if not chunks:
        raise ValueError("No chunks found in vector store")

    query_response = client.embeddings.create(model=embedding_deployment, input=question)
    query_vector = query_response.data[0].embedding
    top = search_top_k(index, chunks, query_vector, k=args.top_k)
    top_chunks = [c for c, _ in top]

    answer = answer_question(client, chat_deployment, question, top_chunks)
    print("\nAnswer:\n")
    print(answer)

    print("\nSources:")
    for chunk, score in top:
        print(f"- {chunk.source} | chunk {chunk.index} | score {score:.4f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="In-memory RAG with Azure OpenAI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    index_parser = subparsers.add_parser("index", help="Index documents to disk")
    index_parser.add_argument("--input-dir", default="input", help="Folder with source files")
    index_parser.add_argument("--store-dir", default=".rag_store", help="Store directory")
    index_parser.add_argument("--chunk-size", type=int, default=1200, help="Chunk size in characters")
    index_parser.add_argument("--overlap", type=int, default=200, help="Chunk overlap in characters")
    add_common_azure_args(index_parser)

    query_parser = subparsers.add_parser("query", help="Query the stored index")
    query_parser.add_argument("question", nargs="?", help="Question to ask")
    query_parser.add_argument("--store-dir", default=".rag_store", help="Store directory")
    query_parser.add_argument("--top-k", type=int, default=4, help="Number of chunks to retrieve")
    add_common_azure_args(query_parser)

    args = parser.parse_args()
    if args.command == "index":
        index_documents(args)
    elif args.command == "query":
        query_documents(args)


if __name__ == "__main__":
    main()
