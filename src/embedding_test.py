"""Embedding test tool for Google Gemini (Python).

Usage examples:
  python -m src.embedding_test --text "Xin chao Gemini"
  python -m src.embedding_test --text "hello" "xin chao"
  python -m src.embedding_test --file ./samples.txt --task retrieval_document
  python -m src.embedding_test --text "what is RAG?" --model text-embedding-004 --task retrieval_query

Notes:
- API key resolution order: --api-key > .env (GOOGLE_API_KEY) > environment variable.
- Model name is normalized: if it does not start with "models/", the prefix will be added automatically.
- Default model: models/text-embedding-004
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import List, Optional, Sequence, Tuple, TypedDict

from dotenv import load_dotenv
import google.generativeai as genai

DEFAULT_MODEL: str = "models/text-embedding-004"
PREVIEW_VALUES: int = 8
SUPPORTED_TASKS: Tuple[str, ...] = (
    "retrieval_document",
    "retrieval_query",
    "semantic_similarity",
    "classification",
    "clustering",
)


class EmbedRequest(TypedDict, total=False):
    """Request shape for batch embedding calls."""
    content: str
    task_type: str
    title: str


def normalize_model_name(model: str) -> str:
    """Ensure the model name has the required "models/" prefix.

    Args:
        model: A model name like "text-embedding-004" or "models/text-embedding-004".
    Returns:
        Normalized model name with the "models/" prefix.
    """
    if model.startswith("models/"):
        return model
    return f"models/{model}"


def resolve_api_key(cli_api_key: Optional[str]) -> str:
    """Resolve API key from CLI flag, .env, or OS environment.

    Args:
        cli_api_key: API key passed via command line.
    Returns:
        Resolved API key string.
    Raises:
        RuntimeError: If no API key is found.
    """
    if cli_api_key:
        return cli_api_key
    load_dotenv()
    env_key: Optional[str] = os.getenv("GOOGLE_API_KEY")
    if env_key:
        return env_key
    raise RuntimeError("GOOGLE_API_KEY is not set. Provide via --api-key or .env or environment variable.")


def configure_client(api_key: str) -> None:
    """Configure the google-generativeai client."""
    genai.configure(api_key=api_key)


def embed_single(text: str, model: str, task_type: str, title: Optional[str]) -> List[float]:
    """Embed a single text using Gemini embeddings.

    Args:
        text: The input text to embed.
        model: The embedding model name (normalized).
        task_type: Embedding task type, e.g., "retrieval_document" or "retrieval_query".
        title: Optional document title for context.
    Returns:
        The embedding vector as a list of floats.
    """
    data: dict = genai.embed_content(model=model, content=text, task_type=task_type, title=title)
    if not isinstance(data, dict):
        raise RuntimeError("Unexpected response type from embed_content().")
    vector: Optional[List[float]] = data.get("embedding")  # type: ignore[assignment]
    if not isinstance(vector, list):
        raise RuntimeError(f"Missing 'embedding' in response keys: {list(data.keys())}")
    return vector  # type: ignore[return-value]


def build_requests(texts: Sequence[str], task_type: str, title: Optional[str]) -> List[EmbedRequest]:
    """Build batch requests for multiple texts.

    Args:
        texts: Collection of texts.
        task_type: Embedding task type.
        title: Optional document title applied to all.
    Returns:
        A list of EmbedRequest dictionaries.
    """
    reqs: List[EmbedRequest] = []
    for t in texts:
        item: EmbedRequest = {"content": t, "task_type": task_type}
        if title:
            item["title"] = title
        reqs.append(item)
    return reqs


def embed_batch(texts: Sequence[str], model: str, task_type: str, title: Optional[str]) -> List[List[float]]:
    """Embed multiple texts in one request.

    Args:
        texts: The texts to embed.
        model: The embedding model name (normalized).
        task_type: Embedding task type.
        title: Optional title applied to each request.
    Returns:
        A list of embedding vectors corresponding to the inputs.
    """
    # Pass list of texts directly to embed_content for batch processing
    data: dict = genai.embed_content(model=model, content=list(texts), task_type=task_type, title=title)
    if not isinstance(data, dict):
        raise RuntimeError("Unexpected response type from embed_content().")
    vectors_data: Optional[List[List[float]]] = data.get("embedding")  # type: ignore[assignment]
    if not isinstance(vectors_data, list):
        raise RuntimeError(f"Missing 'embedding' in batch response keys: {list(data.keys())}")
    return vectors_data


def read_texts_from_file(file_path: str) -> List[str]:
    """Read newline-separated texts from a UTF-8 file."""
    path: Path = Path(file_path)
    lines: List[str] = path.read_text(encoding="utf-8").splitlines()
    return [ln.strip() for ln in lines if ln.strip()]


def preview_vector(vector: Sequence[float]) -> str:
    """Format a short preview of the embedding vector."""
    head: Sequence[float] = vector[:PREVIEW_VALUES]
    head_str: str = ", ".join(f"{v:.6f}" for v in head)
    return f"dim={len(vector)} head=[{head_str}]"


def save_embeddings_json(
    out_path: str,
    model: str,
    task_type: str,
    texts: Sequence[str],
    vectors: Sequence[Sequence[float]],
) -> None:
    """Save embeddings to a JSON file."""
    payload: dict = {
        "model": model,
        "task_type": task_type,
        "dimension": len(vectors[0]) if vectors else 0,
        "items": [
            {"text": t, "embedding": list(v)} for t, v in zip(texts, vectors)
        ],
    }
    Path(out_path).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Parse command line arguments."""
    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        description="Quick test for Gemini embeddings (text-embedding-004)",
    )
    parser.add_argument("--text", nargs="+", help="One or more texts to embed", default=None)
    parser.add_argument("--file", type=str, help="Path to a file with one text per line", default=None)
    parser.add_argument("--model", type=str, help="Embedding model name", default=DEFAULT_MODEL)
    parser.add_argument("--task", type=str, choices=SUPPORTED_TASKS, default="retrieval_document", help="Embedding task type")
    parser.add_argument("--title", type=str, help="Optional document title", default=None)
    parser.add_argument("--api-key", type=str, help="Override GOOGLE_API_KEY", default=None)
    parser.add_argument("--save", type=str, help="Path to save embeddings JSON", default=None)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Entrypoint for CLI usage."""
    try:
        args: argparse.Namespace = parse_args(argv)
        api_key: str = resolve_api_key(args.api_key)
        configure_client(api_key)
        model: str = normalize_model_name(args.model)
        texts: List[str] = []
        if args.text:
            texts.extend([str(t) for t in args.text])
        if args.file:
            texts.extend(read_texts_from_file(args.file))
        if not texts:
            raise RuntimeError("No input texts. Provide --text or --file.")
        if len(texts) == 1:
            vec: List[float] = embed_single(text=texts[0], model=model, task_type=args.task, title=args.title)
            print(f"[1/1] {preview_vector(vec)}")
            if args.save:
                save_embeddings_json(out_path=args.save, model=model, task_type=args.task, texts=texts, vectors=[vec])
        else:
            vecs: List[List[float]] = embed_batch(texts=texts, model=model, task_type=args.task, title=args.title)
            for idx, vec in enumerate(vecs, start=1):
                print(f"[{idx}/{len(vecs)}] {preview_vector(vec)}")
            if args.save:
                save_embeddings_json(out_path=args.save, model=model, task_type=args.task, texts=texts, vectors=vecs)
        return 0
    except Exception as exc:  # pylint: disable=broad-except
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
