from __future__ import annotations

import os
from pathlib import Path
from typing import Protocol

from app.rag import Chunk, LocalVectorStore, MemoryVectorStore


class VectorStore(Protocol):
    chunks: list[Chunk]

    def add_texts(self, source: str, texts: list[str], embed_fn): ...

    def similarity_search(self, query: str, embed_fn, k: int = 4) -> list[Chunk]: ...
    def similarity_search_with_scores(self, query: str, embed_fn, k: int = 4) -> list[tuple[float, Chunk]]: ...

    def clear(self) -> None: ...


def get_vector_backend() -> str:
    return os.getenv("VECTOR_BACKEND", "local_json").strip().lower()


def build_internal_store(db_path: Path) -> VectorStore:
    backend = get_vector_backend()
    if backend == "local_json":
        return LocalVectorStore(db_path)
    raise RuntimeError(
        f"不支持的 VECTOR_BACKEND={backend}。当前仅支持 local_json。"
    )


def build_session_store() -> VectorStore:
    # Session store is intentionally in-memory to avoid persisting user uploads.
    return MemoryVectorStore()
