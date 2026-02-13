from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import numpy as np


@dataclass
class Chunk:
    id: str
    source: str
    text: str
    embedding: list[float]


class LocalVectorStore:
    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        self.chunks: list[Chunk] = []
        self._load()

    def _load(self) -> None:
        if not self.db_path.exists():
            return
        raw = json.loads(self.db_path.read_text(encoding="utf-8"))
        self.chunks = [Chunk(**item) for item in raw]

    def _persist(self) -> None:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.db_path.write_text(
            json.dumps([asdict(c) for c in self.chunks], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def clear(self) -> None:
        self.chunks = []
        self._persist()

    def add_texts(
        self,
        source: str,
        texts: list[str],
        embed_fn: Callable[[str], list[float]],
    ) -> int:
        count = 0
        for idx, text in enumerate(texts):
            payload = text.strip()
            if not payload:
                continue
            chunk = Chunk(
                id=f"{source}-{len(self.chunks)+idx}",
                source=source,
                text=payload,
                embedding=embed_fn(payload),
            )
            self.chunks.append(chunk)
            count += 1
        self._persist()
        return count

    def similarity_search(
        self,
        query: str,
        embed_fn: Callable[[str], list[float]],
        k: int = 4,
    ) -> list[Chunk]:
        scored = self.similarity_search_with_scores(query, embed_fn=embed_fn, k=k)
        return [item[1] for item in scored]

    def similarity_search_with_scores(
        self,
        query: str,
        embed_fn: Callable[[str], list[float]],
        k: int = 4,
    ) -> list[tuple[float, Chunk]]:
        if not self.chunks:
            return []
        query_embedding = np.array(embed_fn(query), dtype=np.float32)
        scored = []
        for chunk in self.chunks:
            emb = np.array(chunk.embedding, dtype=np.float32)
            denom = np.linalg.norm(query_embedding) * np.linalg.norm(emb)
            score = float(np.dot(query_embedding, emb) / max(denom, 1e-8))
            if not math.isfinite(score):
                continue
            scored.append((score, chunk))
        scored.sort(key=lambda item: item[0], reverse=True)
        return scored[:k]


class MemoryVectorStore:
    def __init__(self) -> None:
        self.chunks: list[Chunk] = []

    def clear(self) -> None:
        self.chunks = []

    def add_texts(
        self,
        source: str,
        texts: list[str],
        embed_fn: Callable[[str], list[float]],
    ) -> int:
        count = 0
        for idx, text in enumerate(texts):
            payload = text.strip()
            if not payload:
                continue
            chunk = Chunk(
                id=f"{source}-{len(self.chunks)+idx}",
                source=source,
                text=payload,
                embedding=embed_fn(payload),
            )
            self.chunks.append(chunk)
            count += 1
        return count

    def similarity_search(
        self,
        query: str,
        embed_fn: Callable[[str], list[float]],
        k: int = 4,
    ) -> list[Chunk]:
        scored = self.similarity_search_with_scores(query, embed_fn=embed_fn, k=k)
        return [item[1] for item in scored]

    def similarity_search_with_scores(
        self,
        query: str,
        embed_fn: Callable[[str], list[float]],
        k: int = 4,
    ) -> list[tuple[float, Chunk]]:
        if not self.chunks:
            return []
        query_embedding = np.array(embed_fn(query), dtype=np.float32)
        scored = []
        for chunk in self.chunks:
            emb = np.array(chunk.embedding, dtype=np.float32)
            denom = np.linalg.norm(query_embedding) * np.linalg.norm(emb)
            score = float(np.dot(query_embedding, emb) / max(denom, 1e-8))
            if not math.isfinite(score):
                continue
            scored.append((score, chunk))
        scored.sort(key=lambda item: item[0], reverse=True)
        return scored[:k]


def split_text(text: str, chunk_size: int = 900, overlap: int = 120) -> list[str]:
    normalized = " ".join(text.replace("\r", "\n").split())
    if len(normalized) <= chunk_size:
        return [normalized] if normalized else []
    chunks = []
    start = 0
    while start < len(normalized):
        end = start + chunk_size
        chunks.append(normalized[start:end])
        if end >= len(normalized):
            break
        start = max(end - overlap, start + 1)
    return chunks
