from __future__ import annotations

import json
import re
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


# ---------------------------------------------------------------------------
# Shared vectorised cosine-similarity helper
# ---------------------------------------------------------------------------

def _batch_cosine_scores(
    query_embedding: np.ndarray,
    matrix: np.ndarray,
) -> np.ndarray:
    """Return cosine similarities between *query_embedding* and every row in *matrix*.

    Uses a single matrix–vector product – O(n·d) but fully vectorised and
    much faster than a Python-level loop.
    """
    if matrix.size == 0:
        return np.array([], dtype=np.float32)
    norms = np.linalg.norm(matrix, axis=1)       # (n,)
    q_norm = np.linalg.norm(query_embedding)      # scalar
    denom = norms * q_norm                         # (n,)
    denom = np.where(denom < 1e-8, 1e-8, denom)   # avoid div-by-zero
    scores = matrix @ query_embedding / denom      # (n,)
    return scores


def _top_k_from_scores(
    scores: np.ndarray,
    chunks: list[Chunk],
    k: int,
) -> list[tuple[float, Chunk]]:
    """Select top-*k* finite-scored chunks, sorted descending."""
    finite_mask = np.isfinite(scores)
    if not np.any(finite_mask):
        return []
    # partial argsort is faster when k << n
    n_valid = int(np.sum(finite_mask))
    actual_k = min(k, n_valid)
    if actual_k <= 0:
        return []
    # np.argpartition is O(n) average for finding top-k
    indices = np.where(finite_mask)[0]
    valid_scores = scores[indices]
    if actual_k < len(valid_scores):
        top_idx = np.argpartition(valid_scores, -actual_k)[-actual_k:]
    else:
        top_idx = np.arange(len(valid_scores))
    top_idx = top_idx[np.argsort(valid_scores[top_idx])[::-1]]
    return [(float(valid_scores[i]), chunks[indices[i]]) for i in top_idx]


class LocalVectorStore:
    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        self.chunks: list[Chunk] = []
        self._matrix: np.ndarray | None = None  # lazy embedding matrix cache
        self._load()

    def _load(self) -> None:
        if not self.db_path.exists():
            return
        raw = json.loads(self.db_path.read_text(encoding="utf-8"))
        self.chunks = [Chunk(**item) for item in raw]
        self._matrix = None  # invalidate cache

    def _persist(self) -> None:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.db_path.write_text(
            json.dumps([asdict(c) for c in self.chunks], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def _get_matrix(self) -> np.ndarray:
        if self._matrix is None or self._matrix.shape[0] != len(self.chunks):
            if not self.chunks:
                self._matrix = np.zeros((0, 0), dtype=np.float32)
            else:
                self._matrix = np.array(
                    [c.embedding for c in self.chunks], dtype=np.float32
                )
        return self._matrix

    def clear(self) -> None:
        self.chunks = []
        self._matrix = None
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
        self._matrix = None  # invalidate cache
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
        matrix = self._get_matrix()
        scores = _batch_cosine_scores(query_embedding, matrix)
        return _top_k_from_scores(scores, self.chunks, k)


class MemoryVectorStore:
    def __init__(self) -> None:
        self.chunks: list[Chunk] = []
        self._matrix: np.ndarray | None = None

    def clear(self) -> None:
        self.chunks = []
        self._matrix = None

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
        self._matrix = None  # invalidate cache
        return count

    def _get_matrix(self) -> np.ndarray:
        if self._matrix is None or self._matrix.shape[0] != len(self.chunks):
            if not self.chunks:
                self._matrix = np.zeros((0, 0), dtype=np.float32)
            else:
                self._matrix = np.array(
                    [c.embedding for c in self.chunks], dtype=np.float32
                )
        return self._matrix

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
        matrix = self._get_matrix()
        scores = _batch_cosine_scores(query_embedding, matrix)
        return _top_k_from_scores(scores, self.chunks, k)


# ---------------------------------------------------------------------------
#  Semantic-aware text splitter
# ---------------------------------------------------------------------------

# Patterns that indicate a structural boundary (paragraph, heading, table row)
_PARAGRAPH_SEP = re.compile(r"\n{2,}")
_SENTENCE_SEP = re.compile(r"(?<=[。！？；;!?])\s*")
_TABLE_ROW = re.compile(r"^\|.*\|$", re.MULTILINE)
_HEADING = re.compile(r"^#{1,6}\s", re.MULTILINE)


def _is_table_block(text: str) -> bool:
    """Heuristic: more than half the lines look like markdown table rows."""
    lines = [l for l in text.strip().splitlines() if l.strip()]
    if not lines:
        return False
    table_lines = sum(1 for l in lines if l.strip().startswith("|") and l.strip().endswith("|"))
    return table_lines >= len(lines) * 0.5


def split_text(text: str, chunk_size: int = 900, overlap: int = 120) -> list[str]:
    """Split *text* into chunks respecting paragraph / sentence boundaries.

    Improvements over the previous fixed-window approach:
    1. First split on paragraph boundaries (double newline).
    2. If a paragraph still exceeds *chunk_size*, split on sentence boundaries.
    3. Table blocks (markdown ``|...|`` rows) are kept together when possible.
    4. Overlap is created by repeating the tail of the previous chunk.
    """
    if not text or not text.strip():
        return []

    # Normalise whitespace within lines but preserve paragraph breaks.
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # Split into paragraph-level segments.
    paragraphs = _PARAGRAPH_SEP.split(text)
    segments: list[str] = []
    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        if len(para) <= chunk_size or _is_table_block(para):
            segments.append(para)
        else:
            # Further split long paragraphs on sentence boundaries.
            sentences = _SENTENCE_SEP.split(para)
            for sent in sentences:
                sent = sent.strip()
                if sent:
                    segments.append(sent)

    if not segments:
        return []

    # Greedily pack segments into chunks.
    chunks: list[str] = []
    current = ""
    for seg in segments:
        candidate = f"{current} {seg}".strip() if current else seg
        if len(candidate) <= chunk_size:
            current = candidate
        else:
            if current:
                chunks.append(current)
            # If the individual segment is larger than chunk_size, hard-split it.
            if len(seg) > chunk_size:
                start = 0
                while start < len(seg):
                    end = start + chunk_size
                    chunks.append(seg[start:end].strip())
                    if end >= len(seg):
                        break
                    start = max(end - overlap, start + 1)
                current = ""
            else:
                current = seg
    if current:
        chunks.append(current)

    # Build overlapping context between chunks.
    if overlap > 0 and len(chunks) > 1:
        overlapped: list[str] = [chunks[0]]
        for i in range(1, len(chunks)):
            prev_tail = chunks[i - 1][-overlap:]
            merged = f"{prev_tail} {chunks[i]}".strip()
            overlapped.append(merged)
        chunks = overlapped

    return [c for c in chunks if c.strip()]
