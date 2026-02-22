from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np

try:
    from openviking import SyncOpenViking
    from openviking_cli.utils.config.open_viking_config import OpenVikingConfigSingleton
except Exception:  # pragma: no cover - optional dependency
    SyncOpenViking = None
    OpenVikingConfigSingleton = None


def _env_flag(name: str, default: str = "true") -> bool:
    return os.getenv(name, default).strip().lower() in {"1", "true", "yes", "on"}


def _pick_api_key() -> str:
    return (
        os.getenv("DASHSCOPE_API_KEY")
        or os.getenv("BAILIAN_API_KEY")
        or os.getenv("APIKEY")
        or os.getenv("apikey")
        or os.getenv("OPENAI_API_KEY")
        or ""
    ).strip()


_NATIVE_LOCK = threading.Lock()
_NATIVE_CLIENT: Any | None = None
_NATIVE_INIT_ERROR: str = ""


def _native_storage_path() -> Path:
    raw = os.getenv("OPENVIKING_NATIVE_STORAGE_PATH", "")
    if raw.strip():
        return Path(raw).expanduser().resolve()
    return (Path.cwd() / "data" / "openviking_native").resolve()


def _native_dimension() -> int:
    raw = os.getenv("OPENVIKING_NATIVE_DIM", "")
    if raw.strip().isdigit():
        return max(int(raw), 64)
    model = os.getenv("EMBEDDING_MODEL", "")
    if "v3" in model:
        return 1024
    return 1024


def _build_native_config(path: Path) -> dict[str, Any]:
    api_key = _pick_api_key()
    api_base = os.getenv("BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
    model = os.getenv("EMBEDDING_MODEL", "text-embedding-v3")
    dim = _native_dimension()
    return {
        "storage": {
            "agfs": {
                "backend": "local",
                "path": str(path),
                "port": max(int(os.getenv("OPENVIKING_NATIVE_AGFS_PORT", "1833")), 1024),
            },
            "vectordb": {
                "backend": "local",
                "path": str(path),
                "dimension": dim,
            },
        },
        "embedding": {
            "dense": {
                "provider": "openai",
                "model": model,
                "api_key": api_key,
                "api_base": api_base,
                "dimension": dim,
            }
        },
        "rerank": {},
    }


def _get_native_client() -> Any | None:
    global _NATIVE_CLIENT, _NATIVE_INIT_ERROR
    if not _env_flag("OPENVIKING_NATIVE_ENABLED", "true"):
        return None
    if SyncOpenViking is None or OpenVikingConfigSingleton is None:
        return None
    if _NATIVE_CLIENT is not None:
        return _NATIVE_CLIENT
    if _NATIVE_INIT_ERROR:
        return None

    with _NATIVE_LOCK:
        if _NATIVE_CLIENT is not None:
            return _NATIVE_CLIENT
        if _NATIVE_INIT_ERROR:
            return None
        try:
            store_path = _native_storage_path()
            store_path.mkdir(parents=True, exist_ok=True)
            api_key = _pick_api_key()
            if not api_key:
                raise RuntimeError("missing_api_key")
            OpenVikingConfigSingleton.initialize(config_dict=_build_native_config(store_path))
            _NATIVE_CLIENT = SyncOpenViking(path=str(store_path))
            return _NATIVE_CLIENT
        except Exception as exc:  # pragma: no cover - best effort init
            _NATIVE_INIT_ERROR = f"{type(exc).__name__}: {exc}"
            return None


@dataclass
class LayerItem:
    id: str
    source: str
    layer: str  # L0 | L1
    text: str
    embedding: list[float]
    created_at: float
    updated_at: float
    title: str = ""
    refs: list[str] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)


class OpenVikingStore:
    """Persistent/in-memory store for layered memories (legacy + native OpenViking)."""

    def __init__(self, db_path: Path | None = None, namespace: str | None = None) -> None:
        self.db_path = db_path
        self.namespace = self._normalize_namespace(namespace or self._derive_namespace())
        self.items: list[LayerItem] = []
        self._matrix_cache: dict[str, np.ndarray] = {}
        self._native_client = _get_native_client()
        self._native_sources: dict[str, dict[str, Any]] = {}
        self._native_index_path = (
            db_path.with_name("openviking_native_index.json") if db_path is not None else None
        )
        self._load()
        self._load_native_index()

    @property
    def native_enabled(self) -> bool:
        return self._native_client is not None

    @property
    def native_error(self) -> str:
        return _NATIVE_INIT_ERROR

    def _derive_namespace(self) -> str:
        if self.db_path is None:
            return "session"
        parent = self.db_path.parent.name or "user"
        return f"user-{parent}"

    @staticmethod
    def _normalize_namespace(raw: str) -> str:
        text = re.sub(r"[^a-zA-Z0-9._-]+", "-", str(raw or "").strip()).strip("-")
        return text[:80] if text else "session"

    @staticmethod
    def _source_slug(source: str) -> str:
        base = re.sub(r"[^a-zA-Z0-9._-]+", "-", str(source or "").strip()).strip("-")
        digest = hashlib.sha1(str(source).encode("utf-8")).hexdigest()[:12]
        if not base:
            base = "source"
        return f"{base[:48]}-{digest}"

    @property
    def _native_base_uri(self) -> str:
        return f"viking://resources/{self.namespace}"

    @property
    def base_uri(self) -> str:
        return self._native_base_uri

    def _native_target_uri(self, source: str) -> str:
        return f"{self._native_base_uri}/{self._source_slug(source)}"

    @staticmethod
    def _uri_matches_prefix(uri: str, prefix: str) -> bool:
        px = str(prefix or "").rstrip("/")
        u = str(uri or "")
        return bool(px) and (u == px or u.startswith(px + "/"))

    def _native_match_source(self, uri: str) -> str | None:
        best_source = None
        best_len = -1
        for source, info in self._native_sources.items():
            root = str(info.get("root_uri") or info.get("target_uri") or "")
            if root and self._uri_matches_prefix(uri, root):
                if len(root) > best_len:
                    best_source = source
                    best_len = len(root)
        return best_source

    def _load(self) -> None:
        if self.db_path is None or not self.db_path.exists():
            return
        try:
            raw = json.loads(self.db_path.read_text(encoding="utf-8"))
            loaded: list[LayerItem] = []
            for item in raw:
                loaded.append(
                    LayerItem(
                        id=str(item.get("id", "")),
                        source=str(item.get("source", "")),
                        layer=str(item.get("layer", "L1")).upper(),
                        text=str(item.get("text", "")),
                        embedding=list(item.get("embedding", [])),
                        created_at=float(item.get("created_at", 0.0) or 0.0),
                        updated_at=float(item.get("updated_at", 0.0) or 0.0),
                        title=str(item.get("title", "") or ""),
                        refs=[str(x) for x in (item.get("refs") or []) if str(x)],
                        meta=dict(item.get("meta") or {}),
                    )
                )
            self.items = [x for x in loaded if x.layer in {"L0", "L1"} and x.text.strip()]
            self._matrix_cache = {}
        except Exception:
            self.items = []
            self._matrix_cache = {}

    def _persist(self) -> None:
        if self.db_path is None:
            return
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.db_path.write_text(
            json.dumps([asdict(i) for i in self.items], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def _load_native_index(self) -> None:
        if self._native_index_path is None or not self._native_index_path.exists():
            return
        try:
            raw = json.loads(self._native_index_path.read_text(encoding="utf-8"))
            if isinstance(raw, dict):
                self._native_sources = {
                    str(k): dict(v)
                    for k, v in raw.items()
                    if str(k)
                    and isinstance(v, dict)
                    and (v.get("root_uri") or v.get("target_uri"))
                }
        except Exception:
            self._native_sources = {}

    def _persist_native_index(self) -> None:
        if self._native_index_path is None:
            return
        self._native_index_path.parent.mkdir(parents=True, exist_ok=True)
        self._native_index_path.write_text(
            json.dumps(self._native_sources, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def _invalidate(self) -> None:
        self._matrix_cache = {}

    def _remove_source_legacy(self, source: str) -> int:
        src = str(source)
        before = len(self.items)
        self.items = [it for it in self.items if it.source != src]
        removed = before - len(self.items)
        if removed > 0:
            self._invalidate()
            self._persist()
        return removed

    def _remove_source_native(self, source: str) -> int:
        src = str(source)
        info = self._native_sources.pop(src, None)
        if not info:
            return 0
        root_uri = str(info.get("root_uri") or info.get("target_uri") or "")
        if self._native_client is not None and root_uri:
            try:  # pragma: no cover - remote/client side behavior
                self._native_client.rm(root_uri, recursive=True)
            except Exception:
                pass
        self._persist_native_index()
        return max(int(info.get("l1_count", 0)), 0) + 1

    def clear(self) -> None:
        self.items = []
        self._invalidate()
        self._persist()

        if self._native_sources:
            for info in list(self._native_sources.values()):
                root_uri = str(info.get("root_uri") or info.get("target_uri") or "")
                if self._native_client is not None and root_uri:
                    try:  # pragma: no cover - remote/client side behavior
                        self._native_client.rm(root_uri, recursive=True)
                    except Exception:
                        pass
            self._native_sources = {}
            self._persist_native_index()

    def remove_source(self, source: str) -> int:
        removed = self._remove_source_legacy(source)
        removed += self._remove_source_native(source)
        return removed

    def _upsert_source_layers_legacy(
        self,
        source: str,
        l0_text: str,
        l1_items: list[dict[str, Any]],
        embed_fn: Callable[[str], list[float]],
    ) -> dict[str, int]:
        source = str(source)
        self._remove_source_legacy(source)
        now = time.time()
        added_l1 = 0

        for idx, item in enumerate(l1_items, start=1):
            text = str(item.get("summary", "") or item.get("text", "")).strip()
            if not text:
                continue
            title = str(item.get("title", "") or "").strip()
            refs = [str(x) for x in (item.get("refs") or []) if str(x)]
            self.items.append(
                LayerItem(
                    id=f"{source}::L1::{idx}",
                    source=source,
                    layer="L1",
                    title=title,
                    text=text,
                    embedding=embed_fn(text),
                    created_at=now,
                    updated_at=now,
                    refs=refs,
                    meta={"index": idx},
                )
            )
            added_l1 += 1

        l0_payload = str(l0_text).strip()
        added_l0 = 0
        if l0_payload:
            self.items.append(
                LayerItem(
                    id=f"{source}::L0",
                    source=source,
                    layer="L0",
                    title="summary",
                    text=l0_payload,
                    embedding=embed_fn(l0_payload),
                    created_at=now,
                    updated_at=now,
                    refs=[f"{source}::L1::{i}" for i in range(1, added_l1 + 1)],
                    meta={},
                )
            )
            added_l0 = 1

        self._invalidate()
        self._persist()
        return {"l0": added_l0, "l1": added_l1}

    def _upsert_source_layers_native(
        self,
        source: str,
        l0_text: str,
        l1_items: list[dict[str, Any]],
        l2_texts: list[str] | None = None,
    ) -> dict[str, int]:
        if self._native_client is None:
            return {"l0": 0, "l1": 0}

        self._remove_source_native(source)

        l1_clean = []
        for item in l1_items:
            title = str(item.get("title", "") or "").strip()
            summary = str(item.get("summary", "") or item.get("text", "")).strip()
            if summary:
                l1_clean.append((title, summary))

        lines = [f"# Source: {source}", "", "## L0"]
        lines.append(str(l0_text).strip() or "No abstract.")
        lines.append("")
        lines.append("## L1")
        if not l1_clean:
            lines.append("- No overview items.")
        else:
            for idx, (title, summary) in enumerate(l1_clean, start=1):
                header = title or f"Section {idx}"
                lines.append(f"### {header}")
                lines.append(summary)
                lines.append("")
        lines.append("## L2")
        l2_clean = [str(x).strip() for x in (l2_texts or []) if str(x).strip()]
        if not l2_clean:
            lines.append("- No source chunks.")
        else:
            for idx, text in enumerate(l2_clean, start=1):
                lines.append(f"### Chunk {idx}")
                lines.append(text)
                lines.append("")
        payload = "\n".join(lines).strip() + "\n"

        fd, tmp_name = tempfile.mkstemp(suffix=".md", prefix="openviking_")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fout:
                fout.write(payload)
            target_uri = self._native_target_uri(source)
            result = self._native_client.add_resource(
                tmp_name,
                target=target_uri,
                reason="upload_ingest",
                instruction="preserve key medical facts",
                wait=True,
                timeout=120.0,
            )
            root_uri = str((result or {}).get("root_uri") or target_uri)
            self._native_sources[str(source)] = {
                "target_uri": target_uri,
                "root_uri": root_uri,
                "l1_count": len(l1_clean),
                "l2_count": len(l2_clean),
                "updated_at": time.time(),
            }
            self._persist_native_index()
            return {"l0": 1 if str(l0_text).strip() else 0, "l1": len(l1_clean)}
        finally:
            try:
                os.remove(tmp_name)
            except Exception:
                pass

    def upsert_source_layers(
        self,
        source: str,
        l0_text: str,
        l1_items: list[dict[str, Any]],
        embed_fn: Callable[[str], list[float]],
        l2_texts: list[str] | None = None,
    ) -> dict[str, int]:
        dual_write_legacy = _env_flag("OPENVIKING_LEGACY_DUAL_WRITE", "false")
        legacy_stats = {"l0": 0, "l1": 0}
        if dual_write_legacy:
            legacy_stats = self._upsert_source_layers_legacy(source, l0_text, l1_items, embed_fn)
        if self._native_client is None:
            if not dual_write_legacy:
                return self._upsert_source_layers_legacy(source, l0_text, l1_items, embed_fn)
            return legacy_stats
        try:
            native_stats = self._upsert_source_layers_native(
                source,
                l0_text,
                l1_items,
                l2_texts=l2_texts,
            )
            if int(native_stats.get("l0", 0)) > 0 or int(native_stats.get("l1", 0)) > 0:
                return native_stats
        except Exception:  # pragma: no cover - native path is best effort
            pass
        if not dual_write_legacy:
            return self._upsert_source_layers_legacy(source, l0_text, l1_items, embed_fn)
        return legacy_stats

    def list_by_layer(self, layer: str) -> list[LayerItem]:
        key = str(layer).upper()
        return [it for it in self.items if it.layer == key]

    def list_by_source(self, source: str, layer: str | None = None) -> list[LayerItem]:
        src = str(source)
        if not layer:
            return [it for it in self.items if it.source == src]
        layer_key = str(layer).upper()
        return [it for it in self.items if it.source == src and it.layer == layer_key]

    def stats(self) -> dict[str, int]:
        if self._native_sources:
            l0 = len(self._native_sources)
            l1 = sum(max(int(v.get("l1_count", 0)), 0) for v in self._native_sources.values())
            if l1 == 0:
                l1 = l0
            return {
                "total": l0 + l1,
                "l0": l0,
                "l1": l1,
                "sources": l0,
            }

        l0 = sum(1 for it in self.items if it.layer == "L0")
        l1 = sum(1 for it in self.items if it.layer == "L1")
        return {
            "total": len(self.items),
            "l0": l0,
            "l1": l1,
            "sources": len({it.source for it in self.items}),
        }

    def _layer_matrix(self, layer: str) -> tuple[list[LayerItem], np.ndarray]:
        key = str(layer).upper()
        items = self.list_by_layer(key)
        if not items:
            return [], np.zeros((0, 0), dtype=np.float32)
        cached = self._matrix_cache.get(key)
        if cached is None or cached.shape[0] != len(items):
            cached = np.array([it.embedding for it in items], dtype=np.float32)
            self._matrix_cache[key] = cached
        return items, cached

    @staticmethod
    def _cosine_scores(query_embedding: np.ndarray, matrix: np.ndarray) -> np.ndarray:
        if matrix.size == 0:
            return np.array([], dtype=np.float32)
        norms = np.linalg.norm(matrix, axis=1)
        q_norm = np.linalg.norm(query_embedding)
        denom = np.maximum(norms * q_norm, 1e-8)
        return (matrix @ query_embedding) / denom

    def similarity_search_layer_by_embedding(
        self,
        query_embedding: list[float] | np.ndarray,
        layer: str,
        k: int = 4,
    ) -> list[tuple[float, LayerItem]]:
        items, matrix = self._layer_matrix(layer)
        if not items:
            return []
        q = np.array(query_embedding, dtype=np.float32)
        scores = self._cosine_scores(q, matrix)
        if scores.size == 0:
            return []
        idx = np.argsort(scores)[::-1][: max(int(k), 1)]
        out: list[tuple[float, LayerItem]] = []
        for i in idx:
            s = float(scores[i])
            if not np.isfinite(s):
                continue
            out.append((s, items[int(i)]))
        return out

    def score_items_by_embedding(
        self,
        query_embedding: list[float] | np.ndarray,
        items: list[LayerItem],
        k: int = 4,
    ) -> list[tuple[float, LayerItem]]:
        if not items:
            return []
        q = np.array(query_embedding, dtype=np.float32)
        matrix = np.array([it.embedding for it in items], dtype=np.float32)
        scores = self._cosine_scores(q, matrix)
        idx = np.argsort(scores)[::-1][: max(int(k), 1)]
        out: list[tuple[float, LayerItem]] = []
        for i in idx:
            s = float(scores[i])
            if not np.isfinite(s):
                continue
            out.append((s, items[int(i)]))
        return out

    def _native_find_resources(self, query: str, limit: int) -> list[Any]:
        if self._native_client is None:
            return []
        try:
            result = self._native_client.find(
                query=str(query),
                target_uri=self._native_base_uri,
                limit=max(int(limit), 1),
            )
        except Exception:
            return []

        out = []
        for ctx in list(getattr(result, "resources", []) or []):
            uri = str(getattr(ctx, "uri", "") or "")
            if not uri:
                continue
            if self._uri_matches_prefix(uri, self._native_base_uri) or self._native_match_source(uri):
                out.append(ctx)
        return out

    @staticmethod
    def _ctx_attr(ctx: Any, key: str, default: Any = "") -> Any:
        if isinstance(ctx, dict):
            return ctx.get(key, default)
        return getattr(ctx, key, default)

    def _normalize_resources(self, result: Any) -> list[dict[str, Any]]:
        resources = []
        if isinstance(result, dict):
            resources = result.get("resources") or result.get("items") or []
        else:
            resources = list(getattr(result, "resources", []) or [])
        out: list[dict[str, Any]] = []
        for ctx in resources:
            uri = str(self._ctx_attr(ctx, "uri", "") or "")
            if not uri:
                continue
            if not self._uri_matches_prefix(uri, self._native_base_uri) and not self._native_match_source(uri):
                continue
            out.append(
                {
                    "uri": uri,
                    "score": float(self._ctx_attr(ctx, "score", 0.0) or 0.0),
                    "is_leaf": bool(self._ctx_attr(ctx, "is_leaf", False)),
                    "abstract": str(self._ctx_attr(ctx, "abstract", "") or "").strip(),
                    "overview": str(self._ctx_attr(ctx, "overview", "") or "").strip(),
                }
            )
        return out

    def session(self, session_id: str | None = None) -> Any | None:
        if self._native_client is None:
            return None
        try:
            return self._native_client.session(session_id=session_id)
        except Exception:
            return None

    def wait_processed(self, timeout: float | None = None) -> dict[str, Any]:
        if self._native_client is None:
            return {"ok": False, "reason": "native_unavailable"}
        try:
            out = self._native_client.wait_processed(timeout=timeout)
            if isinstance(out, dict):
                return out
            return {"ok": True}
        except Exception as exc:
            return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}

    def find(
        self,
        query: str,
        target_uri: str | None = None,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        if self._native_client is None:
            return []
        try:
            result = self._native_client.find(
                query=str(query or ""),
                target_uri=str(target_uri or self._native_base_uri),
                limit=max(int(limit), 1),
            )
        except Exception:
            return []
        return self._normalize_resources(result)

    def search(
        self,
        query: str,
        target_uri: str | None = None,
        session: Any | None = None,
        filters: dict[str, Any] | None = None,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        if self._native_client is None:
            return []
        kwargs: dict[str, Any] = {
            "query": str(query or ""),
            "target_uri": str(target_uri or self._native_base_uri),
            "limit": max(int(limit), 1),
        }
        if session is not None:
            kwargs["session"] = session
        if filters:
            kwargs["filter"] = filters
        try:
            result = self._native_client.search(**kwargs)
        except Exception:
            # Best-effort fallback: semantic find without multi-step planning.
            return self.find(query=query, target_uri=target_uri, limit=limit)
        return self._normalize_resources(result)

    def overview(self, uri: str) -> str:
        if self._native_client is None:
            return ""
        try:
            text = self._native_client.overview(str(uri or ""))
            return str(text or "").strip()
        except Exception:
            return ""

    def read(self, uri: str) -> str:
        if self._native_client is None:
            return ""
        try:
            text = self._native_client.read(str(uri or ""))
            return str(text or "").strip()
        except Exception:
            return ""

    def ls(self, uri: str) -> list[dict[str, Any]]:
        if self._native_client is None:
            return []
        try:
            rows = self._native_client.ls(str(uri or ""))
        except Exception:
            return []
        out: list[dict[str, Any]] = []
        for row in list(rows or []):
            if isinstance(row, dict):
                ru = str(row.get("uri", "") or row.get("path", "")).strip()
                out.append({"uri": ru, "is_leaf": bool(row.get("is_leaf", False)), "raw": row})
                continue
            ru = str(getattr(row, "uri", "") or getattr(row, "path", "") or "").strip()
            out.append({"uri": ru, "is_leaf": bool(getattr(row, "is_leaf", False)), "raw": row})
        return [x for x in out if x.get("uri")]

    def glob(self, pattern: str, root_uri: str | None = None) -> list[str]:
        if self._native_client is None:
            return []
        try:
            out = self._native_client.glob(str(pattern or ""), uri=str(root_uri or self._native_base_uri))
        except Exception:
            return []
        if isinstance(out, dict):
            rows = out.get("matches") or out.get("uris") or out.get("items") or []
            return [str(x).strip() for x in rows if str(x).strip()]
        if isinstance(out, list):
            return [str(x).strip() for x in out if str(x).strip()]
        return []

    def _native_directory_text(self, uri: str, abstract: str, overview: str) -> str:
        if overview:
            return overview
        if abstract:
            return abstract
        if self._native_client is None:
            return ""
        try:
            value = self._native_client.overview(uri)
            if value:
                return str(value).strip()
        except Exception:
            pass
        try:
            value = self._native_client.abstract(uri)
            if value:
                return str(value).strip()
        except Exception:
            pass
        return ""

    def _native_leaf_text(self, uri: str, abstract: str) -> str:
        if abstract:
            return abstract
        if self._native_client is None:
            return ""
        try:
            value = self._native_client.read(uri)
            text = str(value or "").strip()
            return text[:420]
        except Exception:
            return ""

    def _search_layer_native(
        self,
        query: str,
        layer: str,
        k: int = 4,
        sources: list[str] | None = None,
    ) -> list[tuple[float, LayerItem]]:
        if self._native_client is None:
            return []

        candidates = self._native_find_resources(str(query or ""), limit=max(int(k) * 10, 12))
        if not candidates:
            return []

        source_filter = {str(s) for s in (sources or []) if str(s)}
        now = time.time()

        if str(layer).upper() == "L0":
            best: dict[str, tuple[float, LayerItem]] = {}
            for ctx in candidates:
                uri = str(getattr(ctx, "uri", "") or "")
                source = self._native_match_source(uri)
                if not source:
                    continue
                if source_filter and source not in source_filter:
                    continue
                score = float(getattr(ctx, "score", 0.0) or 0.0)
                abstract = str(getattr(ctx, "abstract", "") or "").strip()
                overview = str(getattr(ctx, "overview", "") or "").strip()
                text = self._native_directory_text(uri, abstract=abstract, overview=overview)
                if not text:
                    continue
                item = LayerItem(
                    id=f"{source}::L0::native",
                    source=source,
                    layer="L0",
                    title="summary",
                    text=text,
                    embedding=[],
                    created_at=now,
                    updated_at=now,
                    refs=[],
                    meta={"uri": uri},
                )
                prev = best.get(source)
                if prev is None or score > prev[0]:
                    best[source] = (score, item)
            ranked = sorted(best.values(), key=lambda x: x[0], reverse=True)
            return ranked[: max(int(k), 1)]

        seen = set()
        scored: list[tuple[float, LayerItem]] = []
        for ctx in candidates:
            uri = str(getattr(ctx, "uri", "") or "")
            if not uri or uri in seen:
                continue
            seen.add(uri)
            source = self._native_match_source(uri)
            if not source:
                continue
            if source_filter and source not in source_filter:
                continue

            is_leaf = bool(getattr(ctx, "is_leaf", False))
            score = float(getattr(ctx, "score", 0.0) or 0.0)
            abstract = str(getattr(ctx, "abstract", "") or "").strip()
            overview = str(getattr(ctx, "overview", "") or "").strip()

            if is_leaf:
                text = self._native_leaf_text(uri, abstract=abstract)
            else:
                text = self._native_directory_text(uri, abstract=abstract, overview=overview)
            if not text:
                continue

            title = uri.rstrip("/").split("/")[-1] if uri else ""
            scored.append(
                (
                    score,
                    LayerItem(
                        id=f"{source}::L1::native::{hashlib.sha1(uri.encode('utf-8')).hexdigest()[:10]}",
                        source=source,
                        layer="L1",
                        title=title,
                        text=text,
                        embedding=[],
                        created_at=now,
                        updated_at=now,
                        refs=[],
                        meta={"uri": uri, "is_leaf": is_leaf},
                    ),
                )
            )

        scored.sort(key=lambda x: x[0], reverse=True)
        return scored[: max(int(k), 1)]

    def search_layer(
        self,
        query: str,
        query_embedding: list[float] | np.ndarray,
        layer: str,
        k: int = 4,
        sources: list[str] | None = None,
    ) -> list[tuple[float, LayerItem]]:
        key = str(layer).upper()

        native_hits = self._search_layer_native(query=query, layer=key, k=k, sources=sources)
        if native_hits:
            return native_hits

        if key == "L0":
            return self.similarity_search_layer_by_embedding(query_embedding, layer="L0", k=k)

        l1_candidates = self.list_by_layer("L1")
        if sources:
            source_set = {str(s) for s in sources if str(s)}
            if source_set:
                l1_candidates = [it for it in l1_candidates if it.source in source_set]
        return self.score_items_by_embedding(query_embedding, l1_candidates, k=k)
