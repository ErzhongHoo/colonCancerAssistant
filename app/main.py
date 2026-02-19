from __future__ import annotations

import hashlib
import io
import base64
import json
import mimetypes
import os
import re
import time
import uuid
import numpy as np
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, Request, Response, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from PIL import Image
from pydantic import BaseModel

from app.auth import UserManager
from pypdf import PdfReader

from app.audit import AuditLogger
from app.evidence_guard import verify_answer_with_evidence
from app.literature_agent import (
    LITERATURE_STATE_PATH,
    load_index,
    run_incremental_update,
    search_from_local_store,
)
from app.literature_search import (
    literature_to_context,
    search_literature,
    verify_literature_citations,
)
from app.llm import (
    build_client,
    embed_text,
    extract_date_fields_with_vision_bytes,
    extract_date_with_vision_bytes,
    ocr_with_vision_bytes,
    ocr_with_vision_bytes_transcribe,
)
from app.ocr import (
    draw_redaction_boxes,
    is_paddle_available,
    ocr_with_paddle_bytes,
    ocr_with_paddle_bytes_detailed,
)
from app.privacy import detect_sensitive_types, extract_redaction_preview, redact_sensitive_info
from app.rag import split_text
from app.openviking import OpenVikingStore
from app.timeline import (
    ClinicalEvent,
    build_retrieval_hint,
    build_timeline_summary,
    encode_timeline_state,
    events_to_dict,
    extract_events,
    extract_events_with_llm,
    merge_events,
)
from app.vector_store import VectorStore, build_internal_store, build_session_store, build_user_store, get_vector_backend

try:
    import fitz  # type: ignore
except Exception:
    fitz = None

load_dotenv()

ROOT = Path(__file__).resolve().parent.parent
STATIC_DIR = ROOT / "web"
GUIDELINES_DIR = ROOT / "data" / "guidelines"
INTERNAL_VECTOR_DB = ROOT / "data" / "internal_vector_store.json"
GUIDELINES_VERSION_PATH = ROOT / "data" / "guidelines_versions.json"
AUDIT_LOG_PATH = ROOT / "data" / "audit_log.jsonl"
USER_DB_PATH = ROOT / "data" / "users.db"
USER_DATA_ROOT = ROOT / "data" / "user_data"

CHAT_MODEL = os.getenv("CHAT_MODEL", "qwen-plus")
VISION_MODEL = os.getenv("VISION_MODEL", "qwen-vl-max")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-v3")
SESSION_TTL_SECONDS = int(os.getenv("SESSION_TTL_SECONDS", "1800"))
INTERNAL_RAG_TOKEN = os.getenv("INTERNAL_RAG_TOKEN", "")
ENABLE_UPLOAD_DEID = os.getenv("ENABLE_UPLOAD_DEID", "true").strip().lower() in {"1", "true", "yes", "on"}
OCR_PROVIDER = os.getenv("OCR_PROVIDER", "auto").strip().lower()
PADDLE_OCR_LANG = os.getenv("PADDLE_OCR_LANG", "ch").strip()
PDF_OCR_MAX_PAGES = int(os.getenv("PDF_OCR_MAX_PAGES", "500"))
ENABLE_IMAGE_DATE_VLM = os.getenv("ENABLE_IMAGE_DATE_VLM", "true").strip().lower() in {"1", "true", "yes", "on"}
TIMELINE_ENCODER = os.getenv("TIMELINE_ENCODER", "mamba").strip().lower()
if TIMELINE_ENCODER not in {"linear", "ssm", "mamba"}:
    TIMELINE_ENCODER = "mamba"
ENABLE_LLM_TIMELINE = (
    os.getenv("ENABLE_LLM_TIMELINE", "true").strip().lower() in {"1", "true", "yes", "on"}
)
MANUAL_SESSION_CLEAR_ONLY = (
    os.getenv("MANUAL_SESSION_CLEAR_ONLY", "true").strip().lower() in {"1", "true", "yes", "on"}
)
ENABLE_WEB_LITERATURE = (
    os.getenv("ENABLE_WEB_LITERATURE", "true").strip().lower() in {"1", "true", "yes", "on"}
)
ENABLE_LOCAL_LITERATURE_AGENT = (
    os.getenv("ENABLE_LOCAL_LITERATURE_AGENT", "true").strip().lower() in {"1", "true", "yes", "on"}
)
LITERATURE_AGENT_TOPIC_QUERY = os.getenv(
    "LITERATURE_AGENT_TOPIC_QUERY",
    (
        "(breast cancer OR mammary carcinoma OR breast neoplasm) "
        "AND (clinical OR guideline OR trial OR treatment)"
    ),
).strip()
LITERATURE_AGENT_BOOTSTRAP_MAX_RESULTS = max(int(os.getenv("LITERATURE_AGENT_BOOTSTRAP_MAX_RESULTS", "120")), 20)
LITERATURE_AGENT_REFRESH_HOURS = max(float(os.getenv("LITERATURE_AGENT_REFRESH_HOURS", "24")), 1.0)
LITERATURE_PROVIDERS = [
    p.strip().lower()
    for p in os.getenv("LITERATURE_PROVIDER", "pubmed").split(",")
    if p.strip()
]
LITERATURE_TOP_K = max(int(os.getenv("LITERATURE_TOP_K", "8")), 1)
LITERATURE_TIMEOUT_SECONDS = max(float(os.getenv("LITERATURE_TIMEOUT_SECONDS", "8")), 1.0)
LITERATURE_MEDICAL_ONCOLOGY_ONLY = (
    os.getenv("LITERATURE_MEDICAL_ONCOLOGY_ONLY", "true").strip().lower() in {"1", "true", "yes", "on"}
)
LITERATURE_MIN_RELEVANCE = float(os.getenv("LITERATURE_MIN_RELEVANCE", "0.18"))
AUTO_EVIDENCE_REWRITE = (
    os.getenv("AUTO_EVIDENCE_REWRITE", "true").strip().lower() in {"1", "true", "yes", "on"}
)
EVIDENCE_REWRITE_MIN_COVERAGE = float(os.getenv("EVIDENCE_REWRITE_MIN_COVERAGE", "0.75"))
EVIDENCE_REWRITE_MAX_UNSUPPORTED = max(int(os.getenv("EVIDENCE_REWRITE_MAX_UNSUPPORTED", "1")), 0)
OPENVIKING_ENABLED = os.getenv("OPENVIKING_ENABLED", "true").strip().lower() in {"1", "true", "yes", "on"}
OPENVIKING_USE_LLM = os.getenv("OPENVIKING_USE_LLM", "true").strip().lower() in {"1", "true", "yes", "on"}
OPENVIKING_L0_TOPK = max(int(os.getenv("OPENVIKING_L0_TOPK", "3")), 1)
OPENVIKING_L1_TOPK = max(int(os.getenv("OPENVIKING_L1_TOPK", "4")), 1)
OPENVIKING_L2_TOPK = max(int(os.getenv("OPENVIKING_L2_TOPK", "2")), 0)
OPENVIKING_SUMMARY_INPUT_CHARS = max(int(os.getenv("OPENVIKING_SUMMARY_INPUT_CHARS", "10000")), 1200)
OPENVIKING_EVIDENCE_SNIPPET_MAX_CHARS = max(int(os.getenv("OPENVIKING_EVIDENCE_SNIPPET_MAX_CHARS", "700")), 240)

app = FastAPI(title="Colon Cancer RAG MVP")
client = build_client()
internal_store = build_internal_store(INTERNAL_VECTOR_DB)
audit_logger = AuditLogger(AUDIT_LOG_PATH)
user_manager = UserManager(USER_DB_PATH, USER_DATA_ROOT)

# ---------------------------------------------------------------------------
# Global LRU embedding cache (cross-request)
# ---------------------------------------------------------------------------
from collections import OrderedDict

_EMBED_CACHE_MAX = int(os.getenv("EMBED_CACHE_MAX", "2048"))
_global_embed_cache: OrderedDict[str, list[float]] = OrderedDict()


def _global_embed(text: str) -> list[float]:
    """Embed *text* with global LRU caching."""
    cached = _global_embed_cache.get(text)
    if cached is not None:
        _global_embed_cache.move_to_end(text)
        return cached
    vec = embed_text(client, EMBEDDING_MODEL, text)
    _global_embed_cache[text] = vec
    while len(_global_embed_cache) > _EMBED_CACHE_MAX:
        _global_embed_cache.popitem(last=False)
    return vec

IMAGE_SUFFIXES = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
    ".gif",
    ".tif",
    ".tiff",
    ".heic",
    ".heif",
    ".jfif",
}


def _is_image_input(file_name: str, content_type: str | None) -> tuple[bool, str]:
    guessed_mime = mimetypes.guess_type(file_name)[0] or ""
    mime = content_type or guessed_mime or ""
    suffix = Path(file_name).suffix.lower()
    is_image = mime.startswith("image/") or suffix in IMAGE_SUFFIXES
    if mime == "application/octet-stream" and suffix in IMAGE_SUFFIXES:
        is_image = True
        mime = guessed_mime or "image/jpeg"
    return is_image, (mime or "image/jpeg")


@dataclass
class SessionStoreState:
    store: VectorStore
    updated_at: float


@dataclass
class RetrievedContext:
    id: str
    source: str
    text: str
    layer: str = "L2"
    title: str = ""
    refs: list[str] | None = None


session_stores: dict[str, SessionStoreState] = {}
session_timeline_events: dict[str, list[ClinicalEvent]] = {}
# Per-user persistent stores (lazy loaded)
user_stores: dict[str, VectorStore] = {}
user_timeline_events: dict[str, list[ClinicalEvent]] = {}
session_openviking_stores: dict[str, OpenVikingStore] = {}
user_openviking_stores: dict[str, OpenVikingStore] = {}
literature_index: dict[str, dict[str, object]] = load_index()
literature_last_refresh_at = 0.0
literature_last_refresh_result: dict[str, Any] = {}

if literature_index and LITERATURE_STATE_PATH.exists():
    try:
        state_payload = json.loads(LITERATURE_STATE_PATH.read_text(encoding="utf-8"))
        literature_last_refresh_at = float(state_payload.get("last_run_at", 0) or 0)
    except Exception:
        literature_last_refresh_at = time.time()
elif literature_index:
    literature_last_refresh_at = time.time()


def _refresh_local_literature_if_needed(force: bool = False, max_results: int | None = None) -> dict[str, Any]:
    global literature_index, literature_last_refresh_at, literature_last_refresh_result
    if not ENABLE_LOCAL_LITERATURE_AGENT:
        return {"ok": False, "reason": "local_literature_agent_disabled"}
    now = time.time()
    stale_seconds = LITERATURE_AGENT_REFRESH_HOURS * 3600.0
    should_refresh = force or (not literature_index) or (now - literature_last_refresh_at >= stale_seconds)
    if not should_refresh:
        return {
            "ok": True,
            "skipped": True,
            "reason": "fresh",
            "cached_records": len(literature_index),
            "last_refresh_at": literature_last_refresh_at,
        }
    try:
        result = run_incremental_update(
            topic_query=LITERATURE_AGENT_TOPIC_QUERY,
            embed_fn=lambda t: embed_text(client, EMBEDDING_MODEL, t),
            max_results=max_results or LITERATURE_AGENT_BOOTSTRAP_MAX_RESULTS,
            timeout=LITERATURE_TIMEOUT_SECONDS,
        )
        literature_index = load_index()
        literature_last_refresh_at = now
        literature_last_refresh_result = result
        audit_logger.log(
            event_type="literature_refresh_completed",
            session_id="system",
            details={
                "records": len(literature_index),
                "topic_query": LITERATURE_AGENT_TOPIC_QUERY,
                "force": force,
                **result,
            },
        )
        return {"ok": True, "records": len(literature_index), **result}
    except Exception as exc:
        err = {"ok": False, "error": str(exc), "records": len(literature_index)}
        literature_last_refresh_result = err
        audit_logger.log(
            event_type="literature_refresh_failed",
            session_id="system",
            details={"error": str(exc), "topic_query": LITERATURE_AGENT_TOPIC_QUERY, "force": force},
        )
        return err


def _expire_session(session_id: str, reason: str = "ttl") -> bool:
    removed_store = session_stores.pop(session_id, None)
    removed_events = session_timeline_events.pop(session_id, [])
    removed_openviking = session_openviking_stores.pop(session_id, None)
    removed = removed_store is not None or bool(removed_events)
    if removed_openviking is not None:
        removed = True
    if removed:
        audit_logger.log(
            event_type="session_expired",
            session_id=session_id,
            details={
                "ttl_seconds": SESSION_TTL_SECONDS,
                "dropped_timeline_events": len(removed_events),
                "dropped_openviking_items": removed_openviking.stats().get("total", 0) if removed_openviking else 0,
                "proof": "session_vector_store_removed",
                "reason": reason,
            },
        )
    return removed


def _cleanup_expired_sessions() -> None:
    if MANUAL_SESSION_CLEAR_ONLY:
        return
    now = time.time()
    expired = [
        sid
        for sid, state in session_stores.items()
        if now - state.updated_at > SESSION_TTL_SECONDS
    ]
    for sid in expired:
        _expire_session(sid, reason="ttl")


def _normalize_session_id(raw: str) -> str:
    allowed = []
    for ch in raw[:64]:
        if ch.isalnum() or ch in {"-", "_"}:
            allowed.append(ch)
        else:
            allowed.append("-")
    return "".join(allowed).strip("-")


def _get_session_id(request: Request) -> str:
    incoming = request.headers.get("x-session-id", "").strip()
    session_id = _normalize_session_id(incoming)
    if session_id:
        return session_id
    return f"s-{uuid.uuid4().hex[:16]}"


def _get_auth_user(request: Request):
    """Try to get the authenticated user from the Authorization header."""
    auth_header = request.headers.get("authorization", "").strip()
    if not auth_header.startswith("Bearer "):
        return None, None
    token = auth_header[7:].strip()
    if not token:
        return None, None
    user = user_manager.get_user_by_token(token)
    if user:
        return user, user.user_id
    return None, None


def _get_or_create_user_store(user_id: str) -> VectorStore:
    """Get or create a persistent vector store for an authenticated user."""
    if user_id in user_stores:
        return user_stores[user_id]
    user_dir = user_manager.get_user_data_dir(user_id)
    store = build_user_store(user_dir)
    user_stores[user_id] = store
    return store


def _get_user_store(user_id: str) -> VectorStore | None:
    """Get the persistent vector store for an authenticated user."""
    if user_id in user_stores:
        return user_stores[user_id]
    user_dir = user_manager.get_user_data_dir(user_id)
    store = build_user_store(user_dir)
    user_stores[user_id] = store
    return store


def _load_user_timeline(user_id: str) -> list[ClinicalEvent]:
    """Load timeline events from user persistent storage."""
    if user_id in user_timeline_events:
        return user_timeline_events[user_id]
    user_dir = user_manager.get_user_data_dir(user_id)
    timeline_path = user_dir / "timeline_events.json"
    if timeline_path.exists():
        try:
            raw = json.loads(timeline_path.read_text(encoding="utf-8"))
            events = [ClinicalEvent(**item) for item in raw]
            user_timeline_events[user_id] = events
            return events
        except Exception:
            pass
    user_timeline_events[user_id] = []
    return []


def _save_user_timeline(user_id: str) -> None:
    """Persist timeline events for an authenticated user."""
    events = user_timeline_events.get(user_id, [])
    user_dir = user_manager.get_user_data_dir(user_id)
    timeline_path = user_dir / "timeline_events.json"
    from dataclasses import asdict
    timeline_path.write_text(
        json.dumps([asdict(ev) for ev in events], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _get_or_create_session_store(session_id: str) -> VectorStore:
    _cleanup_expired_sessions()
    now = time.time()
    state = session_stores.get(session_id)
    if state is None:
        state = SessionStoreState(store=build_session_store(), updated_at=now)
        session_stores[session_id] = state
    state.updated_at = now
    return state.store


def _get_session_store(session_id: str) -> VectorStore | None:
    _cleanup_expired_sessions()
    state = session_stores.get(session_id)
    if state is None:
        return None
    state.updated_at = time.time()
    return state.store


def _get_or_create_user_openviking_store(user_id: str) -> OpenVikingStore:
    if user_id in user_openviking_stores:
        return user_openviking_stores[user_id]
    user_dir = user_manager.get_user_data_dir(user_id)
    store = OpenVikingStore(user_dir / "openviking_layers.json", namespace=f"user-{user_id}")
    user_openviking_stores[user_id] = store
    return store


def _get_user_openviking_store(user_id: str) -> OpenVikingStore:
    return _get_or_create_user_openviking_store(user_id)


def _get_or_create_session_openviking_store(session_id: str) -> OpenVikingStore:
    _cleanup_expired_sessions()
    store = session_openviking_stores.get(session_id)
    if store is None:
        store = OpenVikingStore(None, namespace=f"session-{session_id}")
        session_openviking_stores[session_id] = store
    return store


def _get_session_openviking_store(session_id: str) -> OpenVikingStore | None:
    _cleanup_expired_sessions()
    return session_openviking_stores.get(session_id)


def _extract_json_object(raw: str) -> str:
    text = str(raw or "").strip()
    if not text:
        return "{}"
    m = re.search(r"\{[\s\S]*\}", text)
    return m.group(0) if m else text


def _fallback_openviking_layers(source: str, chunks: list[Any]) -> tuple[str, list[dict[str, Any]]]:
    texts = [str(getattr(c, "text", "")).strip() for c in chunks if str(getattr(c, "text", "")).strip()]
    if not texts:
        return ("暂无摘要。", [{"title": "概览", "summary": "暂无可用文本内容。"}])
    merged = "\n".join(texts[:8])
    tnm_hits = list(dict.fromkeys(re.findall(r"[cCpPyYrR]?[Tt][0-4][a-cA-C]?\s*[Nn][0-3][a-cA-C]?\s*[Mm][0-1][a-cA-C]?", merged)))
    stage_hits = list(dict.fromkeys(re.findall(r"(?:IIIC|IIIB|IIIA|IV|III|II|I)\s*期", merged, flags=re.IGNORECASE)))
    diagnosis_line = ""
    for line in re.split(r"[。\n]", merged):
        if any(k in line for k in ("诊断", "乳腺", "癌", "肿瘤")):
            diagnosis_line = line.strip()
            if diagnosis_line:
                break
    l1: list[dict[str, Any]] = []
    if diagnosis_line:
        l1.append({"title": "诊断概览", "summary": diagnosis_line[:220]})
    if tnm_hits or stage_hits:
        l1.append(
            {
                "title": "分期信息",
                "summary": f"TNM: {', '.join(tnm_hits) if tnm_hits else '未见明确TNM'}；分期: {', '.join(stage_hits) if stage_hits else '未见明确分期'}",
            }
        )
    l1.append({"title": "原文片段", "summary": merged[:420]})
    source_name = source.removeprefix("session/")
    l0 = f"{source_name} 病历摘要：{diagnosis_line[:120] if diagnosis_line else '存在肿瘤相关病历信息'}。"
    if tnm_hits:
        l0 += f" 关键TNM：{', '.join(tnm_hits[:2])}。"
    if stage_hits:
        l0 += f" 临床分期：{', '.join(stage_hits[:2])}。"
    return l0[:260], l1[:3]


def _generate_openviking_layers(source: str, chunks: list[Any]) -> tuple[str, list[dict[str, Any]]]:
    fallback_l0, fallback_l1 = _fallback_openviking_layers(source, chunks)
    if not OPENVIKING_USE_LLM:
        return fallback_l0, fallback_l1
    lines: list[str] = []
    total_chars = 0
    max_chars = OPENVIKING_SUMMARY_INPUT_CHARS
    for idx, c in enumerate(chunks, start=1):
        text = str(getattr(c, "text", "")).strip()
        if not text:
            continue
        piece = f"[chunk#{idx}] {text}"
        if total_chars + len(piece) > max_chars:
            remain = max_chars - total_chars
            if remain > 120:
                lines.append(piece[:remain])
            break
        lines.append(piece)
        total_chars += len(piece)
    payload = "\n\n".join(lines).strip()
    if not payload:
        return fallback_l0, fallback_l1

    try:
        resp = client.chat.completions.create(
            model=CHAT_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "你是OpenViking分层摘要器。"
                        "请把L2病历片段整理为可检索的L1/L0，并严格返回JSON。"
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        "仅输出JSON，格式为："
                        "{\"l0_abstract\":\"...\","
                        "\"l1_overviews\":[{\"title\":\"...\",\"summary\":\"...\"}]}\n"
                        "要求：\n"
                        "1) l1_overviews 输出1-3条，每条聚焦不同主题；\n"
                        "2) 优先保留诊断、TNM/分期、治疗方案、影像/病理结论；\n"
                        "3) 不要编造未出现的信息；\n"
                        f"4) source={source}\n\n"
                        f"L2输入:\n{payload}"
                    ),
                },
            ],
            temperature=0.0,
        )
        raw = resp.choices[0].message.content or ""
        parsed = json.loads(_extract_json_object(raw))
        l0 = str(parsed.get("l0_abstract", "") or "").strip()
        l1_raw = parsed.get("l1_overviews", [])
        l1: list[dict[str, Any]] = []
        if isinstance(l1_raw, list):
            for obj in l1_raw[:3]:
                if not isinstance(obj, dict):
                    continue
                title = str(obj.get("title", "") or "").strip()
                summary = str(obj.get("summary", "") or "").strip()
                if not summary:
                    continue
                l1.append({"title": title[:60], "summary": summary[:360]})
        if not l0:
            l0 = fallback_l0
        if not l1:
            l1 = fallback_l1
        return l0[:260], l1[:3]
    except Exception:
        return fallback_l0, fallback_l1


def _upsert_openviking_layers(store: OpenVikingStore, source: str, chunks: list[Any]) -> dict[str, int]:
    if not OPENVIKING_ENABLED:
        return {"l0": 0, "l1": 0}
    l0_text, l1_items = _generate_openviking_layers(source, chunks)
    return store.upsert_source_layers(
        source=source,
        l0_text=l0_text,
        l1_items=l1_items,
        embed_fn=_global_embed,
    )


def _top_chunks_from_sources(
    vector_store: VectorStore | None,
    query_embedding: list[float],
    sources: list[str],
    k: int = 2,
) -> list[tuple[float, Any]]:
    if vector_store is None or not sources:
        return []
    source_set = {str(s) for s in sources if str(s)}
    if not source_set:
        return []
    candidates = [c for c in getattr(vector_store, "chunks", []) if str(getattr(c, "source", "")) in source_set]
    if not candidates:
        return []
    q = np.array(query_embedding, dtype=np.float32)
    matrix = np.array([c.embedding for c in candidates], dtype=np.float32)
    norms = np.linalg.norm(matrix, axis=1)
    qn = np.linalg.norm(q)
    denom = np.maximum(norms * qn, 1e-8)
    scores = (matrix @ q) / denom
    idx = np.argsort(scores)[::-1][: max(int(k), 1)]
    out: list[tuple[float, Any]] = []
    for i in idx:
        out.append((float(scores[i]), candidates[int(i)]))
    return out


def _build_evidence_snippet(text: str, query: str, max_chars: int = OPENVIKING_EVIDENCE_SNIPPET_MAX_CHARS) -> str:
    payload = str(text or "")
    if not payload:
        return ""
    if len(payload) <= max_chars:
        return payload
    tokens = re.findall(r"[A-Za-z0-9_+\-]{2,}|[\u4e00-\u9fff]{2,}", str(query or ""))
    ignored = {"请问", "怎么", "什么", "这个", "那个", "一下", "需要", "建议", "可以", "是否"}
    tokens = [t for t in tokens if t not in ignored]
    for tok in tokens:
        p = payload.lower().find(tok.lower())
        if p >= 0:
            half = max_chars // 2
            start = max(0, p - half)
            end = min(len(payload), start + max_chars)
            return payload[start:end].strip()
    return payload[:max_chars].strip()


def _rebuild_openviking_from_vector_store(
    layered_store: OpenVikingStore,
    vector_store: VectorStore | None,
    reset: bool = True,
) -> dict[str, Any]:
    if vector_store is None:
        return {"sources": 0, "rebuilt": [], "stats": layered_store.stats()}
    source_map: dict[str, list[Any]] = {}
    for chunk in getattr(vector_store, "chunks", []):
        src = str(getattr(chunk, "source", "") or "")
        if not src:
            continue
        source_map.setdefault(src, []).append(chunk)
    if not source_map:
        if reset:
            layered_store.clear()
        return {"sources": 0, "rebuilt": [], "stats": layered_store.stats()}
    if reset:
        layered_store.clear()
    rebuilt: list[dict[str, Any]] = []
    for src in sorted(source_map.keys()):
        chunks = source_map[src]
        stats = _upsert_openviking_layers(layered_store, src, chunks)
        rebuilt.append(
            {
                "source": src,
                "chunks": len(chunks),
                "l0": int(stats.get("l0", 0)),
                "l1": int(stats.get("l1", 0)),
            }
        )
    return {"sources": len(source_map), "rebuilt": rebuilt, "stats": layered_store.stats()}


def _collect_openviking_context(
    query: str,
    query_embedding: list[float],
    user_store: VectorStore | None,
    openviking_store: OpenVikingStore | None,
) -> tuple[list[tuple[float, RetrievedContext]], dict[str, Any]]:
    if not OPENVIKING_ENABLED or openviking_store is None:
        return [], {"enabled": False}
    l0_scored = openviking_store.search_layer(
        query=query,
        query_embedding=query_embedding,
        layer="L0",
        k=OPENVIKING_L0_TOPK,
    )
    if not l0_scored:
        return [], {"enabled": True, "l0_hits": 0}
    candidate_sources: list[str] = []
    for _, item in l0_scored:
        if item.source not in candidate_sources:
            candidate_sources.append(item.source)

    l1_scored = openviking_store.search_layer(
        query=query,
        query_embedding=query_embedding,
        layer="L1",
        k=OPENVIKING_L1_TOPK,
        sources=candidate_sources,
    )
    if not l1_scored:
        l1_scored = openviking_store.search_layer(
            query=query,
            query_embedding=query_embedding,
            layer="L1",
            k=OPENVIKING_L1_TOPK,
        )
    l2_scored = _top_chunks_from_sources(user_store, query_embedding, candidate_sources, k=OPENVIKING_L2_TOPK)

    merged: list[tuple[float, RetrievedContext]] = []
    for score, item in l1_scored:
        title_prefix = f"{item.title}\n" if item.title else ""
        merged.append(
            (
                float(score) + 0.08,
                RetrievedContext(
                    id=item.id,
                    source=item.source,
                    text=f"[L1概览]\n{title_prefix}{item.text}".strip(),
                    layer="L1",
                    title=item.title,
                    refs=item.refs,
                ),
            )
        )
    for score, chunk in l2_scored:
        merged.append(
            (
                float(score),
                RetrievedContext(
                    id=str(getattr(chunk, "id", "")),
                    source=str(getattr(chunk, "source", "")),
                    text=str(getattr(chunk, "text", "")),
                    layer="L2",
                ),
            )
        )

    merged.sort(key=lambda x: x[0], reverse=True)
    return merged[: max(OPENVIKING_L1_TOPK + OPENVIKING_L2_TOPK, 4)], {
        "enabled": True,
        "l0_hits": len(l0_scored),
        "l1_hits": len(l1_scored),
        "l2_hits": len(l2_scored),
        "candidate_sources": candidate_sources,
    }


def _add_timeline_events(session_id: str, source: str, text: str, *, user_id: str | None = None) -> tuple[int, bool]:
    """Extract and add timeline events. If user_id is provided, persist to disk."""
    if user_id:
        existing = _load_user_timeline(user_id)
    else:
        existing = session_timeline_events.get(session_id, [])
    if ENABLE_LLM_TIMELINE:
        extracted = extract_events_with_llm(client, CHAT_MODEL, text, source=source)
    else:
        extracted = extract_events(text, source=source)
    has_known_date = any(str(ev.date).strip() and str(ev.date).strip() != "\u672a\u77e5\u65e5\u671f" for ev in extracted)
    if not extracted:
        return 0, has_known_date
    merged = merge_events(existing, extracted)
    if user_id:
        user_timeline_events[user_id] = merged
        _save_user_timeline(user_id)
    else:
        session_timeline_events[session_id] = merged
    return len(extracted), has_known_date


def _get_timeline_events(session_id: str, *, user_id: str | None = None) -> list[ClinicalEvent]:
    if user_id:
        return _load_user_timeline(user_id)
    _cleanup_expired_sessions()
    return session_timeline_events.get(session_id, [])


def _get_session_expiry(session_id: str, *, user_id: str | None = None) -> dict[str, Any]:
    if user_id:
        # Authenticated users have persistent data, never expires
        return {
            "session_active": True,
            "manual_clear_only": True,
            "ttl_seconds": 0,
            "expires_in_seconds": None,
            "persistent": True,
        }
    _cleanup_expired_sessions()
    state = session_stores.get(session_id)
    if MANUAL_SESSION_CLEAR_ONLY:
        return {
            "session_active": state is not None,
            "manual_clear_only": True,
            "ttl_seconds": SESSION_TTL_SECONDS,
            "expires_in_seconds": None,
        }
    if state is None:
        return {
            "session_active": False,
            "manual_clear_only": False,
            "ttl_seconds": SESSION_TTL_SECONDS,
            "expires_in_seconds": 0,
        }
    now = time.time()
    remaining = max(int(SESSION_TTL_SECONDS - (now - state.updated_at)), 0)
    return {
        "session_active": True,
        "manual_clear_only": False,
        "ttl_seconds": SESSION_TTL_SECONDS,
        "expires_in_seconds": remaining,
    }


def _extract_pdf_text(path: Path) -> str:
    reader = PdfReader(str(path))
    pages = [page.extract_text() or "" for page in reader.pages]
    text = "\n".join(pages).strip()
    if len(text) >= 50:
        return text
    fallback = _extract_pdf_text_with_ocr(path)
    return fallback or text


def _extract_pdf_text_from_bytes(file_bytes: bytes, ocr_mode: str = "clinical") -> str:
    reader = PdfReader(io.BytesIO(file_bytes))
    pages = [page.extract_text() or "" for page in reader.pages]
    text = "\n".join(pages).strip()
    if len(text) >= 50:
        return text
    fallback = _extract_pdf_text_with_ocr_bytes(file_bytes, ocr_mode=ocr_mode)
    return fallback or text


def _extract_pdf_text_for_guideline(path: Path) -> str:
    reader = PdfReader(str(path))
    pages = [page.extract_text() or "" for page in reader.pages]
    text = "\n".join(pages).strip()
    if len(text) >= 50:
        return text
    fallback = _extract_pdf_text_with_ocr(path, ocr_mode="guide")
    return fallback or text


def _extract_pdf_text_with_ocr(path: Path, ocr_mode: str = "clinical") -> str:
    if fitz is None:
        return ""
    doc = fitz.open(str(path))
    extracted = _extract_pdf_doc_with_ocr(doc, ocr_mode=ocr_mode)
    doc.close()
    return extracted


def _extract_pdf_text_with_ocr_bytes(file_bytes: bytes, ocr_mode: str = "clinical") -> str:
    if fitz is None:
        return ""
    doc = fitz.open(stream=file_bytes, filetype="pdf")
    extracted = _extract_pdf_doc_with_ocr(doc, ocr_mode=ocr_mode)
    doc.close()
    return extracted


def _extract_pdf_doc_with_ocr(doc: Any, ocr_mode: str = "clinical") -> str:
    extracted: list[str] = []
    max_pages = min(len(doc), max(PDF_OCR_MAX_PAGES, 1))
    for i in range(max_pages):
        page = doc.load_page(i)
        page_text = _ocr_pdf_page_with_retry(page, ocr_mode=ocr_mode).strip()
        if page_text:
            extracted.append(f"[第{i + 1}页]\n{page_text}")
    return "\n\n".join(extracted).strip()


def _ocr_pdf_page_with_retry(page: Any, ocr_mode: str = "clinical") -> str:
    # Avoid payload size issues by progressively reducing size/quality.
    scales = [1.6, 1.3, 1.1, 0.9, 0.75]
    qualities = [80, 70, 60, 50]
    last_error = ""
    for scale in scales:
        pix = page.get_pixmap(matrix=fitz.Matrix(scale, scale), alpha=False)
        for quality in qualities:
            try:
                image_bytes = pix.tobytes("jpg", jpg_quality=quality)
                if len(image_bytes) > 14 * 1024 * 1024:
                    continue
                return _ocr_image_with_fallback(
                    image_bytes,
                    mime_type="image/jpeg",
                    ocr_mode=ocr_mode,
                )
            except Exception as exc:
                last_error = str(exc)
                continue
    if last_error:
        raise HTTPException(status_code=400, detail=f"PDF OCR 失败: {last_error}")
    raise HTTPException(status_code=400, detail="PDF OCR 失败: 页面过大且无法压缩到可处理范围")


def _extract_raw_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return path.read_text(encoding="gb18030", errors="ignore")


def _extract_raw_text_from_bytes(file_bytes: bytes) -> str:
    try:
        return file_bytes.decode("utf-8")
    except UnicodeDecodeError:
        return file_bytes.decode("gb18030", errors="ignore")


def _extract_file_text(path: Path, content_type: str | None, ocr_mode: str = "clinical") -> str:
    guessed_mime = mimetypes.guess_type(path.name)[0] or ""
    mime = content_type or guessed_mime or ""
    suffix = path.suffix.lower()
    is_image = mime.startswith("image/") or suffix in IMAGE_SUFFIXES
    # Some browsers upload images as application/octet-stream.
    if mime == "application/octet-stream" and suffix in IMAGE_SUFFIXES:
        is_image = True
        mime = guessed_mime or "image/jpeg"

    if suffix == ".pdf" or mime == "application/pdf":
        if ocr_mode == "guide":
            return _extract_pdf_text_for_guideline(path)
        return _extract_pdf_text(path)
    if mime.startswith("text/") or suffix in {".md", ".txt", ".csv"}:
        return _extract_raw_text(path)
    if is_image:
        text, _meta = _extract_image_text_with_date(
            file_bytes=path.read_bytes(),
            mime_type=mime or "image/png",
            ocr_mode=ocr_mode,
        )
        return text
    raise HTTPException(status_code=400, detail=f"不支持的文件类型: {path.name}")


def _extract_file_text_from_bytes(
    file_name: str,
    content_type: str | None,
    file_bytes: bytes,
    ocr_mode: str = "clinical",
) -> tuple[str, dict[str, str]]:
    guessed_mime = mimetypes.guess_type(file_name)[0] or ""
    mime = content_type or guessed_mime or ""
    suffix = Path(file_name).suffix.lower()
    is_image = mime.startswith("image/") or suffix in IMAGE_SUFFIXES
    if mime == "application/octet-stream" and suffix in IMAGE_SUFFIXES:
        is_image = True
        mime = guessed_mime or "image/jpeg"

    if suffix == ".pdf" or mime == "application/pdf":
        return _extract_pdf_text_from_bytes(file_bytes, ocr_mode=ocr_mode), {}
    if mime.startswith("text/") or suffix in {".md", ".txt", ".csv"}:
        return _extract_raw_text_from_bytes(file_bytes), {}
    if is_image:
        return _extract_image_text_with_date(
            file_bytes=file_bytes,
            mime_type=mime or "image/png",
            ocr_mode=ocr_mode,
        )
    raise HTTPException(status_code=400, detail=f"不支持的文件类型: {file_name}")


def _extract_first_iso_date(text: str) -> str | None:
    m = re.search(r"(20\d{2})[-/.年]\s*(\d{1,2})[-/.月]\s*(\d{1,2})", text)
    if m:
        return f"{int(m.group(1)):04d}-{int(m.group(2)):02d}-{int(m.group(3)):02d}"
    m = re.search(r"(?<!\d)(20\d{2})(\d{2})(\d{2})(?!\d)", text)
    if m:
        return f"{int(m.group(1)):04d}-{int(m.group(2)):02d}-{int(m.group(3)):02d}"
    return None


def _normalize_iso_date(raw: str) -> str | None:
    m = re.search(r"(20\d{2})[-/.年]\s*(\d{1,2})[-/.月]\s*(\d{1,2})", raw or "")
    if m:
        return f"{int(m.group(1)):04d}-{int(m.group(2)):02d}-{int(m.group(3)):02d}"
    m = re.search(r"(?<!\d)(20\d{2})(\d{2})(\d{2})(?!\d)", raw or "")
    if m:
        return f"{int(m.group(1)):04d}-{int(m.group(2)):02d}-{int(m.group(3)):02d}"
    return None


def _pick_timeline_date_from_fields(fields: dict[str, str]) -> tuple[str | None, str]:
    exam = _normalize_iso_date(fields.get("exam_date", ""))
    if exam:
        return exam, "exam_date"
    submit = _normalize_iso_date(fields.get("submit_date", ""))
    if submit:
        return submit, "submit_date"
    report = _normalize_iso_date(fields.get("report_date", ""))
    if report:
        return report, "report_date"
    return None, "none"


def _extract_image_text_with_date(
    file_bytes: bytes, mime_type: str, ocr_mode: str = "clinical"
) -> tuple[str, dict[str, str]]:
    text = _ocr_image_with_fallback(file_bytes, mime_type=mime_type, ocr_mode=ocr_mode)
    if ocr_mode != "clinical" or not ENABLE_IMAGE_DATE_VLM:
        return text, {}
    date_from_text = _extract_first_iso_date(text)
    if date_from_text:
        return text, {"timeline_date": date_from_text, "timeline_date_source": "ocr_text"}
    field_meta: dict[str, str] = {}
    try:
        date_fields = extract_date_fields_with_vision_bytes(client, VISION_MODEL, file_bytes, mime_type=mime_type)
        field_meta = {k: str(v) for k, v in date_fields.items()}
        chosen, source = _pick_timeline_date_from_fields(date_fields)
        if chosen:
            date_line = f"检查日期: {chosen}"
            return (
                f"{date_line}\n{text}".strip(),
                {
                    "timeline_date": chosen,
                    "timeline_date_source": source,
                    **field_meta,
                },
            )
    except Exception:
        pass
    try:
        date_raw = extract_date_with_vision_bytes(client, VISION_MODEL, file_bytes, mime_type=mime_type)
    except Exception:
        return text, {"timeline_date": "NA", "timeline_date_source": "not_found", **field_meta}
    m = re.search(r"(20\d{2})-(\d{2})-(\d{2})", date_raw or "")
    if not m:
        return text, {"timeline_date": "NA", "timeline_date_source": "not_found", **field_meta}
    date_line = f"检查日期: {m.group(1)}-{m.group(2)}-{m.group(3)}"
    return (
        f"{date_line}\n{text}".strip(),
        {
            "timeline_date": f"{m.group(1)}-{m.group(2)}-{m.group(3)}",
            "timeline_date_source": "single_date_fallback",
            **field_meta,
        },
    )


def _build_image_redaction_result(file_bytes: bytes, mime_type: str) -> dict[str, Any]:
    if not is_paddle_available():
        return {
            "available": False,
            "engine": "none",
            "reason": "paddle_unavailable",
            "ocr_word_count": 0,
            "sensitive_box_count": 0,
            "boxes": [],
        }
    try:
        detailed = ocr_with_paddle_bytes_detailed(file_bytes, lang=PADDLE_OCR_LANG)
    except Exception as exc:
        return {
            "available": False,
            "engine": "paddle",
            "reason": f"ocr_failed:{exc}",
            "ocr_word_count": 0,
            "sensitive_box_count": 0,
            "boxes": [],
        }
    words = detailed.get("words", []) if isinstance(detailed, dict) else []
    image = Image.open(io.BytesIO(file_bytes))
    width, height = image.size
    boxes: list[dict[str, Any]] = []
    for word in words:
        text = str(word.get("text", "")).strip()
        if not text:
            continue
        hit_types = detect_sensitive_types(text)
        if not hit_types:
            continue
        box = {
            "text": text[:80],
            "types": hit_types,
            "x1": int(word.get("x1", 0)),
            "y1": int(word.get("y1", 0)),
            "x2": int(word.get("x2", 0)),
            "y2": int(word.get("y2", 0)),
            "image_width": int(width),
            "image_height": int(height),
        }
        boxes.append(box)
    if not boxes:
        return {
            "available": False,
            "engine": "paddle",
            "reason": "no_sensitive_boxes",
            "ocr_word_count": len(words),
            "sensitive_box_count": 0,
            "boxes": [],
        }
    masked_bytes = draw_redaction_boxes(file_bytes, boxes)
    data_url = "data:image/jpeg;base64," + base64.b64encode(masked_bytes).decode("ascii")
    return {
        "available": True,
        "engine": "paddle",
        "reason": "ok",
        "ocr_word_count": len(words),
        "sensitive_box_count": len(boxes),
        "boxes": boxes[:80],
        "masked_data_url": data_url,
    }


def _ocr_image_with_fallback(image_bytes: bytes, mime_type: str, ocr_mode: str = "clinical") -> str:
    provider = OCR_PROVIDER
    errors: list[str] = []

    if provider in {"auto", "paddle"}:
        try:
            return ocr_with_paddle_bytes(image_bytes, lang=PADDLE_OCR_LANG)
        except Exception as exc:
            errors.append(f"paddle: {exc}")
            if provider == "paddle":
                raise

    if provider in {"auto", "vision"}:
        try:
            vision_fn = ocr_with_vision_bytes_transcribe if ocr_mode == "guide" else ocr_with_vision_bytes
            return vision_fn(
                client,
                VISION_MODEL,
                image_bytes,
                mime_type=mime_type,
            )
        except Exception as exc:
            errors.append(f"vision: {exc}")
            raise

    if not errors:
        raise RuntimeError(f"不支持的 OCR_PROVIDER={provider}")
    raise RuntimeError("; ".join(errors))


def _ensure_internal_access(request: Request) -> None:
    if not INTERNAL_RAG_TOKEN:
        return
    token = request.headers.get("x-internal-token", "")
    if token != INTERNAL_RAG_TOKEN:
        raise HTTPException(status_code=401, detail="内部接口鉴权失败")


def _format_source(source: str) -> str:
    if source.startswith("internal/"):
        return f"内部指南:{source.removeprefix('internal/')}"
    if source.startswith("session/"):
        return f"用户病例:{source.removeprefix('session/')}"
    return source


def _extract_cited_ranks(answer: str) -> set[int]:
    cited: set[int] = set()
    for match in re.finditer(r"\[证据#(\d+)\]", answer):
        try:
            cited.add(int(match.group(1)))
        except Exception:
            continue
    return cited


def _extract_cited_pages(answer: str) -> set[int]:
    pages: set[int] = set()
    for match in re.finditer(r"\[(?:P|p)\s*(\d{1,4})\]", answer):
        try:
            pages.add(int(match.group(1)))
        except Exception:
            continue
    for match in re.finditer(r"第\s*(\d{1,4})\s*页", answer):
        try:
            pages.add(int(match.group(1)))
        except Exception:
            continue
    return pages


def _extract_page_from_text(text: str) -> int | None:
    """Extract page number from text using multiple pattern strategies."""
    # Pattern 1: Chinese page marker [第N页]
    m = re.search(r"\[第\s*(\d{1,4})\s*页\]", text)
    if m:
        try:
            return int(m.group(1))
        except Exception:
            pass
    # Pattern 2: English page marker [P123] or [p123]
    m = re.search(r"\[(?:P|p|page)\s*(\d{1,4})\]", text, re.IGNORECASE)
    if m:
        try:
            return int(m.group(1))
        except Exception:
            pass
    # Pattern 3: Page header/footer markers like "—12—" or "- 12 -"
    m = re.search(r"[—\-]\s*(\d{1,4})\s*[—\-]", text)
    if m:
        try:
            return int(m.group(1))
        except Exception:
            pass
    # Pattern 4: Explicit "第X页" without brackets
    m = re.search(r"第\s*(\d{1,4})\s*页", text)
    if m:
        try:
            return int(m.group(1))
        except Exception:
            pass
    return None


def _extract_section_from_text(text: str) -> str | None:
    """Try to identify the section/heading that a chunk belongs to."""
    # Look for markdown headings
    m = re.search(r"^(#{1,4})\s+(.+?)$", text, re.MULTILINE)
    if m:
        return m.group(2).strip()[:80]
    # Look for numbered section headers like "3.2 xxx" or "第三章 xxx"
    m = re.search(r"^(\d+\.[\d.]*\s+\S.+?)$", text, re.MULTILINE)
    if m:
        return m.group(1).strip()[:80]
    m = re.search(r"(第[一二三四五六七八九十\d]+[章节条款]\s*\S.+?)[\n。]", text)
    if m:
        return m.group(1).strip()[:80]
    return None


MODEL_DISCLOSURE_PATTERNS = [
    re.compile(r"\bqwen\d*\b", re.IGNORECASE),
    re.compile(r"通义"),
    re.compile(r"千问"),
    re.compile(r"我是.*模型"),
    re.compile(r"模型版本"),
    re.compile(r"大语言模型"),
]


def _sanitize_model_disclosure(answer: str) -> str:
    if not answer.strip():
        return answer
    hit = any(p.search(answer) for p in MODEL_DISCLOSURE_PATTERNS)
    if not hit:
        return answer
    safe_line = "我是医疗助手，会基于你提供的资料给出风险与就医建议。"
    lines = answer.splitlines()
    sanitized: list[str] = []
    for line in lines:
        if any(p.search(line) for p in MODEL_DISCLOSURE_PATTERNS):
            sanitized.append(safe_line)
        else:
            sanitized.append(line)
    out = "\n".join(sanitized).strip()
    out = re.sub(r"(?:Qwen|通义|千问)\S*", "医疗助手", out, flags=re.IGNORECASE)
    return out


def _should_use_external_literature(query: str, timeline_events: list[ClinicalEvent]) -> bool:
    q = (query or "").strip().lower()
    if not q:
        return False

    # --- Negative patterns: suppress false triggers for non-evidence queries ---
    suppress_patterns = ("下载", "在哪里", "链接", "网址", "打不开", "登录", "注册", "密码")
    if any(p in q for p in suppress_patterns):
        return False

    # --- Signal 1: Explicit evidence / literature request ---
    explicit_terms = (
        "文献", "论文", "研究", "依据", "证据", "共识",
        "meta", "trial", "randomized", "nccn", "csco", "asco", "esmo",
        "pubmed", "lancet", "nejm", "jco",
    )
    if any(t in q for t in explicit_terms):
        return True

    # --- Signal 2: Clinical decision questions needing evidence support ---
    # These patterns indicate treatment/management decisions where literature matters.
    clinical_decision_terms = (
        "方案", "治疗", "用药", "换药", "化疗", "靶向", "免疫",
        "二线", "三线", "一线", "后线",
        "复发", "转移", "耐药", "进展",
        "分期", "预后", "生存率", "生存期",
        "手术", "切除", "新辅助", "辅助",
        "推荐", "选择", "对比", "优劣",
        "基因", "突变", "biomarker", "her2", "kras", "braf", "msi",
    )
    decision_score = sum(1 for t in clinical_decision_terms if t in q)
    if decision_score >= 2:
        return True

    # --- Signal 3: Complex patient with any clinical management question ---
    if len(timeline_events) >= 3 and decision_score >= 1:
        return True

    # --- Signal 4: Guideline-related but asked as a clinical question ---
    guideline_context_terms = ("指南", "规范", "标准", "guideline")
    if any(t in q for t in guideline_context_terms) and decision_score >= 1:
        return True

    # --- Signal 5: Interpretation/Analysis intent ---
    interpretation_terms = ("解读", "分析", "看看", "评估", "含义")
    if any(t in q for t in interpretation_terms) and (len(timeline_events) >= 1 or decision_score >= 1):
        return True

    return False



def _assess_reasoning_mode(
    query: str,
    timeline_events: list[ClinicalEvent],
    history: list[dict[str, str]],
) -> tuple[str, dict[str, Any]]:
    q = (query or "").strip().lower()
    score = 0
    reasons: list[str] = []

    # 1. Urgent triage is always quick (direct response).
    urgent_terms = ("胸痛", "呼吸困难", "昏迷", "抽搐", "大出血", "高热不退", "急诊", "120", "救命")
    if any(t in q for t in urgent_terms):
        return "quick", {"score": 10, "reasons": ["urgent_triage"]}

    # 2. Explicit interpretation intent -> Deep
    interpretation_terms = ("解读", "分析", "看看", "评估", "含义", "意思", "说明", "报告", "结果")
    if any(t in q for t in interpretation_terms):
        score += 2
        reasons.append("interpretation_intent")

    # 3. Context complexity
    if len(timeline_events) >= 1:
        score += 1
        reasons.append("has_context")
    if len(query) >= 10:
        score += 1
        reasons.append("query_detailed")
    
    # 4. Medical complexity keywords
    complex_terms = (
        "分期", "方案", "二线", "换药", "耐药", "复发", "转移", "并发症", "鉴别",
        "证据", "依据", "文献", "指南", "对比", "风险", "副作用", "生存", "预后",
        "基因", "靶向", "免疫", "化疗", "放疗", "手术",
    )
    if any(t in q for t in complex_terms):
        score += 2
        reasons.append("medical_complexity")

    # 5. Trivial check: very short, non-medical queries are quick.
    # Only applies if NO other positive score was found.
    if score == 0:
        # e.g. "你好", "在吗", "谢谢"
        # We add interpretation terms to the exemption list just in case score logic missed it.
        medical_chars = ("痛", "药", "病", "查", "瘤", "癌", "医", "诊", "疗")
        if len(q) < 5 and not any(k in q for k in medical_chars):
            return "quick", {"score": 0, "reasons": ["trivial_short"]}

    # Default to deep if any medical context or intent is found (score >= 1).
    mode = "deep" if score >= 1 else "quick"
    return mode, {"score": score, "reasons": reasons}


def _apply_citation_projection(answer: str, ranked_sources: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], str]:
    sources = list(ranked_sources)
    cited_ranks = _extract_cited_ranks(answer)
    available_ranks = {int(s.get("rank", -1)) for s in ranked_sources}
    citation_warning = ""
    if cited_ranks:
        valid_cited = {r for r in cited_ranks if r in available_ranks}
        invalid_cited = sorted([r for r in cited_ranks if r not in available_ranks])
        if valid_cited:
            sources = [s for s in ranked_sources if int(s.get("rank", -1)) in valid_cited]
            if invalid_cited:
                citation_warning = f"存在越界引用: {invalid_cited}，已展示可用证据。"
        else:
            citation_warning = f"引用编号越界: {invalid_cited}。已回退展示可用证据。"
            sources = ranked_sources[: min(3, len(ranked_sources))]
    else:
        cited_pages = _extract_cited_pages(answer)
        if cited_pages:
            page_tags = [f"[第{p}页]" for p in cited_pages]
            filtered = []
            for s in ranked_sources:
                evidence = str(s.get("evidence", ""))
                if any(tag in evidence for tag in page_tags):
                    filtered.append(s)
            sources = filtered if filtered else ranked_sources[:2]
        else:
            sources = []
    return sources, citation_warning


def _extract_key_conclusions(evidence_guard: dict[str, Any], max_items: int = 4) -> list[dict[str, Any]]:
    checks = evidence_guard.get("checks", [])
    key_conclusions: list[dict[str, Any]] = []
    if not isinstance(checks, list):
        return key_conclusions
    sorted_checks = sorted(
        checks,
        key=lambda c: (
            0 if bool(c.get("supported")) else 1,
            -(float(c.get("support_score", 0.0) or 0.0)),
        ),
    )
    for item in sorted_checks:
        cited = item.get("cited_ranks", [])
        if not isinstance(cited, list) or not cited:
            continue
        key_conclusions.append(
            {
                "claim": str(item.get("claim", ""))[:200],
                "supported": bool(item.get("supported")),
                "sufficiency": str(item.get("sufficiency", "low")),
                "support_score": float(item.get("support_score", 0.0) or 0.0),
                "cited_ranks": cited[:3],
                "reason": str(item.get("reason", "")),
            }
        )
        if len(key_conclusions) >= max(max_items, 1):
            break
    return key_conclusions


class ChatRequest(BaseModel):
    message: str
    history: list[dict[str, str]] = []


class ImportGuidelinesRequest(BaseModel):
    reset: bool = False  # False = incremental (only changed files), True = full rebuild


class LiteratureSearchQuery(BaseModel):
    q: str
    top_k: int = 5


class RegisterRequest(BaseModel):
    username: str
    password: str
    display_name: str = ""


class LoginRequest(BaseModel):
    username: str
    password: str


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/favicon.ico")
def favicon() -> Response:
    # Browser auto-requests this; return 204 to avoid noisy 404 logs.
    return Response(status_code=204)


@app.get("/api/health")
def health() -> dict[str, Any]:
    _cleanup_expired_sessions()
    return {
        "status": "ok",
        "literature_topic_query": LITERATURE_AGENT_TOPIC_QUERY,
        "vector_backend": get_vector_backend(),
        "upload_deid": ENABLE_UPLOAD_DEID,
        "ocr_provider": OCR_PROVIDER,
        "paddle_available": is_paddle_available(),
        "pdf_ocr_max_pages": PDF_OCR_MAX_PAGES,
        "internal_chunks": len(internal_store.chunks),
        "active_sessions": len(session_stores),
        "active_timelines": len(session_timeline_events),
        "session_ttl_seconds": SESSION_TTL_SECONDS,
        "manual_session_clear_only": MANUAL_SESSION_CLEAR_ONLY,
        "audit_log_path": str(AUDIT_LOG_PATH.relative_to(ROOT)),
        "models": {"hidden": True},
        "timeline_encoder": TIMELINE_ENCODER,
        "openviking_enabled": OPENVIKING_ENABLED,
        "openviking_use_llm": OPENVIKING_USE_LLM,
        "openviking_l0_topk": OPENVIKING_L0_TOPK,
        "openviking_l1_topk": OPENVIKING_L1_TOPK,
        "openviking_l2_topk": OPENVIKING_L2_TOPK,
        "local_literature_agent_enabled": ENABLE_LOCAL_LITERATURE_AGENT,
        "local_literature_records": len(literature_index),
        "local_literature_last_refresh_at": int(literature_last_refresh_at) if literature_last_refresh_at else None,
        "local_literature_last_refresh_result": literature_last_refresh_result,
        "web_literature_enabled": ENABLE_WEB_LITERATURE,
        "literature_providers": LITERATURE_PROVIDERS,
        "literature_medical_oncology_only": LITERATURE_MEDICAL_ONCOLOGY_ONLY,
    }


# -- Auth API --

@app.post("/api/auth/register")
def auth_register(payload: RegisterRequest) -> JSONResponse:
    ok, result = user_manager.register(
        payload.username, payload.password, payload.display_name
    )
    if not ok:
        return JSONResponse({"ok": False, "error": result}, status_code=400)
    user = user_manager.get_user_by_token(result)
    return JSONResponse({
        "ok": True,
        "token": result,
        "user": user_manager.get_user_info(user) if user else {},
    })


@app.post("/api/auth/login")
def auth_login(payload: LoginRequest) -> JSONResponse:
    ok, result = user_manager.login(payload.username, payload.password)
    if not ok:
        return JSONResponse({"ok": False, "error": result}, status_code=401)
    user = user_manager.get_user_by_token(result)
    return JSONResponse({
        "ok": True,
        "token": result,
        "user": user_manager.get_user_info(user) if user else {},
    })


@app.post("/api/auth/logout")
def auth_logout(request: Request) -> JSONResponse:
    # JWT is stateless – the client simply discards the token.
    # For token revocation, a server-side blocklist can be added later.
    return JSONResponse({"ok": True})


@app.get("/api/auth/me")
def auth_me(request: Request) -> JSONResponse:
    user, user_id = _get_auth_user(request)
    if not user:
        return JSONResponse({"ok": False, "error": "not_authenticated"}, status_code=401)
    user_dir = user_manager.get_user_data_dir(user_id)
    timeline_path = user_dir / "timeline_events.json"
    vector_path = user_dir / "user_vector_store.json"
    openviking_path = user_dir / "openviking_layers.json"
    timeline_count = 0
    chunk_count = 0
    openviking_l0 = 0
    openviking_l1 = 0
    upload_sources: list[str] = []
    if timeline_path.exists():
        try:
            events = json.loads(timeline_path.read_text(encoding="utf-8"))
            timeline_count = len(events)
        except Exception:
            pass
    if vector_path.exists():
        try:
            chunks = json.loads(vector_path.read_text(encoding="utf-8"))
            chunk_count = len(chunks)
            sources_set = set()
            for c in chunks:
                src = c.get("source", "")
                if src.startswith("session/"):
                    sources_set.add(src.removeprefix("session/"))
            upload_sources = sorted(sources_set)
        except Exception:
            pass
    if openviking_path.exists():
        try:
            layered = json.loads(openviking_path.read_text(encoding="utf-8"))
            for item in layered:
                layer = str(item.get("layer", "")).upper()
                if layer == "L0":
                    openviking_l0 += 1
                elif layer == "L1":
                    openviking_l1 += 1
        except Exception:
            pass
    if openviking_l0 == 0 and openviking_l1 == 0 and OPENVIKING_ENABLED:
        try:
            stats = _get_user_openviking_store(user_id).stats()
            openviking_l0 = int(stats.get("l0", 0) or 0)
            openviking_l1 = int(stats.get("l1", 0) or 0)
        except Exception:
            pass
    return JSONResponse({
        "ok": True,
        "user": user_manager.get_user_info(user),
        "data_stats": {
            "timeline_events": timeline_count,
            "vector_chunks": chunk_count,
            "openviking_l0": openviking_l0,
            "openviking_l1": openviking_l1,
            "upload_sources": upload_sources,
        },
    })


@app.post("/api/user/delete-data")
def user_delete_data(request: Request) -> JSONResponse:
    user, user_id = _get_auth_user(request)
    if not user:
        return JSONResponse({"ok": False, "error": "not_authenticated"}, status_code=401)
    result = user_manager.delete_user_data(user_id)
    user_stores.pop(user_id, None)
    user_timeline_events.pop(user_id, None)
    user_openviking_stores.pop(user_id, None)
    audit_logger.log(
        event_type="user_data_deleted",
        session_id=user_id,
        details=result,
    )
    return JSONResponse({"ok": True, **result})



@app.post("/api/user/delete-account")
def user_delete_account(request: Request) -> JSONResponse:
    user, user_id = _get_auth_user(request)
    if not user:
        return JSONResponse({"ok": False, "error": "not_authenticated"}, status_code=401)
    
    # Delete all data first
    user_manager.delete_user_data(user_id)
    user_stores.pop(user_id, None)
    user_timeline_events.pop(user_id, None)
    user_openviking_stores.pop(user_id, None)
    
    # Delete the account record
    ok = user_manager.delete_account(user.username)
    
    audit_logger.log(
        event_type="user_account_deleted",
        session_id=user_id,
        details={"deleted": ok, "username": user.username},
    )
    return JSONResponse({"ok": ok})

@app.post("/api/user/delete-upload")
def user_delete_upload(request: Request, source: str = "") -> JSONResponse:
    user, user_id = _get_auth_user(request)
    if not user:
        return JSONResponse({"ok": False, "error": "not_authenticated"}, status_code=401)
    if not source:
        return JSONResponse({"ok": False, "error": "source required"}, status_code=400)
    full_source = f"session/{source}" if not source.startswith("session/") else source
    store = _get_or_create_user_store(user_id)
    original_count = len(store.chunks)
    store.chunks = [c for c in store.chunks if c.source != full_source]
    removed_chunks = original_count - len(store.chunks)
    if hasattr(store, '_persist'):
        store._persist()
    events = _load_user_timeline(user_id)
    original_events = len(events)
    events = [ev for ev in events if ev.source != full_source]
    user_timeline_events[user_id] = events
    _save_user_timeline(user_id)
    removed_events = original_events - len(events)
    layered_store = _get_or_create_user_openviking_store(user_id)
    removed_layers = layered_store.remove_source(full_source)
    return JSONResponse({
        "ok": True,
        "removed_chunks": removed_chunks,
        "removed_events": removed_events,
        "removed_openviking_layers": removed_layers,
        "remaining_chunks": len(store.chunks),
        "remaining_events": len(events),
        "remaining_openviking_layers": layered_store.stats().get("total", 0),
    })


@app.on_event("startup")
def bootstrap_literature_agent() -> None:
    if ENABLE_LOCAL_LITERATURE_AGENT and not literature_index:
        _refresh_local_literature_if_needed(force=True, max_results=LITERATURE_AGENT_BOOTSTRAP_MAX_RESULTS)


@app.post("/api/upload")
async def upload(request: Request, files: list[UploadFile] = File(...)) -> JSONResponse:
    auth_user, auth_user_id = _get_auth_user(request)
    session_id = _get_session_id(request)

    # Use persistent user store if authenticated, otherwise in-memory session store
    if auth_user_id:
        active_store = _get_or_create_user_store(auth_user_id)
    else:
        active_store = _get_or_create_session_store(session_id)
    active_openviking_store: OpenVikingStore | None = None
    if OPENVIKING_ENABLED:
        if auth_user_id:
            active_openviking_store = _get_or_create_user_openviking_store(auth_user_id)
        else:
            active_openviking_store = _get_or_create_session_openviking_store(session_id)

    results = []
    has_error = False
    for f in files:
        file_name = f.filename or "unnamed"
        try:
            is_image, input_mime = _is_image_input(file_name, f.content_type)
            file_bytes = await f.read()
            text, extract_meta = _extract_file_text_from_bytes(file_name, f.content_type, file_bytes)
            redaction = {}
            redaction_preview: list[str] = []
            image_redaction: dict[str, Any] = {
                "available": False,
                "engine": "none",
                "reason": "not_image",
                "ocr_word_count": 0,
                "sensitive_box_count": 0,
                "boxes": [],
            }
            if ENABLE_UPLOAD_DEID:
                text, redaction_stats = redact_sensitive_info(text)
                redaction = redaction_stats.as_dict()
                redaction_preview = extract_redaction_preview(text)
            if is_image and ENABLE_UPLOAD_DEID:
                image_redaction = _build_image_redaction_result(file_bytes, mime_type=input_mime)
            chunks = split_text(text)
            source_key = f"session/{file_name}"
            added = active_store.add_texts(
                source=source_key,
                texts=chunks,
                embed_fn=lambda t: embed_text(client, EMBEDDING_MODEL, t),
            )
            extracted_events, has_known_date = _add_timeline_events(
                session_id, source=source_key, text=text,
                user_id=auth_user_id,
            )
            layered_stats = {"l0": 0, "l1": 0}
            if active_openviking_store is not None:
                source_chunks = [c for c in active_store.chunks if c.source == source_key]
                layered_stats = _upsert_openviking_layers(active_openviking_store, source_key, source_chunks)
            date_warning = ""
            timeline_date = str((extract_meta or {}).get("timeline_date", "")).strip()
            timeline_date_source = str((extract_meta or {}).get("timeline_date_source", "")).strip()
            if not has_known_date:
                date_warning = "未识别到日期（该文件时间线可能显示为“未知日期”）"
            audit_logger.log(
                event_type="upload_processed",
                session_id=auth_user_id or session_id,
                details={
                    "file": file_name,
                    "chunks": added,
                    "timeline_events": extracted_events,
                    "openviking_l0": layered_stats.get("l0", 0),
                    "openviking_l1": layered_stats.get("l1", 0),
                    "date_detected": has_known_date,
                    "date_warning": date_warning,
                    "timeline_date": timeline_date,
                    "timeline_date_source": timeline_date_source,
                    "date_fields": {
                        "exam_date": (extract_meta or {}).get("exam_date", "NA"),
                        "submit_date": (extract_meta or {}).get("submit_date", "NA"),
                        "report_date": (extract_meta or {}).get("report_date", "NA"),
                    },
                    "redaction": redaction,
                    "redaction_preview": redaction_preview,
                    "image_redaction": image_redaction,
                    "authenticated": auth_user_id is not None,
                },
            )
            results.append(
                {
                    "file": file_name,
                    "chunks": added,
                    "ok": True,
                    "redaction": redaction,
                    "redaction_preview": redaction_preview,
                    "image_redaction": image_redaction,
                    "timeline_events": extracted_events,
                    "openviking_l0": layered_stats.get("l0", 0),
                    "openviking_l1": layered_stats.get("l1", 0),
                    "date_detected": has_known_date,
                    "date_warning": date_warning,
                    "timeline_date": timeline_date,
                    "timeline_date_source": timeline_date_source,
                    "date_fields": {
                        "exam_date": (extract_meta or {}).get("exam_date", "NA"),
                        "submit_date": (extract_meta or {}).get("submit_date", "NA"),
                        "report_date": (extract_meta or {}).get("report_date", "NA"),
                    },
                }
            )
        except HTTPException as exc:
            has_error = True
            results.append({"file": file_name, "ok": False, "error": exc.detail})
        except Exception as exc:
            has_error = True
            results.append({"file": file_name, "ok": False, "error": str(exc)})
    status_code = 207 if has_error else 200
    timeline_events = _get_timeline_events(session_id, user_id=auth_user_id)
    openviking_total = active_openviking_store.stats().get("total", 0) if active_openviking_store else 0
    return JSONResponse(
        {
            "ok": not has_error,
            "results": results,
            "session_chunks": len(active_store.chunks),
            "timeline_event_count": len(timeline_events),
            "openviking_items": openviking_total,
            "persistent": auth_user_id is not None,
        },
        status_code=status_code,
    )


def _file_md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(8192), b""):
            h.update(block)
    return h.hexdigest()


def _load_guideline_versions() -> dict[str, Any]:
    if GUIDELINES_VERSION_PATH.exists():
        try:
            return json.loads(GUIDELINES_VERSION_PATH.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {"files": {}, "history": []}


def _save_guideline_versions(manifest: dict[str, Any]) -> None:
    GUIDELINES_VERSION_PATH.parent.mkdir(parents=True, exist_ok=True)
    GUIDELINES_VERSION_PATH.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )


@app.post("/api/internal/import-guidelines")
def import_guidelines(payload: ImportGuidelinesRequest, request: Request) -> JSONResponse:
    _ensure_internal_access(request)
    if not GUIDELINES_DIR.exists():
        raise HTTPException(status_code=400, detail="data/guidelines 目录不存在")

    files = sorted([p for p in GUIDELINES_DIR.rglob("*") if p.is_file()])
    if not files:
        raise HTTPException(status_code=400, detail="data/guidelines 目录下没有文件")

    manifest = _load_guideline_versions()
    prev_hashes: dict[str, str] = manifest.get("files", {})

    if payload.reset:
        internal_store.clear()
        prev_hashes = {}  # treat everything as new

    # Compute current hashes and detect changes.
    current_hashes: dict[str, str] = {}
    changed_files: list[Path] = []
    for file_path in files:
        relative = str(file_path.relative_to(GUIDELINES_DIR)).replace("\\", "/")
        file_hash = _file_md5(file_path)
        current_hashes[relative] = file_hash
        if prev_hashes.get(relative) != file_hash:
            changed_files.append(file_path)

    # Detect deleted files and remove their chunks.
    deleted_files = set(prev_hashes.keys()) - set(current_hashes.keys())
    for deleted_rel in deleted_files:
        del_source = f"internal/{deleted_rel}"
        internal_store.chunks = [c for c in internal_store.chunks if c.source != del_source]
    if deleted_files and hasattr(internal_store, '_matrix'):
        internal_store._matrix = None  # invalidate cache
    if deleted_files and hasattr(internal_store, '_persist'):
        internal_store._persist()

    results = []
    has_error = False
    skipped = 0
    for file_path in files:
        relative = str(file_path.relative_to(GUIDELINES_DIR)).replace("\\", "/")
        source = f"internal/{relative}"
        if file_path not in changed_files:
            skipped += 1
            continue
        # Remove old chunks for this file before re-adding.
        internal_store.chunks = [c for c in internal_store.chunks if c.source != source]
        if hasattr(internal_store, '_matrix'):
            internal_store._matrix = None
        try:
            text = _extract_file_text(file_path, None, ocr_mode="guide")
            chunks = split_text(text)
            added = internal_store.add_texts(
                source=source,
                texts=chunks,
                embed_fn=lambda t: embed_text(client, EMBEDDING_MODEL, t),
            )
            results.append({"file": relative, "chunks": added, "ok": True, "action": "updated"})
        except HTTPException as exc:
            has_error = True
            results.append({"file": relative, "ok": False, "error": exc.detail})
        except Exception as exc:
            has_error = True
            results.append({"file": relative, "ok": False, "error": str(exc)})

    # Update version manifest.
    manifest["files"] = current_hashes
    history = manifest.get("history", [])
    history.append({
        "timestamp": int(time.time()),
        "reset": payload.reset,
        "changed": len(changed_files),
        "deleted": len(deleted_files),
        "skipped": skipped,
        "total_files": len(files),
        "total_chunks": len(internal_store.chunks),
    })
    manifest["history"] = history[-50:]  # keep last 50 entries
    _save_guideline_versions(manifest)

    status_code = 207 if has_error else 200
    return JSONResponse(
        {
            "ok": not has_error,
            "results": results,
            "internal_chunks": len(internal_store.chunks),
            "reset": payload.reset,
            "skipped_unchanged": skipped,
            "deleted_files": sorted(deleted_files),
            "version_entries": len(manifest.get("history", [])),
        },
        status_code=status_code,
    )


@app.get("/api/internal/guidelines")
def list_guidelines(request: Request) -> JSONResponse:
    _ensure_internal_access(request)
    if not GUIDELINES_DIR.exists():
        return JSONResponse({"files": []})
    files = []
    for f in GUIDELINES_DIR.iterdir():
        if f.is_file() and not f.name.startswith("."):
            stats = f.stat()
            files.append({
                "name": f.name,
                "size": stats.st_size,
                "modified": stats.st_mtime
            })
    # Sort: updated recently first
    files.sort(key=lambda x: x["modified"], reverse=True)
    return JSONResponse({"files": files})


@app.post("/api/internal/guidelines")
async def upload_guideline(request: Request, file: UploadFile = File(...)) -> JSONResponse:
    _ensure_internal_access(request)
    if not file or not file.filename:
        raise HTTPException(status_code=400, detail="No file uploaded")
    
    filename = Path(file.filename).name
    # Simple sanitization
    filename = filename.replace("/", "_").replace("\\", "_")
    target_path = GUIDELINES_DIR / filename
    
    GUIDELINES_DIR.mkdir(parents=True, exist_ok=True)
    
    with target_path.open("wb") as buffer:
        import shutil
        shutil.copyfileobj(file.file, buffer)
        
    return JSONResponse({"ok": True, "name": filename, "msg": "上传成功"})


@app.delete("/api/internal/guidelines")
def delete_guideline(request: Request, filename: str) -> JSONResponse:
    _ensure_internal_access(request)
    # Security check to prevent directory traversal
    safe_name = Path(filename).name
    target_path = GUIDELINES_DIR / safe_name
    
    if target_path.exists() and target_path.is_file():
        target_path.unlink()
        return JSONResponse({"ok": True, "msg": "已删除"})
    return JSONResponse({"ok": False, "msg": "文件不存在"}, status_code=404)


@app.post("/api/chat")
def chat(payload: ChatRequest, request: Request) -> JSONResponse:
    if not payload.message.strip():
        raise HTTPException(status_code=400, detail="\u6d88\u606f\u4e0d\u80fd\u4e3a\u7a7a")

    auth_user, auth_user_id = _get_auth_user(request)
    session_id = _get_session_id(request)
    query_embedding = _global_embed(payload.message)
    cached_embed = lambda _text: query_embedding

    def _embed_with_cache(text: str) -> list[float]:
        return _global_embed(text)

    # Retrieve from internal guidelines + user context (OpenViking preferred).
    _INTERNAL_BOOST = 0.03  # slight boost so guidelines win ties
    internal_scored = internal_store.similarity_search_with_scores(
        payload.message,
        embed_fn=cached_embed,
        k=4,
    )
    internal_context_scored: list[tuple[float, RetrievedContext]] = [
        (
            float(score),
            RetrievedContext(
                id=str(chunk.id),
                source=str(chunk.source),
                text=str(chunk.text),
                layer="L2",
            ),
        )
        for score, chunk in internal_scored
    ]
    session_scored: list[tuple[float, RetrievedContext]] = []
    openviking_trace: dict[str, Any] = {"enabled": False}
    # Use persistent user store if authenticated
    if auth_user_id:
        user_store = _get_user_store(auth_user_id)
        openviking_store = _get_user_openviking_store(auth_user_id) if OPENVIKING_ENABLED else None
    else:
        user_store = _get_session_store(session_id)
        openviking_store = _get_session_openviking_store(session_id) if OPENVIKING_ENABLED else None
    timeline_events = _get_timeline_events(session_id, user_id=auth_user_id)
    reasoning_mode, reasoning_meta = _assess_reasoning_mode(
        payload.message,
        timeline_events,
        payload.history,
    )
    timeline_state = encode_timeline_state(timeline_events, encoder=TIMELINE_ENCODER)
    retrieval_hint = build_retrieval_hint(payload.message, timeline_events, state=timeline_state)
    timeline_summary = build_timeline_summary(timeline_events, max_items=8)

    if user_store is not None and OPENVIKING_ENABLED and openviking_store is not None:
        layered_total = int(openviking_store.stats().get("total", 0) or 0)
        if layered_total == 0 and len(getattr(user_store, "chunks", [])) > 0:
            auto_rebuild = _rebuild_openviking_from_vector_store(
                openviking_store,
                user_store,
                reset=False,
            )
            openviking_trace["auto_rebuild"] = {
                "sources": int(auto_rebuild.get("sources", 0) or 0),
                "stats": auto_rebuild.get("stats", {}),
            }

    # Priority path: L0 -> L1 -> optional L2 drill-down.
    if user_store is not None and OPENVIKING_ENABLED and openviking_store is not None:
        session_scored, layered_trace = _collect_openviking_context(
            query=payload.message,
            query_embedding=query_embedding,
            user_store=user_store,
            openviking_store=openviking_store,
        )
        openviking_trace.update(layered_trace)

    # Fallback: flat L2 retrieval when layered memory isn't available yet.
    if user_store is not None and not session_scored:
        flat_scored = user_store.similarity_search_with_scores(
            payload.message,
            embed_fn=cached_embed,
            k=4,
        )
        session_scored = [
            (
                float(score),
                RetrievedContext(
                    id=str(chunk.id),
                    source=str(chunk.source),
                    text=str(chunk.text),
                    layer="L2",
                ),
            )
            for score, chunk in flat_scored
        ]
        openviking_trace.setdefault("fallback_flat_l2", True)

    # Merge by similarity score (internal chunks get a small boost at ties).
    _merged = [(score + _INTERNAL_BOOST, chunk) for score, chunk in internal_context_scored] + list(session_scored)
    _merged.sort(key=lambda item: item[0], reverse=True)
    all_scored = _merged[:6]
    context_chunks = [item[1] for item in all_scored]
    context = "\n\n".join([f"[来源:{_format_source(c.source)}]\n{c.text}" for c in context_chunks])
    if not context:
        context = "暂无可用资料。"
    external_literature = []
    use_external_literature = _should_use_external_literature(
        payload.message,
        timeline_events,
    )
    if use_external_literature:
        _refresh_local_literature_if_needed(force=False, max_results=LITERATURE_AGENT_BOOTSTRAP_MAX_RESULTS)
        local_literature: list[dict[str, object]] = []
        web_literature: list[dict[str, object]] = []
        if ENABLE_LOCAL_LITERATURE_AGENT:
            local_literature = search_from_local_store(
                payload.message,
                embed_fn=_embed_with_cache,
                top_k=LITERATURE_TOP_K,
                index=literature_index,
            )
        if ENABLE_WEB_LITERATURE:
            web_literature = [
                item.as_dict()
                for item in search_literature(
                    payload.message,
                    providers=LITERATURE_PROVIDERS,
                    top_k=LITERATURE_TOP_K,
                    timeout=LITERATURE_TIMEOUT_SECONDS,
                    embed_fn=_embed_with_cache,
                    medical_oncology_only=LITERATURE_MEDICAL_ONCOLOGY_ONLY,
                    min_relevance=LITERATURE_MIN_RELEVANCE,
                )
            ]
        # Merge and deduplicate by doc_id or DOI.
        seen_ids: set[str] = set()
        merged_literature: list[dict[str, object]] = []
        for item in local_literature + web_literature:
            dedup_key = str(item.get("doc_id") or item.get("doi") or item.get("title", "")).strip().lower()
            if dedup_key and dedup_key in seen_ids:
                continue
            if dedup_key:
                seen_ids.add(dedup_key)
            merged_literature.append(item)
        # Sort by relevance score descending.
        merged_literature.sort(key=lambda x: float(x.get("relevance", 0) or 0), reverse=True)
        external_literature = merged_literature[:LITERATURE_TOP_K]
    literature_context = literature_to_context(external_literature, max_items=LITERATURE_TOP_K)

    messages = [
        {
            "role": "system",
            "content": (
                "你是一位经验丰富、温和且严谨的肿瘤科主治医师。你的对话对象是焦虑的患者或家属。\n"
                "你的核心任务：像在门诊面对面交流一样，用“我”的视角解读病历，给出有温度、有依据的专业建议。\n\n"
                "【沟通风格：医生视角 + 自然对话】\n"
                "- **第一人称（强制）**：必须时刻使用“我看了您的...”、“我认为...”、“我建议...”。严禁使用“该患者”、“根据结果显示”这种第三方冷漠语体。\n"
                "- **拒绝机器味**：严禁上来就罗列“一、xxx；二、xxx”。请像真人聊天一样，用自然的连接词（“首先...”、“从目前的指标看...”、“至于您担心的...”）来组织语言。\n"
                "- **共情与安抚**：在抛出异常指标或坏消息前，必须先有一句缓冲（如“这个指标确实偏高，但先别急，我们需要结合影像看...”）。\n\n"
                "【版面美学：清晰大方】\n"
                "虽然语气是口语化的，但排版必须清晰易读。请巧妙利用 Markdown：\n"
                "- **小标题引导**：用 `### 关于报告解读`、`### 下一步建议` 等小标题区隔话题，不要挤成一团。\n"
                "- **重点加粗**：关键的药物名、检查项、异常值请用 **粗体** 高亮，方便患者一眼抓重点。\n"
                "- **适度列表**：列举多个检查项目或注意事项时，请使用无序列表（- ），保持整洁。\n\n"
                "【临床逻辑：严谨闭环】\n"
                "- **先核实，后建议**：若病历信息孤立（如只有一张化验单），必须先询问前情（“这是术后复查还是初诊？之前做过什么治疗？”），不要盲目建议做已完成的检查。\n"
                "- **条件式建议**：尽量说“若您近期未查...建议...”，而不是生硬的“请去做...”。\n\n"
                "【范例参考（请模仿这种语感）】\n"
                "❌ 错误：\n"
                "一、诊断结果：CA19-9升高。\n"
                "二、风险评估：高风险。\n"
                "三、建议：进行PET-CT检查。\n\n"
                "✅ 正确：\n"
                "### 📋 关于您的血液报告\n"
                "我仔细看了您上传的化验单，**CA19-9** 这一项指标确实引起了我的注意，它比正常值高出了一些。这通常提示我们需要警惕消化道来源的问题，但也可能是炎症引起的波动。\n\n"
                "### 💡 我的建议\n"
                "考虑到您之前已经做过切除手术，我倾向于认为这需要进一步排查。**如果您近期没有做过全腹增强CT**，我强烈建议您安排一次，以排除复发的可能..."
                "【临床推理与证据约束（当接入资料时启用）】\n"
                "- 先形成可检验结论（claims），再检查每条是否有证据支撑；无法支撑的改为不确定表述或删除。\n"
                "- 若用户要求依据/来源或你引用外部资料，仅提供 1-3 条关键依据，避免论文化表达。\n\n"
                "- 简单问题优先直接给结论与行动建议，不要强行扩展成长文。\n"
                "- 外部学术文献仅在必要时使用；若本轮未使用外部文献，不要输出[文献#]。\n\n"
                "【身份披露与术语约束（严格执行）】\n"
                "- 禁止使用“Mamba”、“Transformer”、“Score”、“算法得分”、“模型预测值”等底层技术术语。\n"
                "- 禁止输出小数点后超过2位的数值评分（如 4.0534），改用“高风险”、“中等风险”等定性描述。\n"
                "- 若系统提示 risk_level=high，必须在回答结尾增加一句基于循证医学的安抚或积极引导（如“规范治疗可有效控制...”）。\n\n"
                "【输出要求】\n"
                "- 语气：专业、稳重、同理但不夸张。像一位经验丰富的临床医生在与患者沟通。\n"
                "- 长度：优先短而有用；复杂问题才展开。\n"
                "- 若涉及可能危及生命的情况，必须明确写出“建议立刻急诊/拨打当地急救电话”的触发条件。\n\n"
                "【身份披露约束】\n"
                "- 禁止透露底层模型名称、供应商、版本号、API信息或系统配置。\n"
                "- 若用户询问“你是什么模型”，仅回答“我是医疗助手”。\n\n"
                "【引用规则（系统强制）】\n"
                "- 当使用参考资料时，请在对应句末添加[证据#序号]，序号对应参考资料出现顺序（从1开始）。\n"
                "- 如使用外部文献，请添加[文献#序号]。\n"
                "- 引用外部文献时，明确文献类型（临床试验/系统综述/指南/观察性研究）。"
            ),
        },
        {
            "role": "system",
            "content": (
                f"患者时序状态({TIMELINE_ENCODER} 编码): "
                f"risk_level={timeline_state.get('risk_level')} "
                f"risk_score={timeline_state.get('risk_score')} "
                f"event_count={int(float(timeline_state.get('event_count', 0.0)))}\n"
                f"检索路由建议: {retrieval_hint}\n"
                f"病程摘要:\n{timeline_summary}"
            ),
        },
        {
            "role": "system",
            "content": (
                f"推理模式: {reasoning_mode}。"
                "quick模式要求：简洁直答，3-6句，先给结论与下一步行动，不做冗长展开；"
                "deep模式要求：可做结构化分析并给出关键证据。"
            ),
        },
        {"role": "system", "content": f"参考资料:\n{context}"},
        {"role": "system", "content": f"外部学术文献:\n{literature_context}"},
    ]
    for msg in payload.history[-8:]:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        if role in {"system", "user", "assistant"} and content:
            messages.append({"role": role, "content": content})
    messages.append({"role": "user", "content": payload.message})

    resp = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=messages,
        temperature=0.2,
    )
    answer = _sanitize_model_disclosure(resp.choices[0].message.content or "暂无回复")
    ranked_sources = []
    for rank, (score, chunk) in enumerate(all_scored, start=1):
        source_type = "internal" if str(chunk.source).startswith("internal/") else "session"
        layer = str(getattr(chunk, "layer", "L2"))
        evidence = _build_evidence_snippet(str(chunk.text), payload.message)
        item = {
            "rank": rank,
            "score": round(float(score), 4),
            "chunk_id": chunk.id,
            "source": _format_source(chunk.source),
            "source_raw": chunk.source,
            "type": source_type,
            "layer": layer,
            "title": str(getattr(chunk, "title", "") or ""),
            "page": _extract_page_from_text(chunk.text),
            "section": _extract_section_from_text(chunk.text),
            "text_length": len(chunk.text),
            "preview": chunk.text[:220],
            "evidence": evidence,
        }
        ranked_sources.append(item)

    def _evaluate_answer(candidate: str) -> tuple[list[dict[str, Any]], str, dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
        candidate_sources, candidate_warning = _apply_citation_projection(candidate, ranked_sources)
        candidate_evidence_guard = verify_answer_with_evidence(
            candidate,
            ranked_sources,
            embed_similarity_fn=_embed_with_cache,
        ).as_dict()
        candidate_literature_guard = verify_literature_citations(
            candidate,
            external_literature,
            embed_fn=_embed_with_cache if external_literature else None,
        )
        candidate_conclusions = _extract_key_conclusions(candidate_evidence_guard, max_items=4)
        return (
            candidate_sources,
            candidate_warning,
            candidate_evidence_guard,
            candidate_literature_guard,
            candidate_conclusions,
        )

    (
        sources,
        citation_warning,
        evidence_guard,
        literature_guard,
        key_conclusions,
    ) = _evaluate_answer(answer)

    # --- Conflict detection between internal guidelines and external literature ---
    evidence_conflicts: list[dict[str, str]] = []
    if external_literature and ranked_sources:
        internal_evidence = [
            s for s in ranked_sources if s.get("type") == "internal"
        ]
        if internal_evidence and len(external_literature) > 0:
            try:
                for ie in internal_evidence[:3]:
                    ie_text = str(ie.get("evidence", ""))
                    if not ie_text:
                        continue
                    ie_vec = _embed_with_cache(ie_text)
                    for idx, lit in enumerate(external_literature[:3]):
                        lit_text = f"{lit.get('title', '')}. {lit.get('abstract', '')}"
                        if not lit_text.strip():
                            continue
                        lit_vec = _embed_with_cache(lit_text)
                        # Compute cosine similarity.
                        ie_arr = np.array(ie_vec, dtype=np.float32)
                        lit_arr = np.array(lit_vec, dtype=np.float32)
                        denom = np.linalg.norm(ie_arr) * np.linalg.norm(lit_arr)
                        sim = float(np.dot(ie_arr, lit_arr) / max(denom, 1e-8)) if denom > 1e-8 else 0.0
                        # Related content (> 0.3) but not well-aligned (< 0.7) suggests potential conflict.
                        if 0.3 < sim < 0.7:
                            evidence_conflicts.append({
                                "internal_source": str(ie.get("source", "")),
                                "literature_ref": f"[文献#{idx+1}] {lit.get('title', '')}",
                                "similarity": round(sim, 3),
                                "warning": "内部指南与外部文献在此主题上可能存在差异，请结合最新证据综合判断。",
                            })
            except Exception:
                pass  # conflict detection is best-effort

    auto_rewrite_applied = False
    rewrite_triggered = False
    rewrite_rounds = 0
    _MAX_REWRITE_ROUNDS = max(int(os.getenv("MAX_REWRITE_ROUNDS", "2")), 1)
    if AUTO_EVIDENCE_REWRITE and reasoning_mode == "deep":
        current_coverage = float(evidence_guard.get("coverage", 1.0) or 0.0)
        current_unsupported = int(evidence_guard.get("unsupported_claims", 0) or 0)
        rewrite_triggered = (
            current_coverage < EVIDENCE_REWRITE_MIN_COVERAGE
            or current_unsupported > EVIDENCE_REWRITE_MAX_UNSUPPORTED
        )
        while rewrite_triggered and ranked_sources and rewrite_rounds < _MAX_REWRITE_ROUNDS:
            rewrite_rounds += 1
            evidence_lines = []
            for item in ranked_sources:
                page = item.get("page")
                page_str = f"P{page}" if isinstance(page, int) else "P?"
                evidence_lines.append(
                    f"[证据#{item['rank']}] {item['source']} | {page_str} | {item['evidence']}"
                )
            external_lines = []
            for idx, lit in enumerate(external_literature[:LITERATURE_TOP_K], start=1):
                external_lines.append(
                    f"[文献#{idx}] {lit.get('title', '')} | type={lit.get('paper_type', 'other')} | "
                    f"year={lit.get('year', 'n/a')} | venue={lit.get('venue', 'n/a')}"
                )
            rewrite_messages = [
                {
                    "role": "system",
                    "content": (
                        "你是医学答案证据约束改写器。请把给定草稿改写为“仅保留可被证据支持”的版本。"
                        "规则："
                        "1) 每条医学结论必须在句末标注[证据#n]；"
                        "2) 不得使用不存在的证据编号；"
                        "3) 无证据支撑的结论要删除，或改写成“需进一步检查确认”；"
                        "4) 保持简洁，避免夸大；"
                        "5) 只输出最终回答正文，不要输出“改写说明/策略/过程”。"
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"用户问题:\n{payload.message}\n\n"
                        f"原始回答:\n{answer}\n\n"
                        f"可用证据:\n" + "\n".join(evidence_lines) + "\n\n"
                        f"可用外部文献:\n" + ("\n".join(external_lines) if external_lines else "无")
                    ),
                },
            ]
            rewrite_resp = client.chat.completions.create(
                model=CHAT_MODEL,
                messages=rewrite_messages,
                temperature=0.0,
            )
            rewritten_answer = _sanitize_model_disclosure(rewrite_resp.choices[0].message.content or answer)
            (
                rewritten_sources,
                rewritten_warning,
                rewritten_evidence_guard,
                rewritten_literature_guard,
                rewritten_key_conclusions,
            ) = _evaluate_answer(rewritten_answer)

            rewritten_coverage = float(rewritten_evidence_guard.get("coverage", 0.0) or 0.0)
            rewritten_unsupported = int(rewritten_evidence_guard.get("unsupported_claims", 999) or 999)
            if (
                rewritten_coverage > current_coverage
                or (
                    rewritten_coverage >= current_coverage
                    and rewritten_unsupported < current_unsupported
                )
            ):
                answer = rewritten_answer
                sources = rewritten_sources
                citation_warning = rewritten_warning
                evidence_guard = rewritten_evidence_guard
                literature_guard = rewritten_literature_guard
                key_conclusions = rewritten_key_conclusions
                auto_rewrite_applied = True
                current_coverage = rewritten_coverage
                current_unsupported = rewritten_unsupported
            else:
                break  # no improvement, stop iterating
            # Check if we've met the target.
            if (
                current_coverage >= EVIDENCE_REWRITE_MIN_COVERAGE
                and current_unsupported <= EVIDENCE_REWRITE_MAX_UNSUPPORTED
            ):
                break
    audit_logger.log(
        event_type="chat_completed",
        session_id=session_id,
        details={
            "query_length": len(payload.message),
            "retrieved_internal": len(internal_scored),
            "retrieved_session": len(session_scored),
            "returned_sources": len(sources),
            "external_literature": len(external_literature),
            "external_literature_enabled": use_external_literature,
            "reasoning_mode": reasoning_mode,
            "reasoning_meta": reasoning_meta,
            "timeline_events": len(timeline_events),
            "openviking_trace": openviking_trace,
            "evidence_coverage": evidence_guard.get("coverage", 0),
            "literature_coverage": literature_guard.get("coverage", 0),
            "rewrite_triggered": rewrite_triggered,
            "auto_rewrite_applied": auto_rewrite_applied,
        },
    )
    return JSONResponse(
        {
            "answer": answer,
            "sources": sources,
            "timeline_state": timeline_state,
            "timeline_summary": timeline_summary,
            "retrieval_hint": retrieval_hint,
            "evidence_guard": evidence_guard,
            "citation_warning": citation_warning,
            "key_conclusions": key_conclusions,
            "external_sources": external_literature,
            "external_literature_enabled": use_external_literature,
            "reasoning_mode": reasoning_mode,
            "reasoning_meta": reasoning_meta,
            "openviking_trace": openviking_trace,
            "literature_guard": literature_guard,
            "auto_rewrite_applied": auto_rewrite_applied,
            "rewrite_triggered": rewrite_triggered,
            "rewrite_rounds": rewrite_rounds,
            "evidence_conflicts": evidence_conflicts,
        }
    )


@app.get("/api/session/timeline")
def session_timeline(request: Request) -> JSONResponse:
    auth_user, auth_user_id = _get_auth_user(request)
    session_id = _get_session_id(request)
    events = _get_timeline_events(session_id, user_id=auth_user_id)
    layered_store = (
        _get_user_openviking_store(auth_user_id)
        if auth_user_id
        else _get_session_openviking_store(session_id)
    )
    layered_stats = layered_store.stats() if layered_store else {"total": 0, "l0": 0, "l1": 0, "sources": 0}
    expiry = _get_session_expiry(session_id, user_id=auth_user_id)
    return JSONResponse(
        {
            "session_id": auth_user_id or session_id,
            "event_count": len(events),
            "state": encode_timeline_state(events, encoder=TIMELINE_ENCODER),
            "encoder": TIMELINE_ENCODER,
            "summary": build_timeline_summary(events, max_items=10),
            "events": events_to_dict(events),
            "openviking_stats": layered_stats,
            "persistent": auth_user_id is not None,
            **expiry,
        }
    )


@app.get("/api/openviking/stats")
def openviking_stats(request: Request) -> JSONResponse:
    auth_user, auth_user_id = _get_auth_user(request)
    session_id = _get_session_id(request)
    layered_store = (
        _get_user_openviking_store(auth_user_id)
        if auth_user_id
        else _get_session_openviking_store(session_id)
    )
    vector_store = _get_user_store(auth_user_id) if auth_user_id else _get_session_store(session_id)
    return JSONResponse(
        {
            "ok": True,
            "session_id": auth_user_id or session_id,
            "enabled": OPENVIKING_ENABLED,
            "stats": layered_store.stats() if layered_store else {"total": 0, "l0": 0, "l1": 0, "sources": 0},
            "vector_chunks": len(vector_store.chunks) if vector_store is not None else 0,
            "persistent": auth_user_id is not None,
        }
    )


@app.post("/api/openviking/rebuild")
def openviking_rebuild(request: Request, reset: bool = True) -> JSONResponse:
    auth_user, auth_user_id = _get_auth_user(request)
    session_id = _get_session_id(request)
    if auth_user_id:
        vector_store = _get_user_store(auth_user_id)
        layered_store = _get_or_create_user_openviking_store(auth_user_id)
    else:
        vector_store = _get_session_store(session_id)
        layered_store = _get_or_create_session_openviking_store(session_id)
    if vector_store is None:
        return JSONResponse({"ok": False, "error": "no_active_store"}, status_code=400)

    rebuild = _rebuild_openviking_from_vector_store(layered_store, vector_store, reset=reset)
    out_stats = rebuild.get("stats", layered_store.stats())
    audit_logger.log(
        event_type="openviking_rebuild",
        session_id=auth_user_id or session_id,
        details={"sources": int(rebuild.get("sources", 0) or 0), "stats": out_stats, "reset": bool(reset)},
    )
    return JSONResponse(
        {
            "ok": True,
            "sources": int(rebuild.get("sources", 0) or 0),
            "rebuilt": rebuild.get("rebuilt", []),
            "stats": out_stats,
            "persistent": auth_user_id is not None,
        }
    )


@app.post("/api/session/expire-now")
def session_expire_now(request: Request) -> JSONResponse:
    session_id = _get_session_id(request)
    removed = _expire_session(session_id, reason="manual_demo")
    return JSONResponse({"ok": True, "session_id": session_id, "removed": removed})


@app.get("/api/literature/search")
def literature_search(q: str, top_k: int = 8) -> JSONResponse:
    if not q.strip():
        raise HTTPException(status_code=400, detail="q 不能为空")
    _refresh_local_literature_if_needed(force=False, max_results=LITERATURE_AGENT_BOOTSTRAP_MAX_RESULTS)
    max_k = min(max(top_k, 1), 30)
    items: list[dict[str, object]] = []
    mode = "local_agent"
    if ENABLE_LOCAL_LITERATURE_AGENT:
        items = search_from_local_store(
            q,
            embed_fn=lambda t: embed_text(client, EMBEDDING_MODEL, t),
            top_k=max_k,
            index=literature_index,
        )
    if not items and ENABLE_WEB_LITERATURE:
        mode = "web_fallback"
        items = [
            item.as_dict()
            for item in search_literature(
                q,
                providers=LITERATURE_PROVIDERS,
                top_k=max_k,
                timeout=LITERATURE_TIMEOUT_SECONDS,
                medical_oncology_only=LITERATURE_MEDICAL_ONCOLOGY_ONLY,
                min_relevance=LITERATURE_MIN_RELEVANCE,
            )
        ]
    return JSONResponse(
        {
            "ok": True,
            "items": items,
            "mode": mode,
            "local_records": len(literature_index),
            "providers": LITERATURE_PROVIDERS,
        }
    )


@app.post("/api/literature/refresh")
def literature_refresh(force: bool = True, max_results: int | None = None) -> JSONResponse:
    result = _refresh_local_literature_if_needed(force=force, max_results=max_results)
    return JSONResponse(result)


@app.get("/api/audit/recent")
def audit_recent(limit: int = 50) -> JSONResponse:
    return JSONResponse({"items": audit_logger.recent(limit=limit)})


@app.get("/api/audit/ttl-proof")
def audit_ttl_proof(session_id: str | None = None) -> JSONResponse:
    return JSONResponse({"items": audit_logger.ttl_proof(session_id=session_id)})


@app.get("/admin")
def admin_page() -> FileResponse:
    admin_html = STATIC_DIR / "admin.html"
    if not admin_html.exists():
        raise HTTPException(status_code=404, detail="Admin console not found")
    return FileResponse(admin_html)
