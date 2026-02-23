from __future__ import annotations

import hashlib
import io
import base64
import concurrent.futures
import json
import mimetypes
import os
import re
import shutil
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable
from urllib.parse import quote, unquote

from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, Request, Response, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from PIL import Image
from pydantic import BaseModel

from app.auth import UserManager
from pypdf import PdfReader

from app.audit import AuditLogger
from app.literature_agent import (
    LITERATURE_STATE_PATH,
    load_index,
    run_incremental_update,
    search_from_local_store,
)
from app.literature_search import (
    search_literature,
)
from app.llm import (
    build_client,
    embed_text,
    extract_date_fields_with_vision_bytes,
    extract_date_with_vision_bytes,
    ocr_with_aliyun_ocr_bytes,
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
from app.vector_store import VectorStore, build_session_store, build_user_store, get_vector_backend

try:
    import fitz  # type: ignore
except Exception:
    fitz = None

load_dotenv()

ROOT = Path(__file__).resolve().parent.parent
STATIC_DIR = ROOT / "web"
GUIDELINES_DIR = ROOT / "data" / "guidelines"
INTERNAL_OPENVIKING_DB = ROOT / "data" / "openviking_internal_layers.json"
GUIDELINES_VERSION_PATH = ROOT / "data" / "guidelines_versions.json"
AUDIT_LOG_PATH = ROOT / "data" / "audit_log.jsonl"
USER_DB_PATH = ROOT / "data" / "users.db"
USER_DATA_ROOT = ROOT / "data" / "user_data"
SESSION_UPLOAD_ORIGINALS_ROOT = ROOT / "data" / "session_upload_originals"

CHAT_MODEL = os.getenv("CHAT_MODEL", "qwen-plus")
VISION_MODEL = os.getenv("VISION_MODEL", "qwen-vl-max")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-v3")
SESSION_TTL_SECONDS = int(os.getenv("SESSION_TTL_SECONDS", "1800"))
INTERNAL_RAG_TOKEN = os.getenv("INTERNAL_RAG_TOKEN", "")
ENABLE_UPLOAD_DEID = os.getenv("ENABLE_UPLOAD_DEID", "true").strip().lower() in {"1", "true", "yes", "on"}
SAVE_UPLOAD_ORIGINALS = os.getenv("SAVE_UPLOAD_ORIGINALS", "false").strip().lower() in {"1", "true", "yes", "on"}
OCR_PROVIDER = os.getenv("OCR_PROVIDER", "auto").strip().lower()
PADDLE_OCR_LANG = os.getenv("PADDLE_OCR_LANG", "ch").strip()
PDF_OCR_MAX_PAGES = int(os.getenv("PDF_OCR_MAX_PAGES", "500"))
ALIYUN_OCR_MODEL = os.getenv("ALIYUN_OCR_MODEL", "qwen-vl-ocr-latest").strip()
_aliyun_ocr_min_pixels_raw = os.getenv("ALIYUN_OCR_MIN_PIXELS", "3072").strip()
_aliyun_ocr_max_pixels_raw = os.getenv("ALIYUN_OCR_MAX_PIXELS", "8388608").strip()
try:
    ALIYUN_OCR_MIN_PIXELS = max(int(_aliyun_ocr_min_pixels_raw), 0)
except Exception:
    ALIYUN_OCR_MIN_PIXELS = 3072
try:
    ALIYUN_OCR_MAX_PIXELS = max(int(_aliyun_ocr_max_pixels_raw), 0)
except Exception:
    ALIYUN_OCR_MAX_PIXELS = 8_388_608
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
        "(colorectal cancer OR colon cancer OR rectal cancer) "
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
OPENVIKING_RAG_L1_BUDGET = max(int(os.getenv("OPENVIKING_RAG_L1_BUDGET", "6")), 1)
OPENVIKING_RAG_L2_BUDGET = max(int(os.getenv("OPENVIKING_RAG_L2_BUDGET", "2")), 0)
OPENVIKING_RAG_DEEP_L1_BUDGET = max(int(os.getenv("OPENVIKING_RAG_DEEP_L1_BUDGET", "10")), OPENVIKING_RAG_L1_BUDGET)
OPENVIKING_RAG_DEEP_L2_BUDGET = max(int(os.getenv("OPENVIKING_RAG_DEEP_L2_BUDGET", "4")), OPENVIKING_RAG_L2_BUDGET)
OPENVIKING_INTERNAL_COMPLEX_SEARCH = (
    os.getenv("OPENVIKING_INTERNAL_COMPLEX_SEARCH", "false").strip().lower() in {"1", "true", "yes", "on"}
)
_chat_timeout_raw = os.getenv("CHAT_COMPLETION_TIMEOUT_SECONDS", "120").strip().lower()
if _chat_timeout_raw in {"0", "none", "off", "false", "no"}:
    CHAT_COMPLETION_TIMEOUT_SECONDS: float | None = None
else:
    try:
        CHAT_COMPLETION_TIMEOUT_SECONDS = max(float(_chat_timeout_raw or "120"), 5.0)
    except Exception:
        CHAT_COMPLETION_TIMEOUT_SECONDS = 120.0
CHAT_PROGRESS_TTL_SECONDS = max(int(os.getenv("CHAT_PROGRESS_TTL_SECONDS", "1800")), 60)

app = FastAPI(title="Colon Cancer RAG MVP")
client = build_client()
internal_openviking_store = OpenVikingStore(INTERNAL_OPENVIKING_DB, namespace="internal-guidelines")
audit_logger = AuditLogger(AUDIT_LOG_PATH)
user_manager = UserManager(USER_DB_PATH, USER_DATA_ROOT)

# ---------------------------------------------------------------------------
# Global LRU embedding cache (cross-request)
# ---------------------------------------------------------------------------
from collections import OrderedDict

_EMBED_CACHE_MAX = int(os.getenv("EMBED_CACHE_MAX", "2048"))
_global_embed_cache: OrderedDict[str, list[float]] = OrderedDict()


def _chat_timeout_kwargs() -> dict[str, Any]:
    if CHAT_COMPLETION_TIMEOUT_SECONDS is None:
        return {}
    return {"timeout": CHAT_COMPLETION_TIMEOUT_SECONDS}


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


def _ms(start: float, end: float | None = None) -> float:
    end_ts = time.perf_counter() if end is None else end
    return max((end_ts - start) * 1000.0, 0.0)

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


@dataclass
class ChatProgressState:
    request_id: str
    session_id: str
    step: str
    label: str
    status: str
    done: bool
    started_at: float
    updated_at: float
    elapsed_seconds: int = 0
    error: str = ""
    meta: dict[str, Any] = field(default_factory=dict)


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
chat_progress_states: dict[str, ChatProgressState] = {}
chat_progress_lock = threading.Lock()

CHAT_PROGRESS_STEPS: list[tuple[str, str]] = [
    ("accepted", "请求已接收"),
    ("analyzing_question", "解析问题"),
    ("retrieving_evidence", "检索证据"),
    ("building_context", "组装证据"),
    ("generating_answer", "生成答案"),
    ("validating_evidence", "校验证据"),
    ("completed", "完成"),
    ("failed", "失败"),
]
CHAT_PROGRESS_LABELS: dict[str, str] = {key: label for key, label in CHAT_PROGRESS_STEPS}
CHAT_PROGRESS_INDEX: dict[str, int] = {key: idx for idx, (key, _) in enumerate(CHAT_PROGRESS_STEPS)}

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


def _chat_progress_step_label(step: str) -> str:
    return CHAT_PROGRESS_LABELS.get(step, step)


def _cleanup_chat_progress(now: float | None = None) -> None:
    current = now if now is not None else time.time()
    expired: list[str] = []
    with chat_progress_lock:
        for request_id, state in chat_progress_states.items():
            if current - state.updated_at > CHAT_PROGRESS_TTL_SECONDS:
                expired.append(request_id)
        for request_id in expired:
            chat_progress_states.pop(request_id, None)


def _chat_progress_begin(request_id: str, session_id: str) -> None:
    now = time.time()
    state = ChatProgressState(
        request_id=request_id,
        session_id=session_id,
        step="accepted",
        label=_chat_progress_step_label("accepted"),
        status="running",
        done=False,
        started_at=now,
        updated_at=now,
        elapsed_seconds=0,
        error="",
        meta={},
    )
    with chat_progress_lock:
        chat_progress_states[request_id] = state
    _cleanup_chat_progress(now)


def _chat_progress_update(request_id: str, step: str, meta: dict[str, Any] | None = None) -> None:
    now = time.time()
    with chat_progress_lock:
        state = chat_progress_states.get(request_id)
        if state is None:
            return
        state.step = step
        state.label = _chat_progress_step_label(step)
        state.status = "running"
        state.done = False
        state.updated_at = now
        state.elapsed_seconds = max(int(now - state.started_at), 0)
        if meta:
            state.meta.update(meta)


def _chat_progress_complete(request_id: str, meta: dict[str, Any] | None = None) -> None:
    now = time.time()
    with chat_progress_lock:
        state = chat_progress_states.get(request_id)
        if state is None:
            return
        state.step = "completed"
        state.label = _chat_progress_step_label("completed")
        state.status = "completed"
        state.done = True
        state.updated_at = now
        state.elapsed_seconds = max(int(now - state.started_at), 0)
        state.error = ""
        if meta:
            state.meta.update(meta)


def _chat_progress_fail(request_id: str, error: str) -> None:
    now = time.time()
    with chat_progress_lock:
        state = chat_progress_states.get(request_id)
        if state is None:
            return
        state.step = "failed"
        state.label = _chat_progress_step_label("failed")
        state.status = "failed"
        state.done = True
        state.updated_at = now
        state.elapsed_seconds = max(int(now - state.started_at), 0)
        state.error = str(error or "unknown_error")[:300]


def _chat_progress_payload(request_id: str) -> dict[str, Any] | None:
    _cleanup_chat_progress()
    with chat_progress_lock:
        state = chat_progress_states.get(request_id)
        if state is None:
            return None
        return {
            "request_id": state.request_id,
            "session_id": state.session_id,
            "step": state.step,
            "label": state.label,
            "status": state.status,
            "done": state.done,
            "step_index": CHAT_PROGRESS_INDEX.get(state.step, -1),
            "elapsed_seconds": state.elapsed_seconds,
            "updated_at": int(state.updated_at),
            "started_at": int(state.started_at),
            "error": state.error,
            "meta": dict(state.meta),
            "steps": [{"key": key, "label": label} for key, label in CHAT_PROGRESS_STEPS],
        }


def _expire_session(session_id: str, reason: str = "ttl") -> bool:
    removed_store = session_stores.pop(session_id, None)
    removed_events = session_timeline_events.pop(session_id, [])
    removed_openviking = session_openviking_stores.pop(session_id, None)
    removed_original_files = _clear_session_originals(session_id)
    removed = removed_store is not None or bool(removed_events)
    if removed_openviking is not None:
        removed = True
    if removed_original_files > 0:
        removed = True
    if removed:
        audit_logger.log(
            event_type="session_expired",
            session_id=session_id,
            details={
                "ttl_seconds": SESSION_TTL_SECONDS,
                "dropped_timeline_events": len(removed_events),
                "dropped_openviking_items": removed_openviking.stats().get("total", 0) if removed_openviking else 0,
                "dropped_original_files": removed_original_files,
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


def _safe_upload_filename(file_name: str) -> str:
    name = Path(str(file_name or "unnamed")).name
    name = name.replace("/", "_").replace("\\", "_").strip()
    return name or "unnamed"


def _normalize_upload_source(source: str) -> str:
    src = str(source or "").strip()
    if not src:
        return "session/unnamed"
    return src if src.startswith("session/") else f"session/{src}"


def _upload_source_storage_name(source: str) -> str:
    normalized = _normalize_upload_source(source)
    base_name = _safe_upload_filename(normalized.removeprefix("session/"))
    digest = hashlib.sha1(normalized.encode("utf-8")).hexdigest()[:12]
    return f"{digest}__{base_name}"


def _relative_to_root(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except Exception:
        return str(path)


def _safe_header_value(value: str) -> str:
    text = str(value or "")
    try:
        text.encode("latin-1")
        return text
    except Exception:
        return quote(text, safe="/:@._-")


def _user_upload_originals_dir(user_id: str) -> Path:
    return user_manager.get_user_data_dir(user_id) / "upload_originals"


def _session_upload_originals_dir(session_id: str) -> Path:
    sid = _normalize_session_id(session_id) or "anonymous"
    return SESSION_UPLOAD_ORIGINALS_ROOT / sid


def _original_file_path_for_source(
    source: str,
    *,
    user_id: str | None = None,
    session_id: str | None = None,
) -> Path | None:
    name = _upload_source_storage_name(source)
    if user_id:
        return _user_upload_originals_dir(user_id) / name
    if session_id:
        return _session_upload_originals_dir(session_id) / name
    return None


def _save_upload_original(
    source: str,
    file_bytes: bytes,
    *,
    user_id: str | None = None,
    session_id: str | None = None,
) -> dict[str, Any]:
    out: dict[str, Any] = {"saved": False, "path": "", "size": int(len(file_bytes))}
    target_path = _original_file_path_for_source(source, user_id=user_id, session_id=session_id)
    if target_path is None:
        return out
    target_path.parent.mkdir(parents=True, exist_ok=True)
    target_path.write_bytes(file_bytes)
    out["saved"] = True
    out["path"] = _relative_to_root(target_path)
    return out


def _delete_upload_original(
    source: str,
    *,
    user_id: str | None = None,
    session_id: str | None = None,
) -> bool:
    target_path = _original_file_path_for_source(source, user_id=user_id, session_id=session_id)
    if target_path is None or not target_path.exists() or not target_path.is_file():
        return False
    try:
        target_path.unlink()
        return True
    except Exception:
        return False


def _clear_session_originals(session_id: str) -> int:
    folder = _session_upload_originals_dir(session_id)
    if not folder.exists():
        return 0
    removed_files = 0
    try:
        for item in folder.rglob("*"):
            if item.is_file():
                try:
                    item.unlink()
                    removed_files += 1
                except Exception:
                    pass
        shutil.rmtree(folder, ignore_errors=True)
    except Exception:
        pass
    return removed_files


def _is_within_base(path: Path, base: Path) -> bool:
    try:
        rp = path.resolve()
        rb = base.resolve()
        return rp == rb or rb in rp.parents
    except Exception:
        return False


def _resolve_original_file_for_source(
    source: str,
    *,
    user_id: str | None = None,
    session_id: str | None = None,
) -> tuple[Path | None, str]:
    raw = str(source or "").strip()
    if not raw:
        return None, ""

    if raw.startswith("viking://"):
        resolved = _resolve_source_from_viking_uri(raw, user_id=user_id, session_id=session_id)
        if resolved:
            raw = resolved
        else:
            return None, raw

    if raw.startswith("internal/"):
        rel = raw.removeprefix("internal/").strip().lstrip("/\\")
        if not rel:
            return None, "internal/"
        candidate = (GUIDELINES_DIR / rel).resolve()
        if not _is_within_base(candidate, GUIDELINES_DIR):
            return None, f"internal/{rel}"
        if candidate.exists() and candidate.is_file():
            normalized_rel = str(Path(rel).as_posix()).strip()
            return candidate, f"internal/{normalized_rel}"
        return None, f"internal/{rel}"

    normalized = _normalize_upload_source(raw)
    candidate = _original_file_path_for_source(
        normalized,
        user_id=user_id,
        session_id=session_id,
    )
    if candidate is None:
        return None, normalized
    if candidate.exists() and candidate.is_file():
        return candidate, normalized
    return None, normalized


def _resolve_source_from_viking_uri(
    viking_uri: str,
    *,
    user_id: str | None = None,
    session_id: str | None = None,
) -> str:
    uri = str(viking_uri or "").strip()
    if not uri.startswith("viking://"):
        return ""

    stores: list[OpenVikingStore] = [internal_openviking_store]
    if user_id:
        stores.append(_get_user_openviking_store(user_id))
    elif session_id:
        s_store = _get_session_openviking_store(session_id)
        if s_store is not None:
            stores.append(s_store)

    for store in stores:
        try:
            src = str(store.resolve_source(uri) or "").strip()
        except Exception:
            src = ""
        if src:
            return src

    prefix = "viking://resources/"
    if not uri.startswith(prefix):
        return ""
    relative = uri[len(prefix):].strip("/")
    if not relative:
        return ""
    # Fallback for URIs like:
    # viking://resources/internal-guidelines/internal/README_COLON.md
    m = re.search(r"(?:^|/)internal/(.+)$", relative)
    if m:
        rel = unquote(str(m.group(1) or "").strip().lstrip("/\\"))
        if rel:
            return f"internal/{rel}"

    # Fallback for session resources
    m2 = re.search(r"(?:^|/)session/(.+)$", relative)
    if m2:
        rel = unquote(str(m2.group(1) or "").strip().lstrip("/\\"))
        if rel:
            return f"session/{rel}"

    return ""


def _guess_file_media_type(file_path: Path, source_hint: str = "") -> str:
    candidates = [file_path.name, str(source_hint or "")]
    for name in candidates:
        guessed = mimetypes.guess_type(name)[0] or ""
        if guessed and guessed != "application/octet-stream":
            return guessed

    head = b""
    try:
        with open(file_path, "rb") as fin:
            head = fin.read(64)
    except Exception:
        return "application/octet-stream"

    if head.startswith(b"%PDF-"):
        return "application/pdf"
    if head.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if head.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if head.startswith((b"GIF87a", b"GIF89a")):
        return "image/gif"
    if head.startswith(b"BM"):
        return "image/bmp"
    if head.startswith((b"II*\x00", b"MM\x00*")):
        return "image/tiff"
    if len(head) >= 12 and head[:4] == b"RIFF" and head[8:12] == b"WEBP":
        return "image/webp"

    try:
        with Image.open(file_path) as img:
            fmt = str(getattr(img, "format", "") or "").upper()
        pil_map = {
            "JPEG": "image/jpeg",
            "PNG": "image/png",
            "GIF": "image/gif",
            "BMP": "image/bmp",
            "TIFF": "image/tiff",
            "WEBP": "image/webp",
            "ICO": "image/x-icon",
        }
        if fmt in pil_map:
            return pil_map[fmt]
    except Exception:
        pass

    return "application/octet-stream"


def _native_storage_root_path() -> Path:
    raw = os.getenv("OPENVIKING_NATIVE_STORAGE_PATH", "").strip()
    if raw:
        return Path(raw).expanduser().resolve()
    return (Path.cwd() / "data" / "openviking_native").resolve()


def _resolve_native_source_markdown(user_id: str, source: str) -> Path | None:
    user_dir = user_manager.get_user_data_dir(user_id)
    native_index = user_dir / "openviking_native_index.json"
    if not native_index.exists():
        return None
    try:
        payload = json.loads(native_index.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            return None
        item = payload.get(source)
        if not isinstance(item, dict):
            return None
        root_uri = str(item.get("root_uri") or item.get("target_uri") or "").strip()
        prefix = "viking://resources/"
        if not root_uri.startswith(prefix):
            return None
        relative = root_uri[len(prefix):].strip("/")
        if not relative:
            return None
        node = Path(relative).name
        return _native_storage_root_path() / "viking" / "resources" / relative / f"{node}.md"
    except Exception:
        return None


def _parse_openviking_native_markdown(markdown: str) -> tuple[str, list[dict[str, str]]]:
    text = str(markdown or "")
    l0 = ""
    l1_items: list[dict[str, str]] = []
    m_l0 = re.search(r"##\s*L0\s*(.*?)(?:\n##\s*L1|\Z)", text, flags=re.S)
    if m_l0:
        l0 = m_l0.group(1).strip()
    m_l1 = re.search(r"##\s*L1\s*(.*?)(?:\n##\s*L2|\Z)", text, flags=re.S)
    if not m_l1:
        return l0, l1_items
    block = m_l1.group(1).strip()
    for m in re.finditer(r"###\s*(.+?)\n(.*?)(?=\n###\s+|\Z)", block, flags=re.S):
        title = str(m.group(1) or "").strip()
        summary = str(m.group(2) or "").strip()
        if summary:
            l1_items.append({"title": title, "summary": summary})
    return l0, l1_items


def _load_native_openviking_source_snapshot(user_id: str, source: str) -> dict[str, Any]:
    md_path = _resolve_native_source_markdown(user_id, source)
    if md_path is None or not md_path.exists():
        return {"available": False}
    try:
        payload = md_path.read_text(encoding="utf-8")
        l0, l1_items = _parse_openviking_native_markdown(payload)
        return {
            "available": True,
            "path": _relative_to_root(md_path),
            "l0": l0,
            "l1": l1_items,
        }
    except Exception:
        return {"available": False}


def _chunk_order_key(chunk_id: str) -> tuple[int, str]:
    m = re.search(r"-(\d+)$", str(chunk_id or ""))
    if m:
        return int(m.group(1)), str(chunk_id or "")
    return 10**9, str(chunk_id or "")


def _sorted_source_chunks(store: VectorStore, source: str) -> list[Any]:
    matched = [c for c in getattr(store, "chunks", []) if str(getattr(c, "source", "")) == source]
    matched.sort(key=lambda c: _chunk_order_key(str(getattr(c, "id", ""))))
    return matched


def _merge_source_chunks_text(chunks: list[Any]) -> str:
    return "\n\n".join(str(getattr(c, "text", "") or "").strip() for c in chunks if str(getattr(c, "text", "") or "").strip())


def _normalize_text_for_compare(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def _first_diff_offset(left: str, right: str) -> int:
    a = str(left or "")
    b = str(right or "")
    size = min(len(a), len(b))
    for idx in range(size):
        if a[idx] != b[idx]:
            return idx
    if len(a) != len(b):
        return size
    return -1


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
        if any(k in line for k in ("诊断", "结肠", "直肠", "癌", "肿瘤")):
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
            **_chat_timeout_kwargs(),
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
    l2_texts = [str(getattr(c, "text", "") or "").strip() for c in chunks]
    return store.upsert_source_layers(
        source=source,
        l0_text=l0_text,
        l1_items=l1_items,
        embed_fn=_global_embed,
        l2_texts=[x for x in l2_texts if x],
    )


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


def _extract_page_hint(text: str) -> int | None:
    m = re.search(r"(?:\[)?第\s*(\d{1,4})\s*页(?:\])?", str(text or ""))
    if not m:
        return None
    try:
        page = int(m.group(1))
    except Exception:
        return None
    return page if page > 0 else None


def _extract_verbatim_quote(text: str, query: str, max_chars: int = 140) -> str:
    payload = str(text or "").strip()
    if not payload:
        return ""
    keywords = _extract_retrieval_keywords(query, max_terms=6)
    compact = re.sub(r"\s+", " ", payload).strip()
    segments = [
        s.strip()
        for s in re.split(r"[。！？!?；;\n]+", compact)
        if len(s.strip()) >= 8
    ]
    if not segments:
        return compact[:max_chars].strip()

    def _score(seg: str) -> tuple[int, int]:
        hit = 0
        low = seg.lower()
        for kw in keywords:
            if kw.lower() in low:
                hit += 1
        # Higher keyword overlap first, then prefer moderate sentence length.
        return hit, -abs(len(seg) - 52)

    ranked = sorted(segments, key=_score, reverse=True)
    picked = ranked[0].strip()
    if len(picked) <= max_chars:
        return picked
    return picked[:max_chars].strip()


def _verify_quote_in_text(source_text: str, quote: str) -> tuple[bool, str]:
    """Check whether *quote* can be found inside *source_text*.

    Returns (verified: bool, match_mode: str).
    We try progressively looser matching so that minor OCR / formatting
    differences don't cause false negatives.
    """
    base = str(source_text or "")
    q = str(quote or "").strip()
    if not base.strip() or not q:
        return False, "missing"
    # 1) exact substring
    if q in base:
        return True, "exact"
    # 2) whitespace-collapsed
    base_compact = re.sub(r"\s+", "", base)
    q_compact = re.sub(r"\s+", "", q)
    if q_compact in base_compact:
        return True, "compact"
    # 3) punctuation-normalised (strip all punctuation & whitespace)
    _punct_re = re.compile(r"[\s\u3000,，.。;；:：!！?？""\"'''\-—–\(\)（）\[\]【】{}\/<>《》、·…\u200b\ufeff]+")
    base_norm = _punct_re.sub("", base).lower()
    q_norm = _punct_re.sub("", q).lower()
    if q_norm and q_norm in base_norm:
        return True, "normalised"
    # 4) fuzzy: ≥70 % of quote characters appear in order in source
    if len(q_norm) >= 6:
        bi = 0
        matched = 0
        for ch in q_norm:
            pos = base_norm.find(ch, bi)
            if pos != -1:
                matched += 1
                bi = pos + 1
        if matched / len(q_norm) >= 0.70:
            return True, "fuzzy"
    return False, "missing"


def _build_source_text_index(vector_store: VectorStore | None) -> dict[str, str]:
    if vector_store is None:
        return {}
    out: dict[str, str] = {}
    by_source: dict[str, list[Any]] = {}
    for chunk in getattr(vector_store, "chunks", []):
        src = str(getattr(chunk, "source", "") or "").strip()
        if not src:
            continue
        by_source.setdefault(src, []).append(chunk)
    for src, chunks in by_source.items():
        chunks.sort(key=lambda c: _chunk_order_key(str(getattr(c, "id", ""))))
        text = _merge_source_chunks_text(chunks)
        if text:
            out[src] = text
    return out


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

    if provider in {"auto", "aliyun"}:
        try:
            prompt = "请逐行转写图片中的全部可见中文/英文文字，保留换行。"
            return ocr_with_aliyun_ocr_bytes(
                client,
                ALIYUN_OCR_MODEL,
                image_bytes,
                mime_type=mime_type,
                prompt_text=prompt,
                min_pixels=ALIYUN_OCR_MIN_PIXELS,
                max_pixels=ALIYUN_OCR_MAX_PIXELS,
            )
        except Exception as exc:
            errors.append(f"aliyun: {exc}")
            if provider == "aliyun":
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


# ---------------------------------------------------------------------------
# Comprehensive interpretation detection
# ---------------------------------------------------------------------------
_COMPREHENSIVE_INTERPRETATION_PATTERNS = [
    # Explicit tag injected by UI quick-action button
    re.compile(r"\[综合解读\]"),
    # Natural Chinese phrases requesting full report interpretation
    re.compile(r"(综合|全面|整体|所有|全部).*(解读|分析|评估|看看|看一下|看下|解释|总结|汇总)"),
    re.compile(r"(解读|分析|看看|评估).*(所有|全部|全面|整体|综合)"),
    re.compile(r"(帮我|请|麻烦).*(一起|综合|全面).*(解读|分析|看|评估)"),
    re.compile(r"(解读|分析|看看).*(报告|文件|资料|检查|检验|影像)"),
    re.compile(r"(报告|文件|资料|检查|检验|影像).*(解读|分析|看看|评估|看一下|看下)"),
    re.compile(r"(帮我|请).*(解读|分析|评估|看看|看一下|看下|说明|梳理)"),
    re.compile(r"(看看|分析).*(我的|目前|当前|现在).*(情况|状况|病情|结果)"),
    re.compile(r"(目前|当前|我的).*(情况|状况|病情).*(怎么样|如何|怎样|什么样)"),
]


def _is_comprehensive_interpretation(query: str) -> bool:
    """Return True if the user's query is requesting a comprehensive
    interpretation of all uploaded reports, optionally combined with timeline."""
    q = str(query or "").strip()
    if not q:
        return False
    return any(p.search(q) for p in _COMPREHENSIVE_INTERPRETATION_PATTERNS)


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


def _is_complex_viking_query(query: str, history: list[dict[str, str]] | None = None) -> bool:
    q = str(query or "").strip().lower()
    if not q:
        return False
    multi_terms = ("以及", "并且", "分别", "对比", "差异", "还是", "vs", "同时", "还想问")
    decision_terms = ("治疗", "方案", "风险", "证据", "依据", "流程", "指南", "recommend", "should")
    if any(t in q for t in multi_terms):
        return True
    if sum(1 for t in decision_terms if t in q) >= 2:
        return True
    if len(q) >= 60:
        return True
    return bool(history and len(history) >= 3)


def _needs_l2_drilldown(query: str) -> bool:
    q = str(query or "").strip().lower()
    if not q:
        return False
    l2_terms = (
        "具体数值",
        "具体数据",
        "百分比",
        "公式",
        "代码",
        "原文",
        "逐字",
        "表格",
        "第几页",
        "引用",
        "exact",
        "verbatim",
        "table",
    )
    return any(t in q for t in l2_terms)


def _wants_deep_answer(query: str) -> bool:
    q = str(query or "").strip().lower()
    if not q:
        return False
    return any(t in q for t in ("深入", "详细", "全面", "原理", "比较", "why", "how"))


def _build_response_style_hint(query: str, reasoning_mode: str, history: list[dict[str, str]] | None = None) -> str:
    q = str(query or "").strip().lower()
    multi_part_terms = ("以及", "并且", "分别", "对比", "差异", "优缺点", "方案", "风险", "依据")
    asks_structure = any(t in q for t in multi_part_terms)
    has_multi_turn = bool(history and len(history) >= 3)
    needs_structured = (
        str(reasoning_mode or "").lower() == "deep"
        or _wants_deep_answer(query)
        or len(q) >= 40
        or asks_structure
        or has_multi_turn
    )
    if not needs_structured:
        return (
            "输出风格：直接自然回答，控制在1-2段内。"
            "不要加固定标题，不要硬分结论/依据/下一步。"
            "不要使用任何 emoji。不要讨论临界/边缘值。"
            "若需给出证据，仅为关键结论添加少量[证据#N]。"
            "正常指标不要逐项列出。"
        )
    return (
        "输出风格：简洁、精准、句句到位。\n"
        "1. 结论先行：直接回答核心问题，结论加粗。\n"
        "2. 分点作答：用加粗列表项（- **要点**：说明）聚焦核心信息。\n"
        "3. 只给关键结论加 [证据#N]，不要每句都加。\n"
        "4. 不要使用任何 emoji 或表情符号。\n"
        "5. 不要讨论临界/边缘的正常指标，正常的一句话带过。\n"
        "6. 仅当确有必要时才列'下一步建议'，且不超过3-5条。"
    )


def _extract_retrieval_keywords(query: str, max_terms: int = 5) -> list[str]:
    terms = re.findall(r"[A-Za-z0-9_+\-]{2,}|[\u4e00-\u9fff]{2,}", str(query or ""))
    stop = {"请问", "这个", "那个", "一下", "什么", "怎么", "是否", "可以", "需要", "如果"}
    out: list[str] = []
    for t in terms:
        if t in stop:
            continue
        if t not in out:
            out.append(t)
        if len(out) >= max(max_terms, 1):
            break
    return out


def _source_title_from_uri(uri: str) -> str:
    return str(uri or "").rstrip("/").split("/")[-1]


def _run_viking_retrieval(
    query: str,
    stores: list[tuple[str, OpenVikingStore]],
    history: list[dict[str, str]],
    session_id: str,
    progress_cb: Callable[[dict[str, Any]], None] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    is_complex = _is_complex_viking_query(query, history)
    query_embedding = _global_embed(query)
    trace: dict[str, Any] = {
        "enabled": True,
        "mode": "search" if is_complex else "find",
        "stores": [],
    }
    merged: list[dict[str, Any]] = []
    total_stores = max(len(stores), 1)

    for store_idx, (store_name, store) in enumerate(stores, start=1):
        # Internal guideline store is usually much larger; prefer fast semantic find by default.
        # Can be overridden with OPENVIKING_INTERNAL_COMPLEX_SEARCH=true.
        use_search = is_complex and (
            store_name != "internal" or OPENVIKING_INTERNAL_COMPLEX_SEARCH
        )
        retrieval_mode = "search" if use_search else "find"
        if progress_cb:
            progress_cb(
                {
                    "retrieval_phase": "store_start",
                    "retrieval_store": store_name,
                    "retrieval_store_index": store_idx,
                    "retrieval_store_total": total_stores,
                    "retrieval_mode": retrieval_mode,
                    "retrieval_detail": f"{store_name} ({store_idx}/{total_stores})",
                }
            )
        store_session = None
        if use_search:
            store_session = store.session(session_id=f"{store.namespace}-{session_id}")
        if use_search:
            rows = store.search(
                query=query,
                target_uri=store.base_uri,
                session=store_session,
                limit=max(OPENVIKING_RAG_L1_BUDGET * 2, 8),
            )
        else:
            rows = store.find(
                query=query,
                target_uri=store.base_uri,
                limit=max(OPENVIKING_RAG_L1_BUDGET * 2, 8),
            )
        used_legacy_fallback = False
        if not rows:
            if progress_cb:
                progress_cb(
                    {
                        "retrieval_phase": "legacy_fallback",
                        "retrieval_store": store_name,
                        "retrieval_store_index": store_idx,
                        "retrieval_store_total": total_stores,
                        "retrieval_detail": f"{store_name} ({store_idx}/{total_stores}) · legacy",
                    }
                )
            # Native unavailable or returned empty; fallback to legacy layer search.
            legacy = store.search_layer(
                query=query,
                query_embedding=query_embedding,
                layer="L1",
                k=max(OPENVIKING_RAG_L1_BUDGET * 2, 8),
            )
            rows = []
            for score, item in legacy:
                legacy_uri = str((item.meta or {}).get("uri") or f"{store.base_uri}/{item.id}")
                rows.append(
                    {
                        "uri": legacy_uri,
                        "score": float(score),
                        "is_leaf": False,
                        "abstract": str(item.text or "").strip(),
                        "overview": str(item.text or "").strip(),
                    }
                )
            used_legacy_fallback = bool(rows)
        if progress_cb:
            progress_cb(
                {
                    "retrieval_phase": "store_done",
                    "retrieval_store": store_name,
                    "retrieval_store_index": store_idx,
                    "retrieval_store_total": total_stores,
                    "retrieval_hits": len(rows),
                    "retrieval_legacy_fallback": used_legacy_fallback,
                    "retrieval_detail": f"{store_name} ({store_idx}/{total_stores}) · hits={len(rows)}",
                }
            )
        trace["stores"].append(
                {
                    "store": store_name,
                    "target_uri": store.base_uri,
                    "mode": retrieval_mode,
                    "hits": len(rows),
                    "legacy_fallback": used_legacy_fallback,
                }
            )
        for item in rows:
            uri = str(item.get("uri", "") or "")
            if not uri:
                continue
            merged.append(
                {
                    "store_name": store_name,
                    "store": store,
                    "uri": uri,
                    "score": float(item.get("score", 0.0) or 0.0),
                    "is_leaf": bool(item.get("is_leaf", False)),
                    "abstract": str(item.get("abstract", "") or "").strip(),
                    "overview": str(item.get("overview", "") or "").strip(),
                }
            )
    merged.sort(key=lambda x: float(x.get("score", 0.0) or 0.0), reverse=True)
    return merged, trace


def _build_viking_evidence_bundle(
    query: str,
    hits: list[dict[str, Any]],
    *,
    force_deep: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    deep_mode = force_deep or _wants_deep_answer(query)
    need_l2 = _needs_l2_drilldown(query)
    l1_budget = OPENVIKING_RAG_DEEP_L1_BUDGET if deep_mode else OPENVIKING_RAG_L1_BUDGET
    l2_budget = OPENVIKING_RAG_DEEP_L2_BUDGET if deep_mode else OPENVIKING_RAG_L2_BUDGET
    if not need_l2:
        l2_budget = 0

    evidence: list[dict[str, Any]] = []
    seen_uri: set[str] = set()
    l1_count = 0

    for hit in hits:
        if l1_count >= l1_budget:
            break
        uri = str(hit.get("uri", "") or "")
        if not uri or uri in seen_uri:
            continue
        store = hit.get("store")
        if not isinstance(store, OpenVikingStore):
            continue
        source_key = store.resolve_source(uri)
        text = str(hit.get("overview", "") or hit.get("abstract", "")).strip()
        if not text:
            text = store.overview(uri)
        if not text:
            continue
        seen_uri.add(uri)
        l1_count += 1
        evidence.append(
            {
                "uri": uri,
                "citation": f"{uri}@L1",
                "layer": "L1",
                "score": float(hit.get("score", 0.0) or 0.0),
                "text": text,
                "store_name": str(hit.get("store_name", "") or ""),
                "title": _source_title_from_uri(uri),
                "source_key": source_key,
            }
        )

    if l1_count < l1_budget and hits:
        # Recall补救：在命中目录下展开子节点（ls）。
        for hit in hits[:3]:
            if l1_count >= l1_budget:
                break
            uri = str(hit.get("uri", "") or "")
            store = hit.get("store")
            if not uri or not isinstance(store, OpenVikingStore):
                continue
            children = store.ls(uri)
            for child in children[:6]:
                if l1_count >= l1_budget:
                    break
                child_uri = str(child.get("uri", "") or "")
                if not child_uri or child_uri in seen_uri:
                    continue
                source_key = store.resolve_source(child_uri)
                text = store.overview(child_uri)
                if not text:
                    continue
                seen_uri.add(child_uri)
                l1_count += 1
                evidence.append(
                    {
                        "uri": child_uri,
                        "citation": f"{child_uri}@L1",
                        "layer": "L1",
                        "score": float(hit.get("score", 0.0) or 0.0),
                        "text": text,
                        "store_name": str(hit.get("store_name", "") or ""),
                        "title": _source_title_from_uri(child_uri),
                        "source_key": source_key,
                    }
                )

    if l1_count < l1_budget and hits:
        # Recall仍不足时，按关键词做glob扩展。
        keywords = _extract_retrieval_keywords(query, max_terms=3)
        for hit in hits[:3]:
            if l1_count >= l1_budget:
                break
            uri = str(hit.get("uri", "") or "")
            store = hit.get("store")
            if not uri or not isinstance(store, OpenVikingStore):
                continue
            for kw in keywords:
                if l1_count >= l1_budget:
                    break
                matched = store.glob(pattern=f"*{kw}*", root_uri=uri)
                for m_uri in matched[:4]:
                    if l1_count >= l1_budget:
                        break
                    if not m_uri or m_uri in seen_uri:
                        continue
                    source_key = store.resolve_source(m_uri)
                    text = store.overview(m_uri)
                    if not text:
                        continue
                    seen_uri.add(m_uri)
                    l1_count += 1
                    evidence.append(
                        {
                            "uri": m_uri,
                            "citation": f"{m_uri}@L1",
                            "layer": "L1",
                            "score": float(hit.get("score", 0.0) or 0.0),
                            "text": text,
                            "store_name": str(hit.get("store_name", "") or ""),
                            "title": _source_title_from_uri(m_uri),
                            "source_key": source_key,
                        }
                    )

    if l2_budget > 0:
        l2_count = 0
        for hit in hits:
            if l2_count >= l2_budget:
                break
            uri = str(hit.get("uri", "") or "")
            store = hit.get("store")
            if not uri or not isinstance(store, OpenVikingStore):
                continue
            source_key = store.resolve_source(uri)
            text = store.read(uri)
            if not text:
                continue
            l2_count += 1
            evidence.append(
                {
                    "uri": uri,
                    "citation": f"{uri}#L?",
                    "layer": "L2",
                    "score": float(hit.get("score", 0.0) or 0.0),
                    "text": text[:2600],
                    "store_name": str(hit.get("store_name", "") or ""),
                    "title": _source_title_from_uri(uri),
                    "source_key": source_key,
                }
            )

    return evidence, {"l1_budget": l1_budget, "l2_budget": l2_budget, "need_l2": need_l2}


def _extract_source_citations(answer: str) -> set[str]:
    cited: set[str] = set()
    for m in re.finditer(r"\[source:\s*([^\]]+)\]", str(answer or ""), flags=re.IGNORECASE):
        value = str(m.group(1) or "").strip()
        if value:
            cited.add(value)
    return cited


def _extract_evidence_ranks(answer: str) -> set[int]:
    ranks: set[int] = set()
    for m in re.finditer(
        r"\[(?:证据|evidence)\s*#\s*(\d+)(?:\s*@\s*[A-Za-z0-9?_+\-]+)?\]",
        str(answer or ""),
        flags=re.IGNORECASE,
    ):
        try:
            rank = int(str(m.group(1) or "0"))
        except Exception:
            continue
        if rank > 0:
            ranks.add(rank)
    return ranks


def _normalize_answer_citations(answer: str, ranked_sources: list[dict[str, Any]] | None = None) -> str:
    _ = ranked_sources
    text = str(answer or "")
    if not text:
        return text

    # Normalize typed source tags: [pathology: ...] -> [source: ...]
    text = re.sub(
        r"\[(source|pathology|imaging|observation|treatment|lab|stage)\s*:\s*([^\]]+)\]",
        lambda m: f"[source: {str(m.group(2) or '').strip()}]",
        text,
        flags=re.IGNORECASE,
    )

    # Normalize literature tags to a single display form.
    text = re.sub(
        r"\[(?:literature|paper|文献)\s*#\s*(\d+)\]",
        lambda m: f"[文献#{m.group(1)}]",
        text,
        flags=re.IGNORECASE,
    )

    # --- Pre-expand compound evidence citations ---
    # e.g. [evidence#2, #25-27] -> [证据#2][证据#25][证据#26][证据#27]
    # e.g. [evidence#4, #5, #8, #11等] -> [证据#4][证据#5][证据#8][证据#11]
    def _expand_compound_evidence(match: re.Match[str]) -> str:
        inner = match.group(1)
        parts = re.split(r"[,，]", inner)
        nums: list[int] = []
        for part in parts:
            cleaned = part.replace("#", "").strip()
            range_m = re.match(r"^(\d+)\s*[-–]\s*(\d+)$", cleaned)
            if range_m:
                start, end = int(range_m.group(1)), int(range_m.group(2))
                for i in range(start, min(end + 1, start + 20)):
                    nums.append(i)
            else:
                try:
                    nums.append(int(cleaned))
                except ValueError:
                    pass
        if not nums:
            return match.group(0)
        return "".join(f"[证据#{n}]" for n in nums)

    text = re.sub(
        r"\[(?:evidence|证据)\s*#\s*(\d+(?:\s*[-–]\s*\d+)?(?:\s*[,，]\s*#?\s*\d+(?:\s*[-–]\s*\d+)?)*)(?:\s*等)?\]",
        _expand_compound_evidence,
        text,
        flags=re.IGNORECASE,
    )

    # Normalize evidence tags to a single display form: [证据#N].
    def _replace_evidence_tag(match: re.Match[str]) -> str:
        raw_rank = str(match.group(1) or "").strip()
        if not raw_rank.isdigit():
            return match.group(0)
        rank = int(raw_rank)
        return f"[证据#{rank}]"

    text = re.sub(
        r"\[(?:evidence|证据)\s*#\s*(\d+)(?:\s*@\s*[A-Za-z0-9?_+\-]+)?\]",
        _replace_evidence_tag,
        text,
        flags=re.IGNORECASE,
    )

    # Collapse immediate duplicated [source: ...] markers.
    text = re.sub(
        r"(\[source:\s*[^\]]+\])(?:\s+\1)+",
        r"\1",
        text,
        flags=re.IGNORECASE,
    )
    return text


def _build_openviking_evidence_guard(answer: str, evidence: list[dict[str, Any]]) -> dict[str, Any]:
    if not evidence:
        return {
            "coverage": 0.0,
            "unsupported_claims": 1,
            "verified_claims": 0,
            "checks": [],
        }
    cited_sources = _extract_source_citations(answer)
    cited_ranks = _extract_evidence_ranks(answer)
    expected = [str(e.get("citation", "") or "") for e in evidence]
    audit_scope = expected[: min(len(expected), 4)]

    covered_by_source = sum(1 for c in audit_scope if c and c in cited_sources)
    covered_by_rank = sum(1 for r in cited_ranks if 1 <= r <= len(audit_scope))
    covered = max(covered_by_source, covered_by_rank)
    required = 1 if len(audit_scope) <= 2 else 2
    coverage = round(float(covered) / max(len(audit_scope), 1), 4)
    return {
        "coverage": coverage,
        "unsupported_claims": 0 if covered >= required else 1,
        "verified_claims": covered,
        "checks": [
            {
                "type": "citation_presence",
                "required": required,
                "covered": covered,
                "scope": len(audit_scope),
                "by_source": covered_by_source,
                "by_rank": covered_by_rank,
            }
        ],
    }


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


class ChangeUsernameRequest(BaseModel):
    new_username: str


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
        "save_upload_originals": SAVE_UPLOAD_ORIGINALS,
        "ocr_provider": OCR_PROVIDER,
        "aliyun_ocr_model": ALIYUN_OCR_MODEL,
        "paddle_available": is_paddle_available(),
        "pdf_ocr_max_pages": PDF_OCR_MAX_PAGES,
        "internal_chunks": int(internal_openviking_store.stats().get("total", 0)),
        "active_sessions": len(session_stores),
        "active_timelines": len(session_timeline_events),
        "active_chat_progress": len(chat_progress_states),
        "session_ttl_seconds": SESSION_TTL_SECONDS,
        "chat_completion_timeout_seconds": CHAT_COMPLETION_TIMEOUT_SECONDS,
        "manual_session_clear_only": MANUAL_SESSION_CLEAR_ONLY,
        "audit_log_path": str(AUDIT_LOG_PATH.relative_to(ROOT)),
        "models": {"hidden": True},
        "timeline_encoder": TIMELINE_ENCODER,
        "openviking_enabled": OPENVIKING_ENABLED,
        "openviking_use_llm": OPENVIKING_USE_LLM,
        "openviking_l0_topk": OPENVIKING_L0_TOPK,
        "openviking_l1_topk": OPENVIKING_L1_TOPK,
        "openviking_l2_topk": OPENVIKING_L2_TOPK,
        "openviking_rag_l1_budget": OPENVIKING_RAG_L1_BUDGET,
        "openviking_rag_l2_budget": OPENVIKING_RAG_L2_BUDGET,
        "openviking_internal_complex_search": OPENVIKING_INTERNAL_COMPLEX_SEARCH,
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


@app.post("/api/user/change-username")
def user_change_username(request: Request, payload: ChangeUsernameRequest) -> JSONResponse:
    user, user_id = _get_auth_user(request)
    if not user:
        return JSONResponse({"ok": False, "error": "not_authenticated"}, status_code=401)

    old_username = user.username
    ok, result = user_manager.change_username(user_id=user_id, new_username=payload.new_username)
    if not ok:
        return JSONResponse({"ok": False, "error": result}, status_code=400)

    updated_user, _ = _get_auth_user(request)
    audit_logger.log(
        event_type="user_username_changed",
        session_id=user_id,
        details={
            "old_username": old_username,
            "new_username": updated_user.username if updated_user else payload.new_username,
        },
    )
    return JSONResponse({
        "ok": True,
        "user": user_manager.get_user_info(updated_user) if updated_user else {},
    })


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
    full_source = _normalize_upload_source(source)
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
    removed_original = _delete_upload_original(full_source, user_id=user_id)
    return JSONResponse({
        "ok": True,
        "removed_chunks": removed_chunks,
        "removed_events": removed_events,
        "removed_openviking_layers": removed_layers,
        "removed_original_file": removed_original,
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
    _auth_user, auth_user_id = _get_auth_user(request)
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
        file_name = _safe_upload_filename(f.filename or "unnamed")
        source_key = _normalize_upload_source(file_name)
        try:
            file_start = time.perf_counter()
            stage_ms: dict[str, float] = {}
            is_image, input_mime = _is_image_input(file_name, f.content_type)
            stage_t0 = time.perf_counter()
            file_bytes = await f.read()
            stage_ms["read_bytes_ms"] = round(_ms(stage_t0), 2)
            original_file = {"saved": False, "path": "", "size": int(len(file_bytes))}
            if SAVE_UPLOAD_ORIGINALS:
                original_file = _save_upload_original(
                    source_key,
                    file_bytes,
                    user_id=auth_user_id,
                    session_id=None if auth_user_id else session_id,
                )
            stage_t0 = time.perf_counter()
            text, extract_meta = _extract_file_text_from_bytes(file_name, f.content_type, file_bytes)
            stage_ms["extract_text_ms"] = round(_ms(stage_t0), 2)
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
            stage_t0 = time.perf_counter()
            if ENABLE_UPLOAD_DEID:
                text, redaction_stats = redact_sensitive_info(text)
                redaction = redaction_stats.as_dict()
                redaction_preview = extract_redaction_preview(text)
            stage_ms["deid_ms"] = round(_ms(stage_t0), 2)
            stage_t0 = time.perf_counter()
            if is_image and ENABLE_UPLOAD_DEID:
                image_redaction = _build_image_redaction_result(file_bytes, mime_type=input_mime)
            stage_ms["image_redaction_ms"] = round(_ms(stage_t0), 2)
            stage_t0 = time.perf_counter()
            chunks = split_text(text)
            stage_ms["split_ms"] = round(_ms(stage_t0), 2)
            stage_t0 = time.perf_counter()
            added = active_store.add_texts(
                source=source_key,
                texts=chunks,
                embed_fn=lambda t: embed_text(client, EMBEDDING_MODEL, t),
            )
            stage_ms["vector_add_ms"] = round(_ms(stage_t0), 2)
            layered_stats = {"l0": 0, "l1": 0}
            source_chunks: list[Any] = []
            if active_openviking_store is not None:
                stage_t0 = time.perf_counter()
                source_chunks = _sorted_source_chunks(active_store, source_key)
                stage_ms["source_chunks_ms"] = round(_ms(stage_t0), 2)
            else:
                stage_ms["source_chunks_ms"] = 0.0

            def _timed_call(fn: Callable[..., Any], *args: Any, **kwargs: Any) -> tuple[Any, float]:
                call_t0 = time.perf_counter()
                out = fn(*args, **kwargs)
                return out, _ms(call_t0)

            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
                timeline_future = executor.submit(
                    _timed_call,
                    _add_timeline_events,
                    session_id,
                    source_key,
                    text,
                    user_id=auth_user_id,
                )
                openviking_future: concurrent.futures.Future[tuple[Any, float]] | None = None
                if active_openviking_store is not None:
                    openviking_future = executor.submit(
                        _timed_call,
                        _upsert_openviking_layers,
                        active_openviking_store,
                        source_key,
                        source_chunks,
                    )
                timeline_out, timeline_elapsed = timeline_future.result()
                extracted_events, has_known_date = timeline_out
                stage_ms["timeline_ms"] = round(timeline_elapsed, 2)
                if openviking_future is not None:
                    layered_out, openviking_elapsed = openviking_future.result()
                    layered_stats = dict(layered_out or {})
                    stage_ms["openviking_ms"] = round(openviking_elapsed, 2)
                else:
                    stage_ms["openviking_ms"] = 0.0

            processing_ms = round(_ms(file_start), 2)
            stage_ms["total_ms"] = processing_ms
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
                    "processing_ms": processing_ms,
                    "stage_ms": stage_ms,
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
                    "original_file": original_file,
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
                    "original_file": original_file,
                    "timeline_events": extracted_events,
                    "openviking_l0": layered_stats.get("l0", 0),
                    "openviking_l1": layered_stats.get("l1", 0),
                    "processing_ms": processing_ms,
                    "stage_ms": stage_ms,
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
            "save_upload_originals": SAVE_UPLOAD_ORIGINALS,
        },
        status_code=status_code,
    )


@app.get("/api/upload/debug")
def upload_debug(
    request: Request,
    source: str = "",
    reextract: bool = False,
    preview_chars: int = 1200,
) -> JSONResponse:
    if not source.strip():
        return JSONResponse({"ok": False, "error": "source required"}, status_code=400)
    _auth_user, auth_user_id = _get_auth_user(request)
    session_id = _get_session_id(request)
    full_source = _normalize_upload_source(source)
    active_store = _get_user_store(auth_user_id) if auth_user_id else _get_session_store(session_id)
    if active_store is None:
        return JSONResponse({"ok": False, "error": "no_active_store"}, status_code=404)

    source_chunks = _sorted_source_chunks(active_store, full_source)
    extracted_text = _merge_source_chunks_text(source_chunks)
    max_preview = min(max(int(preview_chars), 120), 8000)
    chunk_samples = [
        {
            "id": str(getattr(chunk, "id", "")),
            "chars": len(str(getattr(chunk, "text", "") or "")),
            "text_preview": str(getattr(chunk, "text", "") or "")[: min(max_preview, 360)],
        }
        for chunk in source_chunks[:8]
    ]

    original_path = _original_file_path_for_source(
        full_source,
        user_id=auth_user_id,
        session_id=None if auth_user_id else session_id,
    )
    original_exists = bool(original_path and original_path.exists() and original_path.is_file())
    original_meta = {
        "enabled": SAVE_UPLOAD_ORIGINALS,
        "exists": original_exists,
        "path": _relative_to_root(original_path) if original_exists and original_path else "",
        "size": int(original_path.stat().st_size) if original_exists and original_path else 0,
        "updated_at": int(original_path.stat().st_mtime) if original_exists and original_path else 0,
    }

    l0_payload: list[dict[str, Any]] = []
    l1_payload: list[dict[str, Any]] = []
    openviking_from = "none"
    if OPENVIKING_ENABLED:
        layered_store = (
            _get_user_openviking_store(auth_user_id)
            if auth_user_id
            else _get_session_openviking_store(session_id)
        )
        if layered_store is not None:
            l0_items = layered_store.list_by_source(full_source, layer="L0")
            l1_items = layered_store.list_by_source(full_source, layer="L1")
            if l0_items or l1_items:
                openviking_from = "legacy_json"
                l0_payload = [
                    {"id": it.id, "text": it.text}
                    for it in l0_items
                ]
                l1_payload = [
                    {"id": it.id, "title": it.title, "summary": it.text}
                    for it in l1_items
                ]
            elif auth_user_id:
                native_snapshot = _load_native_openviking_source_snapshot(auth_user_id, full_source)
                if native_snapshot.get("available"):
                    openviking_from = "native_markdown"
                    l0_text = str(native_snapshot.get("l0", "")).strip()
                    if l0_text:
                        l0_payload = [{"id": f"{full_source}::L0::native", "text": l0_text}]
                    for idx, item in enumerate(native_snapshot.get("l1", []), start=1):
                        title = str((item or {}).get("title", "")).strip()
                        summary = str((item or {}).get("summary", "")).strip()
                        if summary:
                            l1_payload.append(
                                {
                                    "id": f"{full_source}::L1::native::{idx}",
                                    "title": title,
                                    "summary": summary,
                                }
                            )

    if auth_user_id:
        all_events = _load_user_timeline(auth_user_id)
    else:
        all_events = session_timeline_events.get(session_id, [])
    source_events = [ev for ev in all_events if str(getattr(ev, "source", "")) == full_source]

    reextract_result: dict[str, Any] = {
        "requested": bool(reextract),
        "ok": False,
        "error": "",
    }
    if reextract:
        if not original_exists or original_path is None:
            reextract_result["error"] = "original_not_found"
        else:
            try:
                original_bytes = original_path.read_bytes()
                file_name = _safe_upload_filename(full_source.removeprefix("session/"))
                new_text, _meta = _extract_file_text_from_bytes(file_name, None, original_bytes)
                if ENABLE_UPLOAD_DEID:
                    new_text, _ = redact_sensitive_info(new_text)
                normalized_stored = _normalize_text_for_compare(extracted_text)
                normalized_new = _normalize_text_for_compare(new_text)
                diff_at = _first_diff_offset(normalized_stored, normalized_new)
                reextract_result = {
                    "requested": True,
                    "ok": True,
                    "stored_chars": len(extracted_text),
                    "reextracted_chars": len(new_text),
                    "normalized_equal": normalized_stored == normalized_new,
                    "first_diff_offset": diff_at,
                    "reextracted_preview": new_text[:max_preview],
                }
            except Exception as exc:
                reextract_result = {
                    "requested": True,
                    "ok": False,
                    "error": f"{type(exc).__name__}: {exc}",
                }

    return JSONResponse(
        {
            "ok": True,
            "source": full_source,
            "persistent": auth_user_id is not None,
            "scope": "user" if auth_user_id else "session",
            "original_file": original_meta,
            "extraction": {
                "chunk_count": len(source_chunks),
                "total_chars": len(extracted_text),
                "text_preview": extracted_text[:max_preview],
                "chunk_samples": chunk_samples,
            },
            "timeline": {
                "event_count": len(source_events),
                "events_preview": events_to_dict(source_events[:10]),
            },
            "openviking": {
                "enabled": OPENVIKING_ENABLED,
                "source": openviking_from,
                "l0_count": len(l0_payload),
                "l1_count": len(l1_payload),
                "l0": l0_payload,
                "l1": l1_payload,
            },
            "reextract": reextract_result,
        }
    )


@app.get("/api/source/original")
def source_original(request: Request, source: str = "") -> Response:
    if not source.strip():
        return JSONResponse({"ok": False, "error": "source required"}, status_code=400)

    _auth_user, auth_user_id = _get_auth_user(request)
    session_id = _get_session_id(request)
    file_path, source_key = _resolve_original_file_for_source(
        source,
        user_id=auth_user_id,
        session_id=None if auth_user_id else session_id,
    )
    if file_path is None:
        return JSONResponse(
            {
                "ok": False,
                "error": "original_not_found",
                "source": source_key,
                "save_upload_originals": SAVE_UPLOAD_ORIGINALS,
            },
            status_code=404,
        )

    media_type = _guess_file_media_type(file_path, source_hint=source_key)
    return FileResponse(
        file_path,
        media_type=media_type,
        filename=file_path.name,
        headers={
            "Cache-Control": "no-store",
            "X-Source-Key": _safe_header_value(source_key),
        },
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
        internal_openviking_store.clear()
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
        internal_openviking_store.remove_source(del_source)

    results = []
    has_error = False
    skipped = 0
    for file_path in files:
        relative = str(file_path.relative_to(GUIDELINES_DIR)).replace("\\", "/")
        source = f"internal/{relative}"
        if file_path not in changed_files:
            skipped += 1
            continue
        # Remove old layers for this file before re-adding.
        internal_openviking_store.remove_source(source)
        try:
            text = _extract_file_text(file_path, None, ocr_mode="guide")
            chunks = split_text(text)
            class _ImportChunk:
                def __init__(self, text: str) -> None:
                    self.text = text

            pseudo_chunks = [_ImportChunk(item) for item in chunks]
            layered = _upsert_openviking_layers(
                internal_openviking_store,
                source=source,
                chunks=pseudo_chunks,
            )
            results.append(
                {
                    "file": relative,
                    "chunks": len(chunks),
                    "openviking_l0": int(layered.get("l0", 0)),
                    "openviking_l1": int(layered.get("l1", 0)),
                    "ok": True,
                    "action": "updated",
                }
            )
        except HTTPException as exc:
            has_error = True
            results.append({"file": relative, "ok": False, "error": exc.detail})
        except Exception as exc:
            has_error = True
            results.append({"file": relative, "ok": False, "error": str(exc)})

    if OPENVIKING_ENABLED:
        wait_state = internal_openviking_store.wait_processed(timeout=120.0)
    else:
        wait_state = {"ok": False, "reason": "openviking_disabled"}

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
        "total_chunks": int(internal_openviking_store.stats().get("total", 0)),
    })
    manifest["history"] = history[-50:]  # keep last 50 entries
    _save_guideline_versions(manifest)

    status_code = 207 if has_error else 200
    return JSONResponse(
        {
            "ok": not has_error,
            "results": results,
            "internal_chunks": int(internal_openviking_store.stats().get("total", 0)),
            "wait_processed": wait_state,
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


@app.get("/api/chat/progress")
def chat_progress(request_id: str) -> JSONResponse:
    payload = _chat_progress_payload(request_id)
    if payload is None:
        # Polling may arrive slightly earlier than /api/chat registers this request.
        # Return pending to avoid noisy transient 404s in access logs.
        return JSONResponse(
            {
                "ok": True,
                "progress": {
                    "request_id": request_id,
                    "step": "accepted",
                    "label": _chat_progress_step_label("accepted"),
                    "status": "pending",
                    "done": False,
                    "elapsed_seconds": 0,
                    "meta": {"pending": True},
                },
            },
            status_code=202,
        )
    return JSONResponse({"ok": True, "progress": payload})


@app.post("/api/chat")
def chat(payload: ChatRequest, request: Request) -> JSONResponse:
    if not payload.message.strip():
        raise HTTPException(status_code=400, detail="\u6d88\u606f\u4e0d\u80fd\u4e3a\u7a7a")

    incoming_request_id = request.headers.get("x-chat-request-id", "").strip()
    chat_request_id = _normalize_session_id(incoming_request_id) or f"c-{uuid.uuid4().hex[:16]}"
    auth_user, auth_user_id = _get_auth_user(request)
    session_id = _get_session_id(request)
    progress_session_id = auth_user_id or session_id
    _chat_progress_begin(chat_request_id, progress_session_id)

    try:
        _chat_progress_update(chat_request_id, "analyzing_question")
        timeline_events = _get_timeline_events(session_id, user_id=auth_user_id)
        is_comprehensive = _is_comprehensive_interpretation(payload.message)
        reasoning_mode, reasoning_meta = _assess_reasoning_mode(
            payload.message,
            timeline_events,
            payload.history,
        )
        if is_comprehensive:
            reasoning_mode = "deep"
            reasoning_meta["comprehensive"] = True
        timeline_state = encode_timeline_state(timeline_events, encoder=TIMELINE_ENCODER)
        retrieval_hint = build_retrieval_hint(payload.message, timeline_events, state=timeline_state)
        timeline_summary = build_timeline_summary(
            timeline_events,
            max_items=30 if is_comprehensive else 8,
        )
        openviking_trace: dict[str, Any] = {"enabled": OPENVIKING_ENABLED, "comprehensive": is_comprehensive}

        user_store = _get_user_store(auth_user_id) if auth_user_id else _get_session_store(session_id)
        user_openviking_store = (
            _get_user_openviking_store(auth_user_id)
            if auth_user_id
            else _get_session_openviking_store(session_id)
        ) if OPENVIKING_ENABLED else None

        if user_store is not None and user_openviking_store is not None:
            layered_total = int(user_openviking_store.stats().get("total", 0) or 0)
            if layered_total == 0 and len(getattr(user_store, "chunks", [])) > 0:
                auto_rebuild = _rebuild_openviking_from_vector_store(
                    user_openviking_store,
                    user_store,
                    reset=False,
                )
                openviking_trace["auto_rebuild"] = {
                    "sources": int(auto_rebuild.get("sources", 0) or 0),
                    "stats": auto_rebuild.get("stats", {}),
                }

        stores: list[tuple[str, OpenVikingStore]] = [("internal", internal_openviking_store)]
        if user_openviking_store is not None:
            stores.append(("user", user_openviking_store))

        _chat_progress_update(
            chat_request_id,
            "retrieving_evidence",
            meta={"retrieval_detail": f"准备检索（0/{len(stores)}）"},
        )
        all_hits, retrieval_trace = _run_viking_retrieval(
            query=payload.message,
            stores=stores,
            history=payload.history,
            session_id=progress_session_id,
            progress_cb=lambda meta: _chat_progress_update(chat_request_id, "retrieving_evidence", meta=meta),
        )
        openviking_trace.update(retrieval_trace)

        _chat_progress_update(chat_request_id, "building_context", meta={"retrieved_hits": len(all_hits)})
        evidence_items, budget_trace = _build_viking_evidence_bundle(
            payload.message,
            all_hits,
            force_deep=is_comprehensive,
        )
        openviking_trace.update(budget_trace)
        openviking_trace["retrieved_hits"] = len(all_hits)
        openviking_trace["evidence_items"] = len(evidence_items)

        ranked_sources: list[dict[str, Any]] = []
        source_text_index = _build_source_text_index(user_store)
        context_rows: list[str] = []
        for rank, item in enumerate(evidence_items, start=1):
            item_text = str(item.get("text", "") or "")
            snippet = _build_evidence_snippet(
                item_text,
                payload.message,
                max_chars=1600 if str(item.get("layer", "L1")) == "L2" else 900,
            )
            citation = str(item.get("citation", "") or "")
            uri = str(item.get("uri", "") or "")
            layer = str(item.get("layer", "L1"))
            source_key = str(item.get("source_key", "") or "").strip()
            source_doc = source_key or uri
            source_display = source_doc.removeprefix("session/").removeprefix("internal/")
            quote = _extract_verbatim_quote(
                item_text,
                payload.message,
                max_chars=180 if layer == "L2" else 120,
            )
            source_snapshot = source_text_index.get(source_key, "")
            if source_snapshot:
                quote_verified, quote_match_mode = _verify_quote_in_text(source_snapshot, quote)
                quote_scope = "source_text"
            else:
                quote_verified, quote_match_mode = _verify_quote_in_text(item_text, quote)
                quote_scope = "evidence_text"
            page_hint = _extract_page_hint(item_text) or _extract_page_hint(snippet)
            score = round(float(item.get("score", 0.0) or 0.0), 4)
            store_name = str(item.get("store_name", "openviking") or "openviking")
            context_rows.append(
                f"[evidence#{rank}] [source: {citation}] [layer: {layer}] [store: {store_name}]\n{snippet}"
            )
            ranked_sources.append(
                {
                    "rank": rank,
                    "score": score,
                    "chunk_id": f"{uri}::{layer}",
                    "source": uri,
                    "source_raw": uri,
                    "source_doc": source_doc,
                    "source_display": source_display,
                    "type": store_name,
                    "layer": layer,
                    "title": str(item.get("title", "") or ""),
                    "page": page_hint,
                    "section": None,
                    "text_length": len(item_text),
                    "preview": snippet[:220],
                    "evidence": snippet,
                    "quote": quote,
                    "quote_verified": quote_verified,
                    "quote_match_mode": quote_match_mode,
                    "quote_scope": quote_scope,
                    "quote_page": page_hint,
                    "citation": citation,
                }
            )

        next_keywords = _extract_retrieval_keywords(payload.message, max_terms=5)
        citation_warning = ""
        style_hint = _build_response_style_hint(payload.message, reasoning_mode, payload.history)

        if not evidence_items:
            # ── Determine if we can fall back to history/timeline context ──
            has_history = bool(payload.history and len(payload.history) >= 1)
            has_timeline = bool(timeline_events and len(timeline_events) >= 1)
            has_context_fallback = has_history or has_timeline

            if has_context_fallback:
                # Build a context-aware response using chat history + timeline
                _chat_progress_update(chat_request_id, "generating_answer", meta={"fallback": "history_timeline"})

                fallback_context_parts: list[str] = []
                if has_timeline and timeline_summary.strip():
                    fallback_context_parts.append(f"患者病程摘要：\n{timeline_summary}")
                if has_history:
                    recent_history = payload.history[-6:]
                    history_text = "\n".join(
                        f"{'用户' if m.get('role') == 'user' else '助手'}: {m.get('content', '')[:600]}"
                        for m in recent_history
                        if m.get("content", "").strip()
                    )
                    if history_text.strip():
                        fallback_context_parts.append(f"近期对话记录：\n{history_text}")

                fallback_context = "\n\n".join(fallback_context_parts).strip()

                fallback_messages = [
                    {
                        "role": "system",
                        "content": (
                            "当前未检索到直接证据片段，但你拥有患者的对话历史和/或病程时间线。"
                            "请基于这些已有信息直接回答用户的问题，问什么答什么。"
                            "如果用户要求通俗解释，请用容易理解的语言重新表述已有信息。"
                            "不要编造不存在的检查结果或数据；如果信息不足以回答某个方面，请坦诚说明。"
                        ),
                    },
                    {"role": "system", "content": style_hint},
                    {
                        "role": "system",
                        "content": (
                            f"患者时序状态({TIMELINE_ENCODER}): "
                            f"risk_level={timeline_state.get('risk_level')} "
                            f"event_count={int(float(timeline_state.get('event_count', 0.0)))}\n"
                            f"已有上下文信息：\n{fallback_context}"
                        ),
                    },
                ]
                for msg in payload.history[-8:]:
                    role = msg.get("role", "user")
                    content = msg.get("content", "")
                    if role in {"user", "assistant"} and content:
                        fallback_messages.append({"role": role, "content": content})
                fallback_messages.append({"role": "user", "content": payload.message})

                resp = client.chat.completions.create(
                    model=CHAT_MODEL,
                    messages=fallback_messages,
                    temperature=0.3,
                    **_chat_timeout_kwargs(),
                )
                answer = _sanitize_model_disclosure(resp.choices[0].message.content or "暂无回复")
                citation_warning = "本轮回答基于对话历史与病程时间线生成，未使用直接检索证据。"
            else:
                # Truly no context at all — return the hard "证据不足" message
                keyword_text = ", ".join(next_keywords) if next_keywords else payload.message[:30]
                answer = (
                    "证据不足：当前 OpenViking 检索未召回可直接支撑该问题的材料。"
                    "请补充更具体的检查项、时间范围或治疗阶段。"
                    f"可尝试检索关键词：{keyword_text}"
                )

            evidence_guard = {
                "coverage": 0.0,
                "unsupported_claims": 0 if has_context_fallback else 1,
                "verified_claims": 0,
                "checks": [],
            }
            literature_guard = {"coverage": 0.0}
            key_conclusions: list[dict[str, Any]] = []
        else:
            _chat_progress_update(chat_request_id, "generating_answer", meta={"evidence_items": len(evidence_items)})
            context = "\n\n".join(context_rows).strip()
            if is_comprehensive:
                # -- Comprehensive interpretation prompt --
                messages = [
                    {
                        "role": "system",
                        "content": (
                            "你是一位资深临床医生，正在为患者综合解读全部检查报告。"
                            "核心原则：简洁、精准、只讲有临床意义的发现，句句到位。\n\n"
                            "必须遵守的输出规则：\n"
                            "1. 只讨论有明确临床意义的异常——轻微偏离参考值但无临床后果的指标不要提。\n"
                            "2. 不要使用任何 emoji 或表情符号。\n"
                            "3. 不要为了显得全面而罗列正常指标。正常就一句话带过：'肝肾功能正常'即可。\n"
                            "4. 不要做过度推测或展开鉴别诊断长串。聚焦当前诊断和治疗相关。\n"
                            "5. 语言通俗但不失专业准确性，让患者家属能看懂。"
                        ),
                    },
                    {
                        "role": "system",
                        "content": (
                            "输出结构（简洁版）：\n"
                            "1. 总体评价：1-2句话概括当前状态和治疗效果，结论加粗。\n"
                            "2. 关键发现：只列出真正需要关注的异常项（通常2-5项），每项用加粗标题+1-2句解释。\n"
                            "3. 趋势判断：与既往数据对比，指出好转/稳定/恶化。\n"
                            "4. 建议：仅列出确实需要做的事，不超过3-5条。\n\n"
                            "不要反问用户要解读哪份报告。直接综合解读。\n"
                            "引用关键数据时标注 [证据#N]，但只给关键结论加，不要每句都加。"
                        ),
                    },
                    {
                        "role": "system",
                        "content": (
                            f"患者病程时间线（{TIMELINE_ENCODER}）：\n"
                            f"风险等级: {timeline_state.get('risk_level')}\n"
                            f"事件总数: {int(float(timeline_state.get('event_count', 0.0)))}\n"
                            f"高风险事件: {int(float(timeline_state.get('high_risk_events', 0.0)))}\n"
                            f"综合关注指数: {timeline_state.get('risk_score', 0)}\n\n"
                            f"病程摘要:\n{timeline_summary}"
                        ),
                    },
                    {"role": "system", "content": f"以下是用户上传的所有报告/检查证据，请综合解读：\n{context}"},
                ]
            else:
                # -- Standard RAG prompt --
                messages = [
                    {
                        "role": "system",
                        "content": (
                            "你是一位资深临床医生，基于检索到的证据回答患者问题。"
                            "所有事实与结论必须来自当前给定的检索证据，不得用训练记忆补充事实。"
                            "若证据不足，明确说明'证据不足'。\n\n"
                            "核心原则：\n"
                            "- 问什么答什么，不要默认扩展为全局病情总结。\n"
                            "- 只讨论有临床意义的发现，忽略临界/边缘值。\n"
                            "- 不要使用任何 emoji 或表情符号。\n"
                            "- 不要为了显得全面而罗列一堆正常指标或展开鉴别诊断。\n"
                            "- 简洁输出，句句到位。"
                        ),
                    },
                    {
                        "role": "system",
                        "content": (
                            "输出规则：\n"
                            "1. 简单问题直接回答，1-2段即可。复杂问题分点作答，使用加粗标题。\n"
                            "2. 引用策略：仅给关键结论加 [证据#N]，不要每句都加。\n"
                            "3. 不要为了凑结构新增用户未提问的内容。\n"
                            "4. 正常指标一笔带过，不要逐项罗列。"
                        ),
                    },
                    {"role": "system", "content": style_hint},
                    {
                        "role": "system",
                        "content": (
                            "以下信息是可选背景，仅在与本轮问题直接相关时使用；若不相关请忽略，不得据此扩展结论。\n"
                            f"患者时序状态({TIMELINE_ENCODER}): "
                            f"risk_level={timeline_state.get('risk_level')} "
                            f"event_count={int(float(timeline_state.get('event_count', 0.0)))}\n"
                            f"检索路由建议: {retrieval_hint}\n"
                            f"病程摘要:\n{timeline_summary}"
                        ),
                    },
                    {
                        "role": "system",
                        "content": (
                            "若用户问题过短、存在多种解释或缺少对象，请先给出一句澄清问题，"
                            "并提供最多3个可选理解方向；在澄清前不要做全面展开。"
                        ),
                    },
                    {"role": "system", "content": f"OpenViking 检索证据（仅可依据以下内容回答）:\n{context}"},
                ]
            for msg in payload.history[-8:]:
                role = msg.get("role", "user")
                content = msg.get("content", "")
                if role in {"user", "assistant"} and content:
                    messages.append({"role": role, "content": content})
            messages.append({"role": "user", "content": payload.message})

            resp = client.chat.completions.create(
                model=CHAT_MODEL,
                messages=messages,
                temperature=0.1,
                **_chat_timeout_kwargs(),
            )
            answer = _sanitize_model_disclosure(resp.choices[0].message.content or "暂无回复")
            answer = _normalize_answer_citations(answer, ranked_sources)
            cited_sources = _extract_source_citations(answer)
            cited_ranks = _extract_evidence_ranks(answer)
            if not cited_sources and not cited_ranks:
                citation_warning = "回答未显式标注证据，已附加关键证据编号索引。"
                rank_refs = [f"[证据#{int(item.get('rank', 0) or 0)}]" for item in ranked_sources[:3] if int(item.get("rank", 0) or 0) > 0]
                if rank_refs:
                    answer = f"{answer.rstrip()}\n\n参考来源：{' '.join(rank_refs)}（详见下方证据卡片）"

            _chat_progress_update(chat_request_id, "validating_evidence")
            evidence_guard = _build_openviking_evidence_guard(answer, evidence_items)
            literature_guard = {"coverage": 0.0}
            key_conclusions = []

        audit_logger.log(
            event_type="chat_completed",
            session_id=progress_session_id,
            details={
                "query_length": len(payload.message),
                "retrieved_viking_hits": len(all_hits),
                "returned_sources": len(ranked_sources),
                "reasoning_mode": reasoning_mode,
                "reasoning_meta": reasoning_meta,
                "timeline_events": len(timeline_events),
                "openviking_trace": openviking_trace,
                "evidence_coverage": evidence_guard.get("coverage", 0),
            },
        )

        _chat_progress_complete(
            chat_request_id,
            meta={
                "retrieved_hits": len(all_hits),
                "evidence_items": len(evidence_items),
                "returned_sources": len(ranked_sources),
                "evidence_coverage": float(evidence_guard.get("coverage", 0.0) or 0.0),
            },
        )

        return JSONResponse(
            {
                "answer": answer,
                "sources": ranked_sources,
                "timeline_state": timeline_state,
                "timeline_summary": timeline_summary,
                "retrieval_hint": retrieval_hint,
                "evidence_guard": evidence_guard,
                "citation_warning": citation_warning,
                "key_conclusions": key_conclusions,
                "external_sources": [],
                "external_literature_enabled": False,
                "reasoning_mode": reasoning_mode,
                "reasoning_meta": reasoning_meta,
                "openviking_trace": openviking_trace,
                "literature_guard": literature_guard,
                "auto_rewrite_applied": False,
                "rewrite_triggered": False,
                "rewrite_rounds": 0,
                "evidence_conflicts": [],
                "chat_request_id": chat_request_id,
            }
        )
    except Exception as exc:
        _chat_progress_fail(chat_request_id, str(exc))
        raise


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
