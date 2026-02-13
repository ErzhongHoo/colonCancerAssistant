from __future__ import annotations

import io
import mimetypes
import os
import re
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, Request, Response, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
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
from app.llm import build_client, embed_text, ocr_with_vision_bytes, ocr_with_vision_bytes_transcribe
from app.ocr import is_paddle_available, ocr_with_paddle_bytes
from app.privacy import redact_sensitive_info
from app.rag import split_text
from app.timeline import (
    ClinicalEvent,
    build_retrieval_hint,
    build_timeline_summary,
    encode_timeline_state,
    events_to_dict,
    extract_events,
    merge_events,
)
from app.vector_store import VectorStore, build_internal_store, build_session_store, get_vector_backend

try:
    import fitz  # type: ignore
except Exception:
    fitz = None

load_dotenv()

ROOT = Path(__file__).resolve().parent.parent
STATIC_DIR = ROOT / "web"
GUIDELINES_DIR = ROOT / "data" / "guidelines"
INTERNAL_VECTOR_DB = ROOT / "data" / "internal_vector_store.json"
AUDIT_LOG_PATH = ROOT / "data" / "audit_log.jsonl"

CHAT_MODEL = os.getenv("CHAT_MODEL", "qwen-plus")
VISION_MODEL = os.getenv("VISION_MODEL", "qwen-vl-max")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-v3")
SESSION_TTL_SECONDS = int(os.getenv("SESSION_TTL_SECONDS", "1800"))
INTERNAL_RAG_TOKEN = os.getenv("INTERNAL_RAG_TOKEN", "")
ENABLE_UPLOAD_DEID = os.getenv("ENABLE_UPLOAD_DEID", "true").strip().lower() in {"1", "true", "yes", "on"}
OCR_PROVIDER = os.getenv("OCR_PROVIDER", "auto").strip().lower()
PADDLE_OCR_LANG = os.getenv("PADDLE_OCR_LANG", "ch").strip()
PDF_OCR_MAX_PAGES = int(os.getenv("PDF_OCR_MAX_PAGES", "500"))
TIMELINE_ENCODER = os.getenv("TIMELINE_ENCODER", "mamba").strip().lower()
if TIMELINE_ENCODER not in {"linear", "ssm", "mamba"}:
    TIMELINE_ENCODER = "mamba"
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
LITERATURE_TOP_K = min(max(int(os.getenv("LITERATURE_TOP_K", "5")), 1), 5)
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

app = FastAPI(title="Colon Cancer RAG MVP")
client = build_client()
internal_store = build_internal_store(INTERNAL_VECTOR_DB)
audit_logger = AuditLogger(AUDIT_LOG_PATH)
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


@dataclass
class SessionStoreState:
    store: VectorStore
    updated_at: float


session_stores: dict[str, SessionStoreState] = {}
session_timeline_events: dict[str, list[ClinicalEvent]] = {}
literature_index: dict[str, dict[str, object]] = load_index()
literature_last_refresh_at = 0.0
literature_last_refresh_result: dict[str, Any] = {}

if literature_index and LITERATURE_STATE_PATH.exists():
    try:
        import json

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
    removed = removed_store is not None or bool(removed_events)
    if removed:
        audit_logger.log(
            event_type="session_expired",
            session_id=session_id,
            details={
                "ttl_seconds": SESSION_TTL_SECONDS,
                "dropped_timeline_events": len(removed_events),
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


def _add_timeline_events(session_id: str, source: str, text: str) -> int:
    existing = session_timeline_events.get(session_id, [])
    extracted = extract_events(text, source=source)
    if not extracted:
        return 0
    session_timeline_events[session_id] = merge_events(existing, extracted)
    return len(extracted)


def _get_timeline_events(session_id: str) -> list[ClinicalEvent]:
    _cleanup_expired_sessions()
    return session_timeline_events.get(session_id, [])


def _get_session_expiry(session_id: str) -> dict[str, Any]:
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
        return _ocr_image_with_fallback(path.read_bytes(), mime or "image/png", ocr_mode=ocr_mode)
    raise HTTPException(status_code=400, detail=f"不支持的文件类型: {path.name}")


def _extract_file_text_from_bytes(
    file_name: str,
    content_type: str | None,
    file_bytes: bytes,
    ocr_mode: str = "clinical",
) -> str:
    guessed_mime = mimetypes.guess_type(file_name)[0] or ""
    mime = content_type or guessed_mime or ""
    suffix = Path(file_name).suffix.lower()
    is_image = mime.startswith("image/") or suffix in IMAGE_SUFFIXES
    if mime == "application/octet-stream" and suffix in IMAGE_SUFFIXES:
        is_image = True
        mime = guessed_mime or "image/jpeg"

    if suffix == ".pdf" or mime == "application/pdf":
        return _extract_pdf_text_from_bytes(file_bytes, ocr_mode=ocr_mode)
    if mime.startswith("text/") or suffix in {".md", ".txt", ".csv"}:
        return _extract_raw_text_from_bytes(file_bytes)
    if is_image:
        return _ocr_image_with_fallback(file_bytes, mime or "image/png", ocr_mode=ocr_mode)
    raise HTTPException(status_code=400, detail=f"不支持的文件类型: {file_name}")


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
    m = re.search(r"\[第\s*(\d{1,4})\s*页\]", text)
    if m:
        try:
            return int(m.group(1))
        except Exception:
            return None
    m = re.search(r"\[(?:P|p)\s*(\d{1,4})\]", text)
    if m:
        try:
            return int(m.group(1))
        except Exception:
            return None
    return None


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
    reset: bool = True


class LiteratureSearchQuery(BaseModel):
    q: str
    top_k: int = 5


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
        "models": {
            "chat": CHAT_MODEL,
            "vision": VISION_MODEL,
            "embedding": EMBEDDING_MODEL,
        },
        "timeline_encoder": TIMELINE_ENCODER,
        "local_literature_agent_enabled": ENABLE_LOCAL_LITERATURE_AGENT,
        "local_literature_records": len(literature_index),
        "local_literature_last_refresh_at": int(literature_last_refresh_at) if literature_last_refresh_at else None,
        "local_literature_last_refresh_result": literature_last_refresh_result,
        "web_literature_enabled": ENABLE_WEB_LITERATURE,
        "literature_providers": LITERATURE_PROVIDERS,
        "literature_medical_oncology_only": LITERATURE_MEDICAL_ONCOLOGY_ONLY,
    }


@app.on_event("startup")
def bootstrap_literature_agent() -> None:
    if ENABLE_LOCAL_LITERATURE_AGENT and not literature_index:
        _refresh_local_literature_if_needed(force=True, max_results=LITERATURE_AGENT_BOOTSTRAP_MAX_RESULTS)


@app.post("/api/upload")
async def upload(request: Request, files: list[UploadFile] = File(...)) -> JSONResponse:
    session_id = _get_session_id(request)
    user_store = _get_or_create_session_store(session_id)

    results = []
    has_error = False
    for f in files:
        file_name = f.filename or "unnamed"
        try:
            file_bytes = await f.read()
            text = _extract_file_text_from_bytes(file_name, f.content_type, file_bytes)
            redaction = {}
            if ENABLE_UPLOAD_DEID:
                text, redaction_stats = redact_sensitive_info(text)
                redaction = redaction_stats.as_dict()
            chunks = split_text(text)
            added = user_store.add_texts(
                source=f"session/{file_name}",
                texts=chunks,
                embed_fn=lambda t: embed_text(client, EMBEDDING_MODEL, t),
            )
            extracted_events = _add_timeline_events(session_id, source=f"session/{file_name}", text=text)
            audit_logger.log(
                event_type="upload_processed",
                session_id=session_id,
                details={
                    "file": file_name,
                    "chunks": added,
                    "timeline_events": extracted_events,
                    "redaction": redaction,
                },
            )
            results.append(
                {
                    "file": file_name,
                    "chunks": added,
                    "ok": True,
                    "redaction": redaction,
                    "timeline_events": extracted_events,
                }
            )
        except HTTPException as exc:
            has_error = True
            results.append({"file": file_name, "ok": False, "error": exc.detail})
        except Exception as exc:
            has_error = True
            results.append({"file": file_name, "ok": False, "error": str(exc)})
    status_code = 207 if has_error else 200
    return JSONResponse(
        {
            "ok": not has_error,
            "results": results,
            "session_chunks": len(user_store.chunks),
            "timeline_event_count": len(_get_timeline_events(session_id)),
        },
        status_code=status_code,
    )


@app.post("/api/internal/import-guidelines")
def import_guidelines(payload: ImportGuidelinesRequest, request: Request) -> JSONResponse:
    _ensure_internal_access(request)
    if not GUIDELINES_DIR.exists():
        raise HTTPException(status_code=400, detail="data/guidelines 目录不存在")

    files = sorted([p for p in GUIDELINES_DIR.rglob("*") if p.is_file()])
    if not files:
        raise HTTPException(status_code=400, detail="data/guidelines 目录下没有文件")

    if payload.reset:
        internal_store.clear()

    results = []
    has_error = False
    for file_path in files:
        relative = str(file_path.relative_to(GUIDELINES_DIR)).replace("\\", "/")
        source = f"internal/{relative}"
        try:
            text = _extract_file_text(file_path, None, ocr_mode="guide")
            chunks = split_text(text)
            added = internal_store.add_texts(
                source=source,
                texts=chunks,
                embed_fn=lambda t: embed_text(client, EMBEDDING_MODEL, t),
            )
            results.append({"file": relative, "chunks": added, "ok": True})
        except HTTPException as exc:
            has_error = True
            results.append({"file": relative, "ok": False, "error": exc.detail})
        except Exception as exc:
            has_error = True
            results.append({"file": relative, "ok": False, "error": str(exc)})

    status_code = 207 if has_error else 200
    return JSONResponse(
        {
            "ok": not has_error,
            "results": results,
            "internal_chunks": len(internal_store.chunks),
            "reset": payload.reset,
        },
        status_code=status_code,
    )


@app.post("/api/chat")
def chat(payload: ChatRequest, request: Request) -> JSONResponse:
    if not payload.message.strip():
        raise HTTPException(status_code=400, detail="消息不能为空")

    session_id = _get_session_id(request)
    query_embedding = embed_text(client, EMBEDDING_MODEL, payload.message)
    cached_embed = lambda _text: query_embedding

    embed_cache: dict[str, list[float]] = {}

    def _embed_with_cache(text: str) -> list[float]:
        cached = embed_cache.get(text)
        if cached is not None:
            return cached
        vec = embed_text(client, EMBEDDING_MODEL, text)
        embed_cache[text] = vec
        return vec

    # Internal guideline chunks are always prioritized.
    internal_scored = internal_store.similarity_search_with_scores(
        payload.message,
        embed_fn=cached_embed,
        k=4,
    )
    session_scored: list[tuple[float, Any]] = []
    user_store = _get_session_store(session_id)
    timeline_events = _get_timeline_events(session_id)
    timeline_state = encode_timeline_state(timeline_events, encoder=TIMELINE_ENCODER)
    retrieval_hint = build_retrieval_hint(payload.message, timeline_events, state=timeline_state)
    timeline_summary = build_timeline_summary(timeline_events, max_items=8)
    if user_store is not None:
        session_scored = user_store.similarity_search_with_scores(
            payload.message,
            embed_fn=cached_embed,
            k=3,
        )

    all_scored = (internal_scored + session_scored)[:6]
    context_chunks = [item[1] for item in all_scored]
    context = "\n\n".join([f"[来源:{_format_source(c.source)}]\n{c.text}" for c in context_chunks])
    if not context:
        context = "暂无可用资料。"
    _refresh_local_literature_if_needed(force=False, max_results=LITERATURE_AGENT_BOOTSTRAP_MAX_RESULTS)
    external_literature = []
    if ENABLE_LOCAL_LITERATURE_AGENT:
        external_literature = search_from_local_store(
            payload.message,
            embed_fn=_embed_with_cache,
            top_k=LITERATURE_TOP_K,
            index=literature_index,
        )
    if not external_literature and ENABLE_WEB_LITERATURE:
        external_literature = [
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
    literature_context = literature_to_context(external_literature, max_items=LITERATURE_TOP_K)

    messages = [
        {
            "role": "system",
            "content": (
                "你是乳腺肿瘤方向的医学科普助手。"
                "请优先依据内部指南和提供资料，给出：1) 通俗解释；2) 可能分期与风险点；"
                "3) 下一步检查建议；4) 常见治疗路径（手术/化疗/靶向/免疫适用条件）。"
                "不能替代医生诊断，必须明确提示患者线下就医。"
                "输出请使用清晰的 Markdown 结构（短标题、列表、必要时表格），可少量使用 emoji 点缀，但要克制。"
                "当使用参考资料时，请在对应句子末尾添加引用标记，格式必须是[证据#序号]，"
                "其中序号对应参考资料的出现顺序（从1开始）。"
                "如使用外部学术文献，请使用[文献#序号]标注。"
                "当引用外部文献时，请明确写出文献类型（如临床试验/系统综述/指南/观察性研究）。"
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
    answer = resp.choices[0].message.content or "暂无回复"
    ranked_sources = []
    for rank, (score, chunk) in enumerate(all_scored, start=1):
        source_type = "internal" if chunk.source.startswith("internal/") else "session"
        item = {
            "rank": rank,
            "score": round(float(score), 4),
            "chunk_id": chunk.id,
            "source": _format_source(chunk.source),
            "source_raw": chunk.source,
            "type": source_type,
            "page": _extract_page_from_text(chunk.text),
            "text_length": len(chunk.text),
            "preview": chunk.text[:220],
            "evidence": chunk.text[:500],
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

    auto_rewrite_applied = False
    rewrite_triggered = False
    if AUTO_EVIDENCE_REWRITE:
        current_coverage = float(evidence_guard.get("coverage", 1.0) or 0.0)
        current_unsupported = int(evidence_guard.get("unsupported_claims", 0) or 0)
        rewrite_triggered = (
            current_coverage < EVIDENCE_REWRITE_MIN_COVERAGE
            or current_unsupported > EVIDENCE_REWRITE_MAX_UNSUPPORTED
        )
        if rewrite_triggered and ranked_sources:
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
            rewritten_answer = rewrite_resp.choices[0].message.content or answer
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
    audit_logger.log(
        event_type="chat_completed",
        session_id=session_id,
        details={
            "query_length": len(payload.message),
            "retrieved_internal": len(internal_scored),
            "retrieved_session": len(session_scored),
            "returned_sources": len(sources),
            "external_literature": len(external_literature),
            "timeline_events": len(timeline_events),
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
            "literature_guard": literature_guard,
            "auto_rewrite_applied": auto_rewrite_applied,
            "rewrite_triggered": rewrite_triggered,
        }
    )


@app.get("/api/session/timeline")
def session_timeline(request: Request) -> JSONResponse:
    session_id = _get_session_id(request)
    events = _get_timeline_events(session_id)
    expiry = _get_session_expiry(session_id)
    return JSONResponse(
        {
            "session_id": session_id,
            "event_count": len(events),
            "state": encode_timeline_state(events, encoder=TIMELINE_ENCODER),
            "encoder": TIMELINE_ENCODER,
            "summary": build_timeline_summary(events, max_items=10),
            "events": events_to_dict(events),
            **expiry,
        }
    )


@app.post("/api/session/expire-now")
def session_expire_now(request: Request) -> JSONResponse:
    session_id = _get_session_id(request)
    removed = _expire_session(session_id, reason="manual_demo")
    return JSONResponse({"ok": True, "session_id": session_id, "removed": removed})


@app.get("/api/literature/search")
def literature_search(q: str, top_k: int = 5) -> JSONResponse:
    if not q.strip():
        raise HTTPException(status_code=400, detail="q 不能为空")
    _refresh_local_literature_if_needed(force=False, max_results=LITERATURE_AGENT_BOOTSTRAP_MAX_RESULTS)
    max_k = min(max(top_k, 1), 5)
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
