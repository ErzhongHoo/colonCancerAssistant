from __future__ import annotations

import mimetypes
import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, Response, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
from pypdf import PdfReader

from app.llm import build_client, embed_text, ocr_with_vision, ocr_with_vision_bytes
from app.rag import LocalVectorStore, split_text

try:
    import fitz  # type: ignore
except Exception:
    fitz = None

load_dotenv()

ROOT = Path(__file__).resolve().parent.parent
UPLOAD_DIR = ROOT / "data" / "uploads"
VECTOR_DB = ROOT / "data" / "vector_store.json"
STATIC_DIR = ROOT / "web"

CHAT_MODEL = os.getenv("CHAT_MODEL", "qwen-plus")
VISION_MODEL = os.getenv("VISION_MODEL", "qwen-vl-max")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-v3")

app = FastAPI(title="Colon Cancer RAG MVP")
client = build_client()
store = LocalVectorStore(VECTOR_DB)
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


def _extract_pdf_text(path: Path) -> str:
    reader = PdfReader(str(path))
    pages = [page.extract_text() or "" for page in reader.pages]
    text = "\n".join(pages).strip()
    if len(text) >= 50:
        return text
    fallback = _extract_pdf_text_with_ocr(path)
    return fallback or text


def _extract_pdf_text_with_ocr(path: Path) -> str:
    if fitz is None:
        return ""
    doc = fitz.open(str(path))
    extracted: list[str] = []
    max_pages = min(len(doc), 15)
    for i in range(max_pages):
        page = doc.load_page(i)
        page_text = _ocr_pdf_page_with_retry(page).strip()
        if page_text:
            extracted.append(f"[第{i + 1}页]\n{page_text}")
    doc.close()
    return "\n\n".join(extracted).strip()


def _ocr_pdf_page_with_retry(page: Any) -> str:
    # Avoid model payload size limit by progressively reducing size/quality.
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
                return ocr_with_vision_bytes(
                    client,
                    VISION_MODEL,
                    image_bytes,
                    mime_type="image/jpeg",
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


def _extract_file_text(path: Path, content_type: str | None) -> str:
    guessed_mime = mimetypes.guess_type(path.name)[0] or ""
    mime = content_type or guessed_mime or ""
    suffix = path.suffix.lower()
    is_image = mime.startswith("image/") or suffix in IMAGE_SUFFIXES
    # Some browsers upload images as application/octet-stream.
    if mime == "application/octet-stream" and suffix in IMAGE_SUFFIXES:
        is_image = True
        mime = guessed_mime or "image/jpeg"

    if path.suffix.lower() == ".pdf" or mime == "application/pdf":
        return _extract_pdf_text(path)
    if mime.startswith("text/") or suffix in {".md", ".txt", ".csv"}:
        return _extract_raw_text(path)
    if is_image:
        return ocr_with_vision(client, VISION_MODEL, path, mime or "image/png")
    raise HTTPException(status_code=400, detail=f"不支持的文件类型: {path.name}")


class ChatRequest(BaseModel):
    message: str
    history: list[dict[str, str]] = []


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/favicon.ico")
def favicon() -> Response:
    # Browser auto-requests this; return 204 to avoid noisy 404 logs.
    return Response(status_code=204)


@app.get("/api/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "chunks": len(store.chunks),
        "models": {
            "chat": CHAT_MODEL,
            "vision": VISION_MODEL,
            "embedding": EMBEDDING_MODEL,
        },
    }


@app.post("/api/upload")
async def upload(files: list[UploadFile] = File(...)) -> JSONResponse:
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    results = []
    has_error = False
    for f in files:
        file_name = f.filename or "unnamed"
        target = UPLOAD_DIR / file_name
        try:
            target.write_bytes(await f.read())
            text = _extract_file_text(target, f.content_type)
            chunks = split_text(text)
            added = store.add_texts(
                source=file_name,
                texts=chunks,
                embed_fn=lambda t: embed_text(client, EMBEDDING_MODEL, t),
            )
            results.append({"file": file_name, "chunks": added, "ok": True})
        except HTTPException as exc:
            has_error = True
            results.append({"file": file_name, "ok": False, "error": exc.detail})
        except Exception as exc:
            has_error = True
            results.append({"file": file_name, "ok": False, "error": str(exc)})
    status_code = 207 if has_error else 200
    return JSONResponse(
        {"ok": not has_error, "results": results, "total_chunks": len(store.chunks)},
        status_code=status_code,
    )


@app.post("/api/chat")
def chat(payload: ChatRequest) -> JSONResponse:
    if not payload.message.strip():
        raise HTTPException(status_code=400, detail="消息不能为空")
    context_chunks = store.similarity_search(
        payload.message,
        embed_fn=lambda t: embed_text(client, EMBEDDING_MODEL, t),
        k=5,
    )
    context = "\n\n".join(
        [f"[来源:{c.source}]\n{c.text}" for c in context_chunks]
    ) or "暂无病例资料。"
    messages = [
        {
            "role": "system",
            "content": (
                "你是结直肠肿瘤方向的医学科普助手。"
                "请基于提供的资料，给出：1) 通俗解释；2) 可能分期与风险点；"
                "3) 下一步检查建议；4) 常见治疗路径（手术/化疗/靶向/免疫适用条件）。"
                "不能替代医生诊断，必须明确提示患者线下就医。"
            ),
        },
        {"role": "system", "content": f"病例资料:\n{context}"},
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
    sources = [{"source": c.source, "preview": c.text[:100]} for c in context_chunks]
    return JSONResponse({"answer": answer, "sources": sources})
