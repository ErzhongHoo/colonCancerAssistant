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

from app.llm import build_client, embed_text, ocr_with_vision
from app.rag import LocalVectorStore, split_text

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


def _extract_pdf_text(path: Path) -> str:
    reader = PdfReader(str(path))
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n".join(pages).strip()


def _extract_raw_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return path.read_text(encoding="gb18030", errors="ignore")


def _extract_file_text(path: Path, content_type: str | None) -> str:
    mime = content_type or mimetypes.guess_type(path.name)[0] or ""
    if path.suffix.lower() == ".pdf" or mime == "application/pdf":
        return _extract_pdf_text(path)
    if mime.startswith("text/") or path.suffix.lower() in {".md", ".txt", ".csv"}:
        return _extract_raw_text(path)
    if mime.startswith("image/") or path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}:
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
    for f in files:
        target = UPLOAD_DIR / f.filename
        target.write_bytes(await f.read())
        text = _extract_file_text(target, f.content_type)
        chunks = split_text(text)
        added = store.add_texts(
            source=f.filename,
            texts=chunks,
            embed_fn=lambda t: embed_text(client, EMBEDDING_MODEL, t),
        )
        results.append({"file": f.filename, "chunks": added})
    return JSONResponse({"ok": True, "results": results, "total_chunks": len(store.chunks)})


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
