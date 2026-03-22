from __future__ import annotations

import base64
import json
import os
import re
import threading
from pathlib import Path

from openai import OpenAI


_LOCAL_EMBEDDER = None
_LOCAL_EMBEDDER_KEY = ""
_LOCAL_EMBEDDER_LOCK = threading.Lock()


def get_embedding_provider() -> str:
    raw = os.getenv("EMBEDDING_PROVIDER", "remote").strip().lower()
    if raw in {"local", "sentence-transformers", "sentence_transformers", "st"}:
        return "local"
    return "remote"


def _pick_api_key() -> str:
    key = (
        os.getenv("DASHSCOPE_API_KEY")
        or os.getenv("BAILIAN_API_KEY")
        or os.getenv("APIKEY")
        or os.getenv("apikey")
        or os.getenv("OPENAI_API_KEY")
    )
    if not key:
        raise RuntimeError(
            "缺少 API Key。请在 .env 中设置 DASHSCOPE_API_KEY/BAILIAN_API_KEY/OPENAI_API_KEY。"
        )
    return key


def build_client() -> OpenAI:
    base_url = os.getenv("BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
    return OpenAI(api_key=_pick_api_key(), base_url=base_url)


def embed_text(client: OpenAI, model: str, text: str) -> list[float]:
    if get_embedding_provider() == "local":
        embedder = _get_local_embedder()
        normalize = os.getenv("LOCAL_EMBEDDING_NORMALIZE", "true").strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        vector = embedder.encode(
            text,
            convert_to_numpy=True,
            normalize_embeddings=normalize,
            show_progress_bar=False,
        )
        return vector.tolist()
    resp = client.embeddings.create(model=model, input=text)
    return list(resp.data[0].embedding)


def _get_local_embedder():
    global _LOCAL_EMBEDDER, _LOCAL_EMBEDDER_KEY
    model_name = os.getenv("LOCAL_EMBEDDING_MODEL", "BAAI/bge-small-zh-v1.5").strip()
    device = os.getenv("LOCAL_EMBEDDING_DEVICE", "cpu").strip() or "cpu"
    cache_dir = os.getenv("LOCAL_EMBEDDING_CACHE_DIR", "").strip()
    local_files_only = os.getenv("LOCAL_EMBEDDING_LOCAL_ONLY", "false").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    trust_remote_code = os.getenv("LOCAL_EMBEDDING_TRUST_REMOTE_CODE", "false").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    cache_key = (
        f"{model_name}::{device}::{cache_dir}::{int(local_files_only)}::{int(trust_remote_code)}"
    )
    if _LOCAL_EMBEDDER is not None and _LOCAL_EMBEDDER_KEY == cache_key:
        return _LOCAL_EMBEDDER

    with _LOCAL_EMBEDDER_LOCK:
        if _LOCAL_EMBEDDER is not None and _LOCAL_EMBEDDER_KEY == cache_key:
            return _LOCAL_EMBEDDER
        try:
            from sentence_transformers import SentenceTransformer
        except Exception as exc:
            raise RuntimeError(
                "EMBEDDING_PROVIDER=local 但未安装 sentence-transformers。"
                "请执行: uv pip install sentence-transformers 或 pip install sentence-transformers"
            ) from exc

        kwargs: dict[str, object] = {
            "device": device,
            "local_files_only": local_files_only,
            "trust_remote_code": trust_remote_code,
        }
        if cache_dir:
            kwargs["cache_folder"] = cache_dir
        try:
            _LOCAL_EMBEDDER = SentenceTransformer(model_name, **kwargs)
        except Exception as exc:
            if local_files_only:
                raise RuntimeError(
                    "本地 embedding 模型未找到。请确认 LOCAL_EMBEDDING_MODEL 指向本地目录，"
                    "或先把 Hugging Face 模型完整下载到本机。"
                ) from exc
            raise RuntimeError(
                "本地 embedding 模型加载失败。当前配置会尝试从 Hugging Face 拉取模型；"
                "如果当前环境无法联网，请把 LOCAL_EMBEDDING_MODEL 改成本地目录，"
                "并设置 LOCAL_EMBEDDING_LOCAL_ONLY=true。"
            ) from exc
        _LOCAL_EMBEDDER_KEY = cache_key
        return _LOCAL_EMBEDDER


def _vision_extract(client: OpenAI, model: str, data_url: str, prompt_text: str) -> str:
    if len(data_url) > 19_000_000:
        raise ValueError("图片编码后体积过大，超过模型接口限制，请降低分辨率后重试。")
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": "你是医疗文档提取助手。只做信息提取，不要编造。",
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt_text,
                    },
                    {"type": "image_url", "image_url": {"url": data_url}},
                ],
            },
        ],
        temperature=0.1,
    )
    return resp.choices[0].message.content or ""


def ocr_with_vision(client: OpenAI, model: str, image_path: Path, mime_type: str) -> str:
    encoded = base64.b64encode(image_path.read_bytes()).decode("utf-8")
    data_url = f"data:{mime_type};base64,{encoded}"
    return _vision_extract(
        client,
        model,
        data_url,
        "请提取图片中的病例/化验/检查信息。按“检查项-结果-参考范围-异常提示”输出。",
    )


def ocr_with_vision_bytes(
    client: OpenAI, model: str, image_bytes: bytes, mime_type: str = "image/png"
) -> str:
    encoded = base64.b64encode(image_bytes).decode("utf-8")
    data_url = f"data:{mime_type};base64,{encoded}"
    return _vision_extract(
        client,
        model,
        data_url,
        (
            "请逐行转写图片中的全部可见中文/英文文字，必须保留日期、时间、页眉、页脚、单位和编号；"
            "不要总结，不要改写，不要补充解释。"
        ),
    )


def ocr_with_vision_bytes_transcribe(
    client: OpenAI, model: str, image_bytes: bytes, mime_type: str = "image/png"
) -> str:
    encoded = base64.b64encode(image_bytes).decode("utf-8")
    data_url = f"data:{mime_type};base64,{encoded}"
    return _vision_extract(
        client,
        model,
        data_url,
        "请逐行转写图片中的全部可见中文/英文文字，保持原文含义，不要总结，不要改写，不要补充解释。",
    )


def ocr_with_aliyun_ocr_bytes(
    client: OpenAI,
    model: str,
    image_bytes: bytes,
    mime_type: str = "image/png",
    prompt_text: str | None = None,
    min_pixels: int = 3072,
    max_pixels: int = 8_388_608,
) -> str:
    encoded = base64.b64encode(image_bytes).decode("utf-8")
    data_url = f"data:{mime_type};base64,{encoded}"
    def _build_content(text_prompt: str | None) -> list[dict[str, object]]:
        image_item: dict[str, object] = {
            "type": "image_url",
            "image_url": {"url": data_url},
        }
        if min_pixels > 0:
            image_item["min_pixels"] = int(min_pixels)
        if max_pixels > 0:
            image_item["max_pixels"] = int(max_pixels)
        content: list[dict[str, object]] = [image_item]
        if text_prompt and text_prompt.strip():
            content.append({"type": "text", "text": text_prompt.strip()})
        return content

    def _call(text_prompt: str | None) -> str:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": _build_content(text_prompt)}],
            temperature=0.1,
        )
        return _normalize_aliyun_ocr_text(resp.choices[0].message.content or "")

    first = _call(prompt_text)
    if not _is_coord_only_aliyun_output(first):
        return first
    retry_prompt = "请输出OCR文本内容。"
    second = _call(retry_prompt)
    if not _is_coord_only_aliyun_output(second):
        return second
    return first


def ocr_with_aliyun_ocr_bytes_detailed(
    client: OpenAI,
    model: str,
    image_bytes: bytes,
    mime_type: str = "image/png",
    min_pixels: int = 3072,
    max_pixels: int = 8_388_608,
) -> dict[str, object]:
    encoded = base64.b64encode(image_bytes).decode("utf-8")
    data_url = f"data:{mime_type};base64,{encoded}"
    image_item: dict[str, object] = {
        "type": "image_url",
        "image_url": {"url": data_url},
    }
    if min_pixels > 0:
        image_item["min_pixels"] = int(min_pixels)
    if max_pixels > 0:
        image_item["max_pixels"] = int(max_pixels)

    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": [image_item]}],
        temperature=0.1,
    )
    raw = str(resp.choices[0].message.content or "").strip()
    words = _parse_aliyun_ocr_detailed(raw)
    text = "\n".join(item["text"] for item in words if str(item.get("text", "")).strip()).strip()
    return {"text": text, "words": words, "raw": raw}


_COORD_NUM_RE = re.compile(r"^-?\d+(\.\d+)?$")


def _normalize_aliyun_ocr_text(raw: str) -> str:
    text = raw.strip()
    if not text:
        return ""
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    cleaned: list[str] = []
    hit_count = 0
    for line in lines:
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 6:
            continue
        if all(_COORD_NUM_RE.match(part) for part in parts[:5]):
            value = ",".join(parts[5:]).strip()
            if value:
                cleaned.append(value)
                hit_count += 1
    if hit_count > 0 and cleaned:
        return "\n".join(cleaned).strip()
    return text


def _is_coord_only_aliyun_output(text: str) -> bool:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return False
    coord_lines = 0
    for line in lines:
        parts = [part.strip() for part in line.split(",")]
        if len(parts) == 5 and all(_COORD_NUM_RE.match(part) for part in parts):
            coord_lines += 1
    return coord_lines == len(lines)


def _parse_aliyun_ocr_detailed(raw: str) -> list[dict[str, object]]:
    lines = [line.strip() for line in str(raw or "").splitlines() if line.strip()]
    words: list[dict[str, object]] = []
    for line in lines:
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 6:
            continue
        numeric = parts[:5]
        if not all(_COORD_NUM_RE.match(part) for part in numeric):
            continue
        text = ",".join(parts[5:]).strip()
        if not text:
            continue
        try:
            x1 = int(float(parts[0]))
            y1 = int(float(parts[1]))
            x2 = int(float(parts[2]))
            y2 = int(float(parts[3]))
        except Exception:
            continue
        x1, x2 = sorted((x1, x2))
        y1, y2 = sorted((y1, y2))
        words.append(
            {
                "text": text,
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
            }
        )
    return words


def extract_date_with_vision_bytes(
    client: OpenAI, model: str, image_bytes: bytes, mime_type: str = "image/png"
) -> str:
    encoded = base64.b64encode(image_bytes).decode("utf-8")
    data_url = f"data:{mime_type};base64,{encoded}"
    return _vision_extract(
        client,
        model,
        data_url,
        (
            "请在图片中寻找“检查日期/报告日期/采样日期/就诊日期”。"
            "只输出一个日期，格式必须为YYYY-MM-DD；如果找不到输出NA。"
        ),
    ).strip()


def extract_date_fields_with_vision_bytes(
    client: OpenAI, model: str, image_bytes: bytes, mime_type: str = "image/png"
) -> dict[str, str]:
    encoded = base64.b64encode(image_bytes).decode("utf-8")
    data_url = f"data:{mime_type};base64,{encoded}"
    raw = _vision_extract(
        client,
        model,
        data_url,
        (
            "请识别图片中的日期字段，并严格返回JSON对象，字段固定为："
            "{\"exam_date\":\"YYYY-MM-DD或NA\",\"submit_date\":\"YYYY-MM-DD或NA\",\"report_date\":\"YYYY-MM-DD或NA\"}。"
            "exam_date=检查日期；submit_date=送检/采样日期；report_date=报告日期。"
            "只输出JSON，不要输出其他内容。"
        ),
    ).strip()
    try:
        parsed = json.loads(raw)
    except Exception:
        return {"exam_date": "NA", "submit_date": "NA", "report_date": "NA"}
    out = {}
    for key in ("exam_date", "submit_date", "report_date"):
        value = str(parsed.get(key, "NA")).strip()
        out[key] = value if value else "NA"
    return out
