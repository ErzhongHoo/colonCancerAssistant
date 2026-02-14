from __future__ import annotations

import base64
import json
import os
from pathlib import Path

from openai import OpenAI


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
    resp = client.embeddings.create(model=model, input=text)
    return list(resp.data[0].embedding)


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
