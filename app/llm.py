from __future__ import annotations

import base64
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


def ocr_with_vision(client: OpenAI, model: str, image_path: Path, mime_type: str) -> str:
    encoded = base64.b64encode(image_path.read_bytes()).decode("utf-8")
    data_url = f"data:{mime_type};base64,{encoded}"
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
                        "text": "请提取图片中的病例/化验/检查信息。按“检查项-结果-参考范围-异常提示”输出。",
                    },
                    {"type": "image_url", "image_url": {"url": data_url}},
                ],
            },
        ],
        temperature=0.1,
    )
    return resp.choices[0].message.content or ""
