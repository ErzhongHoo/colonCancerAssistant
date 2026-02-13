from __future__ import annotations

import io
from functools import lru_cache
from typing import Any

import numpy as np


@lru_cache(maxsize=1)
def _build_paddle_ocr(lang: str = "ch") -> Any:
    from paddleocr import PaddleOCR  # type: ignore

    return PaddleOCR(use_angle_cls=True, lang=lang)


def is_paddle_available() -> bool:
    try:
        import paddleocr  # type: ignore  # noqa: F401

        return True
    except Exception:
        return False


def ocr_with_paddle_bytes(image_bytes: bytes, lang: str = "ch") -> str:
    if not is_paddle_available():
        raise RuntimeError("PaddleOCR 未安装，请先安装 paddleocr/paddlepaddle。")
    from PIL import Image  # type: ignore

    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img_np = np.array(image)
    ocr = _build_paddle_ocr(lang=lang)
    result = ocr.ocr(img_np, cls=True)
    lines = _parse_paddle_result(result)
    return "\n".join(lines).strip()


def _parse_paddle_result(result: Any) -> list[str]:
    lines: list[str] = []
    if not result:
        return lines

    # PaddleOCR usually returns [page_result], and page_result is list[item].
    candidates = result
    if isinstance(result, list) and len(result) == 1 and isinstance(result[0], list):
        candidates = result[0]

    for item in candidates:
        if not isinstance(item, (list, tuple)) or len(item) < 2:
            continue
        text_part = item[1]
        text = ""
        if isinstance(text_part, (list, tuple)) and text_part:
            text = str(text_part[0]).strip()
        elif isinstance(text_part, str):
            text = text_part.strip()
        if text:
            lines.append(text)
    return lines
