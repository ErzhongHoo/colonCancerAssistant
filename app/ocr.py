from __future__ import annotations

import io
import multiprocessing as mp
import os
from functools import lru_cache
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageOps


PADDLE_OCR_SUBPROCESS_TIMEOUT_SECONDS = max(
    float(os.getenv("PADDLE_OCR_SUBPROCESS_TIMEOUT_SECONDS", "45") or 45.0),
    5.0,
)


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


def _paddle_ocr_worker(
    image_bytes: bytes,
    lang: str,
    detailed: bool,
    queue: Any,
) -> None:
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        ocr = _build_paddle_ocr(lang=lang)
        if detailed:
            best_words: list[dict[str, Any]] = []
            best_text = ""
            best_score = -1.0
            for variant in _build_ocr_variants(image):
                img_np = np.array(variant)
                result = ocr.ocr(img_np, cls=True)
                words = _parse_paddle_result_detailed(result)
                text = "\n".join(item["text"] for item in words).strip()
                score = float(sum(len(item["text"]) for item in words))
                if score > best_score:
                    best_score = score
                    best_words = words
                    best_text = text
            queue.put({"ok": True, "text": best_text, "words": best_words})
            return

        img_np = np.array(image)
        result = ocr.ocr(img_np, cls=True)
        lines = _parse_paddle_result(result)
        queue.put({"ok": True, "text": "\n".join(lines).strip()})
    except Exception as exc:
        queue.put({"ok": False, "error": f"{type(exc).__name__}: {exc}"})


def _run_paddle_ocr_in_subprocess(
    image_bytes: bytes,
    lang: str = "ch",
    *,
    detailed: bool = False,
) -> dict[str, Any]:
    ctx = mp.get_context("spawn")
    queue = ctx.Queue(maxsize=1)
    proc = ctx.Process(
        target=_paddle_ocr_worker,
        args=(image_bytes, lang, detailed, queue),
        daemon=True,
    )
    proc.start()
    proc.join(PADDLE_OCR_SUBPROCESS_TIMEOUT_SECONDS)

    if proc.is_alive():
        proc.terminate()
        proc.join()
        raise TimeoutError(
            f"PaddleOCR 子进程超时（>{PADDLE_OCR_SUBPROCESS_TIMEOUT_SECONDS:.0f}s）"
        )

    payload: dict[str, Any] | None = None
    try:
        if not queue.empty():
            payload = queue.get_nowait()
    except Exception:
        payload = None
    finally:
        queue.close()
        queue.join_thread()

    if proc.exitcode != 0:
        raise RuntimeError(f"PaddleOCR 子进程异常退出（exitcode={proc.exitcode}）")
    if not isinstance(payload, dict):
        raise RuntimeError("PaddleOCR 子进程未返回结果")
    if not payload.get("ok"):
        raise RuntimeError(str(payload.get("error", "PaddleOCR 执行失败")))
    return payload


def ocr_with_paddle_bytes(image_bytes: bytes, lang: str = "ch") -> str:
    if not is_paddle_available():
        raise RuntimeError("PaddleOCR 未安装，请先安装 paddleocr/paddlepaddle。")
    payload = _run_paddle_ocr_in_subprocess(image_bytes, lang=lang, detailed=False)
    return str(payload.get("text", "") or "").strip()


def ocr_with_paddle_bytes_detailed(image_bytes: bytes, lang: str = "ch") -> dict[str, Any]:
    if not is_paddle_available():
        raise RuntimeError("PaddleOCR 未安装，请先安装 paddleocr/paddlepaddle。")
    payload = _run_paddle_ocr_in_subprocess(image_bytes, lang=lang, detailed=True)
    return {
        "text": str(payload.get("text", "") or "").strip(),
        "words": list(payload.get("words") or []),
    }


def draw_redaction_boxes(
    image_bytes: bytes,
    boxes: list[dict[str, Any]],
    max_side: int = 1600,
) -> bytes:
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    if max(image.size) > max_side:
        image.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
    scale_x = image.width / max(1, int(boxes[0].get("image_width", image.width))) if boxes else 1.0
    scale_y = image.height / max(1, int(boxes[0].get("image_height", image.height))) if boxes else 1.0
    draw = ImageDraw.Draw(image)
    for box in boxes:
        x1 = int(float(box.get("x1", 0)) * scale_x)
        y1 = int(float(box.get("y1", 0)) * scale_y)
        x2 = int(float(box.get("x2", 0)) * scale_x)
        y2 = int(float(box.get("y2", 0)) * scale_y)
        x1, x2 = sorted((max(0, x1), max(0, x2)))
        y1, y2 = sorted((max(0, y1), max(0, y2)))
        if x2 - x1 < 2 or y2 - y1 < 2:
            continue
        draw.rectangle([x1, y1, x2, y2], fill=(12, 12, 12))
    out = io.BytesIO()
    image.save(out, format="JPEG", quality=85, optimize=True)
    return out.getvalue()


def _build_ocr_variants(image: Image.Image) -> list[Image.Image]:
    base = image.convert("RGB")
    gray = ImageOps.grayscale(base)
    variants = [
        base,
        ImageEnhance.Contrast(base).enhance(1.4),
        ImageEnhance.Sharpness(base).enhance(1.6),
        ImageEnhance.Contrast(gray).enhance(1.8).convert("RGB"),
        gray.filter(ImageFilter.MedianFilter(size=3)).convert("RGB"),
    ]
    return variants


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


def _parse_paddle_result_detailed(result: Any) -> list[dict[str, Any]]:
    words: list[dict[str, Any]] = []
    if not result:
        return words
    candidates = result
    if isinstance(result, list) and len(result) == 1 and isinstance(result[0], list):
        candidates = result[0]
    for item in candidates:
        if not isinstance(item, (list, tuple)) or len(item) < 2:
            continue
        points = item[0]
        text_part = item[1]
        text = ""
        score = 0.0
        if isinstance(text_part, (list, tuple)) and text_part:
            text = str(text_part[0]).strip()
            if len(text_part) > 1:
                try:
                    score = float(text_part[1])
                except Exception:
                    score = 0.0
        elif isinstance(text_part, str):
            text = text_part.strip()
        if not text:
            continue
        x1, y1, x2, y2 = _bbox_from_points(points)
        words.append(
            {
                "text": text,
                "score": score,
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
            }
        )
    return words


def _bbox_from_points(points: Any) -> tuple[int, int, int, int]:
    xs: list[float] = []
    ys: list[float] = []
    if isinstance(points, (list, tuple)):
        for p in points:
            if isinstance(p, (list, tuple)) and len(p) >= 2:
                try:
                    xs.append(float(p[0]))
                    ys.append(float(p[1]))
                except Exception:
                    continue
    if not xs or not ys:
        return 0, 0, 0, 0
    return int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))
