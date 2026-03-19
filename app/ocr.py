from __future__ import annotations

import atexit
import io
import multiprocessing as mp
import os
import threading
import time
import warnings
from functools import lru_cache
from queue import Empty
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageOps


PADDLE_OCR_SUBPROCESS_TIMEOUT_SECONDS = max(
    float(os.getenv("PADDLE_OCR_SUBPROCESS_TIMEOUT_SECONDS", "45") or 45.0),
    5.0,
)

_PADDLE_OCR_CONTEXT = mp.get_context("spawn")
_PADDLE_OCR_WORKERS_LOCK = threading.Lock()
_PADDLE_OCR_WORKERS: dict[str, "_PaddleOCRWorker"] = {}


@lru_cache(maxsize=1)
def _build_paddle_ocr(lang: str = "ch") -> Any:
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=".*No ccache found.*",
            category=UserWarning,
        )
        from paddleocr import PaddleOCR  # type: ignore

    return PaddleOCR(use_angle_cls=True, lang=lang, show_log=False)


def is_paddle_available() -> bool:
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=".*No ccache found.*",
                category=UserWarning,
            )
            import paddleocr  # type: ignore  # noqa: F401

        return True
    except Exception:
        return False


def _run_paddle_ocr_task(
    image_bytes: bytes,
    lang: str,
    detailed: bool,
    ocr: Any,
) -> dict[str, Any]:
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
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
        return {"ok": True, "text": best_text, "words": best_words}

    img_np = np.array(image)
    result = ocr.ocr(img_np, cls=True)
    lines = _parse_paddle_result(result)
    return {"ok": True, "text": "\n".join(lines).strip()}


def _paddle_ocr_worker(
    image_bytes: bytes,
    lang: str,
    detailed: bool,
    queue: Any,
) -> None:
    """
    Compatibility shim for already-running parent processes that still spawn the
    previous one-shot worker entrypoint by name.
    """
    try:
        ocr = _build_paddle_ocr(lang=lang)
        queue.put(
            _run_paddle_ocr_task(
                image_bytes=image_bytes,
                lang=lang,
                detailed=detailed,
                ocr=ocr,
            )
        )
    except Exception as exc:
        queue.put({"ok": False, "error": f"{type(exc).__name__}: {exc}"})


def _paddle_ocr_worker_loop(lang: str, task_queue: Any, result_queue: Any) -> None:
    try:
        ocr = _build_paddle_ocr(lang=lang)
    except Exception as exc:
        result_queue.put({"request_id": None, "ok": False, "error": f"{type(exc).__name__}: {exc}"})
        return

    while True:
        task = task_queue.get()
        if not isinstance(task, dict):
            continue
        if task.get("cmd") == "shutdown":
            return
        request_id = task.get("request_id")
        try:
            payload = _run_paddle_ocr_task(
                image_bytes=bytes(task.get("image_bytes") or b""),
                lang=lang,
                detailed=bool(task.get("detailed")),
                ocr=ocr,
            )
            payload["request_id"] = request_id
            result_queue.put(payload)
        except Exception as exc:
            result_queue.put(
                {
                    "request_id": request_id,
                    "ok": False,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )


class _PaddleOCRWorker:
    def __init__(self, lang: str) -> None:
        self.lang = lang
        self.task_queue = _PADDLE_OCR_CONTEXT.Queue(maxsize=1)
        self.result_queue = _PADDLE_OCR_CONTEXT.Queue(maxsize=1)
        self.lock = threading.Lock()
        self.proc = _PADDLE_OCR_CONTEXT.Process(
            target=_paddle_ocr_worker_loop,
            args=(lang, self.task_queue, self.result_queue),
            daemon=True,
        )
        self.proc.start()

    def shutdown(self) -> None:
        try:
            if self.proc.is_alive():
                try:
                    self.task_queue.put_nowait({"cmd": "shutdown"})
                except Exception:
                    pass
                self.proc.join(timeout=1.0)
            if self.proc.is_alive():
                self.proc.terminate()
                self.proc.join(timeout=1.0)
        finally:
            try:
                self.task_queue.close()
                self.task_queue.join_thread()
            except Exception:
                pass
            try:
                self.result_queue.close()
                self.result_queue.join_thread()
            except Exception:
                pass


def _get_paddle_worker(lang: str) -> _PaddleOCRWorker:
    with _PADDLE_OCR_WORKERS_LOCK:
        worker = _PADDLE_OCR_WORKERS.get(lang)
        if worker is not None and worker.proc.is_alive():
            return worker
        if worker is not None:
            worker.shutdown()
        worker = _PaddleOCRWorker(lang=lang)
        _PADDLE_OCR_WORKERS[lang] = worker
        return worker


def _discard_paddle_worker(lang: str, worker: _PaddleOCRWorker) -> None:
    with _PADDLE_OCR_WORKERS_LOCK:
        current = _PADDLE_OCR_WORKERS.get(lang)
        if current is worker:
            _PADDLE_OCR_WORKERS.pop(lang, None)
    worker.shutdown()


def _shutdown_paddle_workers() -> None:
    with _PADDLE_OCR_WORKERS_LOCK:
        workers = list(_PADDLE_OCR_WORKERS.values())
        _PADDLE_OCR_WORKERS.clear()
    for worker in workers:
        worker.shutdown()


atexit.register(_shutdown_paddle_workers)


def _run_paddle_ocr_in_worker(
    image_bytes: bytes,
    lang: str = "ch",
    *,
    detailed: bool = False,
) -> dict[str, Any]:
    worker = _get_paddle_worker(lang=lang)
    request_id = f"{time.monotonic_ns()}:{threading.get_ident()}"
    timeout_at = time.monotonic() + PADDLE_OCR_SUBPROCESS_TIMEOUT_SECONDS

    with worker.lock:
        try:
            worker.task_queue.put(
                {
                    "request_id": request_id,
                    "image_bytes": image_bytes,
                    "detailed": detailed,
                },
                timeout=PADDLE_OCR_SUBPROCESS_TIMEOUT_SECONDS,
            )
        except Exception:
            _discard_paddle_worker(lang, worker)
            raise RuntimeError("PaddleOCR worker 请求投递失败")

        payload: dict[str, Any] | None = None
        while time.monotonic() < timeout_at:
            remaining = max(timeout_at - time.monotonic(), 0.0)
            wait_timeout = min(remaining, 0.5)
            try:
                candidate = worker.result_queue.get(timeout=wait_timeout)
            except Empty:
                if not worker.proc.is_alive():
                    _discard_paddle_worker(lang, worker)
                    raise RuntimeError("PaddleOCR worker 异常退出")
                continue
            if isinstance(candidate, dict) and candidate.get("request_id") == request_id:
                payload = candidate
                break
            if isinstance(candidate, dict) and candidate.get("request_id") is None:
                _discard_paddle_worker(lang, worker)
                raise RuntimeError(str(candidate.get("error", "PaddleOCR 初始化失败")))

        if payload is None:
            _discard_paddle_worker(lang, worker)
            raise TimeoutError(
                f"PaddleOCR worker 超时（>{PADDLE_OCR_SUBPROCESS_TIMEOUT_SECONDS:.0f}s）"
            )

    if not isinstance(payload, dict):
        raise RuntimeError("PaddleOCR worker 未返回结果")
    if not payload.get("ok"):
        raise RuntimeError(str(payload.get("error", "PaddleOCR 执行失败")))
    return payload


def ocr_with_paddle_bytes(image_bytes: bytes, lang: str = "ch") -> str:
    if not is_paddle_available():
        raise RuntimeError("PaddleOCR 未安装，请先安装 paddleocr/paddlepaddle。")
    payload = _run_paddle_ocr_in_worker(image_bytes, lang=lang, detailed=False)
    return str(payload.get("text", "") or "").strip()


def ocr_with_paddle_bytes_detailed(image_bytes: bytes, lang: str = "ch") -> dict[str, Any]:
    if not is_paddle_available():
        raise RuntimeError("PaddleOCR 未安装，请先安装 paddleocr/paddlepaddle。")
    payload = _run_paddle_ocr_in_worker(image_bytes, lang=lang, detailed=True)
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
