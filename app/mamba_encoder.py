from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from app.timeline import ClinicalEvent


@dataclass
class MambaWeights:
    w_in: np.ndarray
    b_in: np.ndarray
    w_delta: np.ndarray
    b_delta: np.ndarray
    a_log: np.ndarray
    w_b: np.ndarray
    c: np.ndarray
    d: np.ndarray
    w_out: np.ndarray
    b_out: np.ndarray


_INPUT_DIM = 12
_HIDDEN_DIM = 32
_DEFAULT_SEED = 42


def _default_model_path() -> Path:
    raw = os.getenv("MAMBA_MODEL_PATH", "models/mamba_timeline_v1.npz").strip()
    p = Path(raw)
    if p.is_absolute():
        return p
    root = Path(__file__).resolve().parent.parent
    return root / p


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -50.0, 50.0)))


def _softplus(x: np.ndarray) -> np.ndarray:
    return np.log1p(np.exp(np.clip(x, -50.0, 50.0)))


def _build_bootstrap_weights(seed: int = _DEFAULT_SEED) -> MambaWeights:
    rng = np.random.default_rng(seed)
    return MambaWeights(
        w_in=rng.normal(0.0, 0.08, size=(_INPUT_DIM, _HIDDEN_DIM)).astype(np.float32),
        b_in=rng.normal(0.0, 0.02, size=(_HIDDEN_DIM,)).astype(np.float32),
        w_delta=rng.normal(0.0, 0.05, size=(_INPUT_DIM, _HIDDEN_DIM)).astype(np.float32),
        b_delta=np.full((_HIDDEN_DIM,), -1.5, dtype=np.float32),
        a_log=rng.normal(-0.1, 0.05, size=(_HIDDEN_DIM,)).astype(np.float32),
        w_b=rng.normal(0.0, 0.07, size=(_INPUT_DIM, _HIDDEN_DIM)).astype(np.float32),
        c=rng.normal(0.0, 0.2, size=(_HIDDEN_DIM,)).astype(np.float32),
        d=rng.normal(0.0, 0.1, size=(_INPUT_DIM,)).astype(np.float32),
        w_out=rng.normal(0.0, 0.18, size=(_HIDDEN_DIM + _INPUT_DIM + 3, 3)).astype(np.float32),
        b_out=np.array([-0.2, 0.0, -0.2], dtype=np.float32),
    )


def _save_weights(path: Path, w: MambaWeights) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        w_in=w.w_in,
        b_in=w.b_in,
        w_delta=w.w_delta,
        b_delta=w.b_delta,
        a_log=w.a_log,
        w_b=w.w_b,
        c=w.c,
        d=w.d,
        w_out=w.w_out,
        b_out=w.b_out,
    )


def _load_or_init_weights() -> tuple[MambaWeights, str]:
    model_path = _default_model_path()
    if model_path.exists():
        data = np.load(model_path)
        return (
            MambaWeights(
                w_in=data["w_in"].astype(np.float32),
                b_in=data["b_in"].astype(np.float32),
                w_delta=data["w_delta"].astype(np.float32),
                b_delta=data["b_delta"].astype(np.float32),
                a_log=data["a_log"].astype(np.float32),
                w_b=data["w_b"].astype(np.float32),
                c=data["c"].astype(np.float32),
                d=data["d"].astype(np.float32),
                w_out=data["w_out"].astype(np.float32),
                b_out=data["b_out"].astype(np.float32),
            ),
            "checkpoint",
        )

    weights = _build_bootstrap_weights()
    _save_weights(model_path, weights)
    return weights, "bootstrap"


def _risk_to_onehot(tag: str) -> tuple[float, float, float]:
    if tag == "high":
        return 1.0, 0.0, 0.0
    if tag == "moderate":
        return 0.0, 1.0, 0.0
    return 0.0, 0.0, 1.0


def _event_to_vec(event: ClinicalEvent) -> np.ndarray:
    # [lab, stage, treatment, cea, ca199, inflam, blood, surgery, chemo, relapse, value_norm, log_value]
    x = np.zeros((_INPUT_DIM,), dtype=np.float32)

    if event.event_type == "lab":
        x[0] = 1.0
    elif event.event_type == "stage":
        x[1] = 1.0
    elif event.event_type == "treatment":
        x[2] = 1.0

    metric = event.metric.upper()
    if "CEA" in metric:
        x[3] = 1.0
    if "19-9" in metric or "CA19" in metric:
        x[4] = 1.0
    if metric in {"CRP", "WBC"}:
        x[5] = 1.0
    if metric in {"HB", "HGB", "PLT"}:
        x[6] = 1.0

    raw = event.value
    if "手术" in raw or "切除" in raw:
        x[7] = 1.0
    if "化疗" in raw or "靶向" in raw or "免疫" in raw:
        x[8] = 1.0
    if "复发" in raw or "转移" in raw:
        x[9] = 1.0

    try:
        v = float(raw)
    except Exception:
        v = 0.0
    v = max(v, 0.0)
    x[10] = min(v / 120.0, 5.0)
    x[11] = np.log1p(v) / 8.0
    return x


def _selective_scan(events: list[ClinicalEvent], w: MambaWeights) -> tuple[np.ndarray, np.ndarray]:
    state = np.zeros((_HIDDEN_DIM,), dtype=np.float32)
    last_x = np.zeros((_INPUT_DIM,), dtype=np.float32)

    for ev in events:
        x = _event_to_vec(ev)
        last_x = x
        delta = _softplus(x @ w.w_delta + w.b_delta) + 1e-4
        a_bar = np.exp(-delta * np.exp(w.a_log))
        b_t = np.tanh(x @ w.w_b + w.b_in)
        u_t = np.tanh(x @ w.w_in + w.b_in)
        state = a_bar * state + (1.0 - a_bar) * (0.7 * b_t + 0.3 * u_t)

    return state, last_x


def encode_events_with_mamba(events: list[ClinicalEvent]) -> dict[str, float | str]:
    if not events:
        return {
            "risk_score": 0.0,
            "risk_level": "low",
            "high_risk_events": 0.0,
            "event_count": 0.0,
            "encoder_used": "mamba",
            "checkpoint": "n/a",
        }

    weights, checkpoint_status = _load_or_init_weights()
    state, last_x = _selective_scan(events, weights)

    high = sum(1 for ev in events if ev.risk_tag == "high")
    moderate = sum(1 for ev in events if ev.risk_tag == "moderate")
    low = sum(1 for ev in events if ev.risk_tag == "low")

    summary_vec = np.array(
        [
            float(high) / max(len(events), 1),
            float(moderate) / max(len(events), 1),
            float(low) / max(len(events), 1),
        ],
        dtype=np.float32,
    )
    features = np.concatenate([state, last_x, summary_vec], axis=0)
    logits = features @ weights.w_out + weights.b_out
    probs = _sigmoid(logits)

    # [high, moderate, low] -> risk score in [0, 5]
    risk_score = float(5.0 * (probs[0] * 0.95 + probs[1] * 0.55 + probs[2] * 0.2))
    risk_score = max(0.0, min(risk_score, 5.0))

    if risk_score >= 3.4:
        level = "high"
    elif risk_score >= 1.8:
        level = "moderate"
    else:
        level = "low"

    return {
        "risk_score": round(risk_score, 4),
        "risk_level": level,
        "high_risk_events": float(high),
        "event_count": float(len(events)),
        "encoder_used": "mamba",
        "checkpoint": checkpoint_status,
    }
