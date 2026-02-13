from __future__ import annotations

import math
from typing import Iterable

import numpy as np

from app.timeline import ClinicalEvent


_HIDDEN_DIM = 12
_INPUT_DIM = 9

# Fixed matrices for deterministic inference-only SSM encoder.
_A = np.diag([0.92, 0.9, 0.88, 0.86, 0.84, 0.9, 0.87, 0.85, 0.89, 0.83, 0.91, 0.88]).astype(np.float32)
_B = np.array(
    [
        [0.35, 0.16, 0.12, 0.08, 0.04, 0.18, 0.07, 0.10, 0.05],
        [0.08, 0.21, 0.20, 0.09, 0.04, 0.13, 0.05, 0.06, 0.03],
        [0.10, 0.19, 0.14, 0.05, 0.02, 0.12, 0.06, 0.07, 0.03],
        [0.07, 0.11, 0.13, 0.06, 0.03, 0.14, 0.04, 0.05, 0.02],
        [0.06, 0.10, 0.12, 0.07, 0.03, 0.13, 0.03, 0.04, 0.02],
        [0.11, 0.15, 0.10, 0.08, 0.04, 0.22, 0.05, 0.08, 0.04],
        [0.09, 0.12, 0.11, 0.06, 0.02, 0.15, 0.04, 0.06, 0.03],
        [0.05, 0.09, 0.10, 0.07, 0.03, 0.12, 0.03, 0.05, 0.02],
        [0.07, 0.08, 0.09, 0.08, 0.03, 0.11, 0.03, 0.05, 0.02],
        [0.05, 0.07, 0.08, 0.05, 0.02, 0.10, 0.02, 0.04, 0.02],
        [0.04, 0.06, 0.07, 0.05, 0.02, 0.09, 0.02, 0.04, 0.02],
        [0.06, 0.09, 0.08, 0.05, 0.03, 0.10, 0.03, 0.05, 0.02],
    ],
    dtype=np.float32,
)
_C = np.array([0.28, 0.22, 0.20, 0.17, 0.14, 0.19, 0.16, 0.15, 0.14, 0.13, 0.12, 0.11], dtype=np.float32)
_D = np.array([0.25, 0.16, 0.14, 0.06, 0.03, 0.09, 0.06, 0.08, 0.04], dtype=np.float32)


def _sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def _event_to_input(event: ClinicalEvent) -> np.ndarray:
    # [lab, stage, treatment, high, moderate, low, info, unknown, value_norm]
    v = np.zeros((_INPUT_DIM,), dtype=np.float32)

    if event.event_type == "lab":
        v[0] = 1.0
    elif event.event_type == "stage":
        v[1] = 1.0
    elif event.event_type == "treatment":
        v[2] = 1.0

    risk_idx = {"high": 3, "moderate": 4, "low": 5, "info": 6}.get(event.risk_tag, 7)
    v[risk_idx] = 1.0

    try:
        num = float(event.value)
    except Exception:
        num = 0.0
    v[8] = max(0.0, min(num / 100.0, 5.0))
    return v


def encode_events_with_ssm(events: Iterable[ClinicalEvent]) -> dict[str, float | str]:
    state = np.zeros((_HIDDEN_DIM,), dtype=np.float32)
    high_risk_events = 0.0
    count = 0.0

    for event in events:
        u = _event_to_input(event)
        state = np.tanh(_A @ state + _B @ u)
        if event.risk_tag == "high":
            high_risk_events += 1.0
        count += 1.0

    logits = float(_C @ state)
    if count > 0:
        logits += float((_D @ u) * 0.35)
    prob = _sigmoid(logits)
    risk_score = round(prob * 5.0, 4)

    if risk_score >= 3.4:
        risk_level = "high"
    elif risk_score >= 1.8:
        risk_level = "moderate"
    else:
        risk_level = "low"

    return {
        "risk_score": risk_score,
        "risk_level": risk_level,
        "high_risk_events": high_risk_events,
        "event_count": count,
    }
