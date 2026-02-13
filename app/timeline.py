from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from datetime import datetime


DATE_PATTERNS = [
    re.compile(r"(20\d{2})[-/.年](\d{1,2})[-/.月](\d{1,2})[日号]?"),
    re.compile(r"(20\d{2})[-/.](\d{1,2})"),
]
MONTH_DAY_PATTERN = re.compile(r"(?<!\d)(\d{1,2})月(\d{1,2})[日号]?(?!\d)")
YEAR_PATTERN = re.compile(r"(20\d{2})年")
ISO_DATE_PATTERN = re.compile(r"(20\d{2})-(\d{1,2})-(\d{1,2})")

LAB_PATTERNS: dict[str, re.Pattern[str]] = {
    "CEA": re.compile(r"(?:CEA|癌胚抗原)\s*[:：]?\s*([0-9]+(?:\.[0-9]+)?)\s*([a-zA-Z/%\u00b5\-]*)"),
    "CA19-9": re.compile(r"(?:CA\s*19-?9|糖类抗原\s*19-?9)\s*[:：]?\s*([0-9]+(?:\.[0-9]+)?)\s*([a-zA-Z/%\u00b5\-]*)"),
    "CA72-4": re.compile(r"(?:CA\s*72-?4|糖类抗原\s*72-?4)\s*[:：]?\s*([0-9]+(?:\.[0-9]+)?)\s*([a-zA-Z/%\u00b5\-]*)"),
    "CA125": re.compile(r"(?:CA\s*125|糖类抗原\s*125)\s*[:：]?\s*([0-9]+(?:\.[0-9]+)?)\s*([a-zA-Z/%\u00b5\-]*)"),
    "CRP": re.compile(r"(?:CRP|C反应蛋白)\s*[:：]?\s*([0-9]+(?:\.[0-9]+)?)\s*([a-zA-Z/%\u00b5\-]*)"),
    "WBC": re.compile(r"(?:WBC|白细胞)\s*[:：]?\s*([0-9]+(?:\.[0-9]+)?)\s*([a-zA-Z/%\u00b5\-]*)"),
    "Hb": re.compile(r"(?:Hb|HGB|血红蛋白)\s*[:：]?\s*([0-9]+(?:\.[0-9]+)?)\s*([a-zA-Z/%\u00b5\-]*)"),
    "PLT": re.compile(r"(?:PLT|血小板)\s*[:：]?\s*([0-9]+(?:\.[0-9]+)?)\s*([a-zA-Z/%\u00b5\-]*)"),
}

TNM_PATTERN = re.compile(r"\b([cCpPyYrR]?[Tt][0-4][a-cA-C]?\s*[Nn][0-3][a-cA-C]?\s*[Mm][0-1][a-cA-C]?)\b")
STAGE_PATTERN = re.compile(r"(?:分期|stage)\s*[:：]?\s*(I{1,3}V?|[1-4])[A-Ca-c]?", re.IGNORECASE)


@dataclass
class ClinicalEvent:
    event_id: str
    source: str
    date: str
    event_type: str
    metric: str
    value: str
    unit: str
    risk_tag: str
    raw_line: str

    def as_dict(self) -> dict[str, str]:
        return asdict(self)


def _normalize_date(text: str) -> str | None:
    for pattern in DATE_PATTERNS:
        match = pattern.search(text)
        if not match:
            continue
        groups = match.groups()
        if len(groups) == 3:
            year, month, day = groups
            return f"{int(year):04d}-{int(month):02d}-{int(day):02d}"
        if len(groups) == 2:
            year, month = groups
            return f"{int(year):04d}-{int(month):02d}-01"
    return None


def _extract_context_year(text: str) -> int:
    match = YEAR_PATTERN.search(text)
    if match:
        try:
            return int(match.group(1))
        except Exception:
            pass
    iso = ISO_DATE_PATTERN.search(text)
    if iso:
        try:
            return int(iso.group(1))
        except Exception:
            pass
    return datetime.utcnow().year


def _normalize_date_with_context(text: str, context_year: int, last_date: str | None) -> str:
    direct = _normalize_date(text)
    if direct:
        return direct
    md = MONTH_DAY_PATTERN.search(text)
    if md:
        return f"{context_year:04d}-{int(md.group(1)):02d}-{int(md.group(2)):02d}"
    if last_date:
        return last_date
    return f"{context_year:04d}-01-01"


def _risk_from_metric(metric: str, value: str) -> str:
    try:
        v = float(value)
    except Exception:
        return "unknown"

    if metric == "CEA":
        if v >= 10:
            return "high"
        if v >= 5:
            return "moderate"
        return "low"
    if metric == "CA19-9":
        if v >= 100:
            return "high"
        if v >= 37:
            return "moderate"
        return "low"
    if metric == "CA72-4":
        if v >= 10:
            return "high"
        if v >= 6:
            return "moderate"
        return "low"
    if metric == "CA125":
        if v >= 100:
            return "high"
        if v >= 35:
            return "moderate"
        return "low"
    if metric == "CRP":
        if v >= 30:
            return "high"
        if v >= 10:
            return "moderate"
        return "low"
    if metric == "WBC":
        if v >= 11 or v <= 3:
            return "moderate"
        return "low"
    if metric == "Hb":
        if v < 90:
            return "high"
        if v < 120:
            return "moderate"
        return "low"
    if metric == "PLT":
        if v >= 400:
            return "moderate"
        return "low"
    return "unknown"


def _build_event_id(source: str, index: int, metric: str, date: str) -> str:
    normalized_source = source.replace("/", "-").replace(" ", "-")
    return f"ev-{normalized_source[:30]}-{date}-{metric}-{index}"


def extract_events(text: str, source: str) -> list[ClinicalEvent]:
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    events: list[ClinicalEvent] = []
    context_year = _extract_context_year(text)
    last_date: str | None = None

    for line_idx, line in enumerate(lines):
        event_date = _normalize_date_with_context(line, context_year=context_year, last_date=last_date)
        last_date = event_date

        for metric, pattern in LAB_PATTERNS.items():
            match = pattern.search(line)
            if not match:
                continue
            value = match.group(1)
            unit = (match.group(2) or "").strip() or "-"
            risk = _risk_from_metric(metric, value)
            events.append(
                ClinicalEvent(
                    event_id=_build_event_id(source, line_idx, metric, event_date),
                    source=source,
                    date=event_date,
                    event_type="lab",
                    metric=metric,
                    value=value,
                    unit=unit,
                    risk_tag=risk,
                    raw_line=line[:200],
                )
            )

        tnm_match = TNM_PATTERN.search(line)
        if tnm_match:
            tnm = re.sub(r"\s+", "", tnm_match.group(1).upper())
            events.append(
                ClinicalEvent(
                    event_id=_build_event_id(source, line_idx, "TNM", event_date),
                    source=source,
                    date=event_date,
                    event_type="stage",
                    metric="TNM",
                    value=tnm,
                    unit="-",
                    risk_tag="high" if "M1" in tnm else "moderate",
                    raw_line=line[:200],
                )
            )

        stage_match = STAGE_PATTERN.search(line)
        if stage_match:
            stage = stage_match.group(0)
            events.append(
                ClinicalEvent(
                    event_id=_build_event_id(source, line_idx, "STAGE", event_date),
                    source=source,
                    date=event_date,
                    event_type="stage",
                    metric="STAGE",
                    value=stage,
                    unit="-",
                    risk_tag="moderate",
                    raw_line=line[:200],
                )
            )

        if re.search(r"手术|切除|化疗|放疗|免疫|靶向|新辅助|辅助治疗|复查|随访|复发|转移", line):
            events.append(
                ClinicalEvent(
                    event_id=_build_event_id(source, line_idx, "TREATMENT", event_date),
                    source=source,
                    date=event_date,
                    event_type="treatment",
                    metric="TREATMENT",
                    value=line[:60],
                    unit="-",
                    risk_tag="info",
                    raw_line=line[:200],
                )
            )

    events.sort(key=lambda item: (item.date, item.event_id))
    return events


def merge_events(existing: list[ClinicalEvent], incoming: list[ClinicalEvent]) -> list[ClinicalEvent]:
    merged = {event.event_id: event for event in existing}
    for event in incoming:
        merged[event.event_id] = event
    result = list(merged.values())
    result.sort(key=lambda item: (item.date, item.event_id))
    return result


def linear_state_encode(events: list[ClinicalEvent]) -> dict[str, float | str]:
    state = 0.0
    high_risk_count = 0
    for event in events:
        if event.risk_tag == "high":
            state = state * 0.82 + 1.0
            high_risk_count += 1
        elif event.risk_tag == "moderate":
            state = state * 0.9 + 0.45
        elif event.risk_tag == "low":
            state = state * 0.92 + 0.12
        else:
            state = state * 0.95

    risk_level = "low"
    if state >= 2.8:
        risk_level = "high"
    elif state >= 1.2:
        risk_level = "moderate"

    return {
        "risk_score": round(state, 4),
        "risk_level": risk_level,
        "high_risk_events": float(high_risk_count),
        "event_count": float(len(events)),
        "encoder_used": "linear",
    }


def encode_timeline_state(events: list[ClinicalEvent], encoder: str = "linear") -> dict[str, float | str]:
    if encoder == "ssm":
        from app.ssm_encoder import encode_events_with_ssm

        state = encode_events_with_ssm(events)
        state["encoder_used"] = "ssm"
        return state
    if encoder == "mamba":
        from app.mamba_encoder import encode_events_with_mamba

        return encode_events_with_mamba(events)
    return linear_state_encode(events)


def build_timeline_summary(events: list[ClinicalEvent], max_items: int = 8) -> str:
    if not events:
        return "暂无结构化病程事件。"
    latest = events[-max_items:]
    lines = [
        f"- {event.date} | {event.event_type} | {event.metric}={event.value} {event.unit} | risk={event.risk_tag}"
        for event in latest
    ]
    return "\n".join(lines)


def build_retrieval_hint(
    question: str,
    events: list[ClinicalEvent],
    state: dict[str, float | str] | None = None,
) -> str:
    state = state or linear_state_encode(events)
    risk = state.get("risk_level", "low")
    q = question.lower()
    if "分期" in question or "stage" in q:
        return "优先召回病理分期、影像分期和TNM相关指南。"
    if "化疗" in question or "靶向" in question or "免疫" in question:
        return "优先召回治疗路径、适应证、禁忌证和不良反应管理。"
    if risk == "high":
        return "优先召回高风险复发评估与强化随访建议。"
    return "优先召回基础指南、随访策略和检查建议。"


def events_to_dict(events: list[ClinicalEvent]) -> list[dict[str, str]]:
    return [event.as_dict() for event in events]
