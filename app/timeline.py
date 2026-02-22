from __future__ import annotations

import json
import logging
import re
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from openai import OpenAI

logger = logging.getLogger(__name__)


DATE_PATTERNS = [
    re.compile(r"(20\d{2})\s*[-/.年]\s*(\d{1,2})\s*[-/.月]\s*(\d{1,2})\s*[日号]?"),
    re.compile(r"(20\d{2})\s*[-/.]\s*(\d{1,2})"),
]
MONTH_DAY_PATTERN = re.compile(r"(?<!\d)(\d{1,2})\s*月\s*(\d{1,2})\s*[日号]?(?!\d)")
YEAR_PATTERN = re.compile(r"(20\d{2})年")
ISO_DATE_PATTERN = re.compile(r"(20\d{2})\s*-\s*(\d{1,2})\s*-\s*(\d{1,2})")
PLAIN_DATE_PATTERN = re.compile(r"(?<!\d)(20\d{2})(\d{2})(\d{2})(?!\d)")

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
BIRADS_PATTERN = re.compile(
    r"(?:BI\s*[-\s]?RADS|BIRADS|BI-RADS)\s*(?:分级|分类)?\s*[:：]?\s*([0-6])\s*([A-Ca-c]?)"
)


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
    plain = PLAIN_DATE_PATTERN.search(text)
    if plain:
        try:
            year = int(plain.group(1))
            month = int(plain.group(2))
            day = int(plain.group(3))
            if 1 <= month <= 12 and 1 <= day <= 31:
                return f"{year:04d}-{month:02d}-{day:02d}"
        except Exception:
            pass
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
    return "未知日期"


def _resolve_event_date(text: str, context_year: int, last_date: str | None) -> tuple[str, bool]:
    direct = _normalize_date(text)
    if direct:
        return direct, True
    md = MONTH_DAY_PATTERN.search(text)
    if md:
        return f"{context_year:04d}-{int(md.group(1)):02d}-{int(md.group(2)):02d}", True
    if last_date:
        return last_date, False
    return "未知日期", False


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


def _risk_from_birads(level: int, suffix: str) -> str:
    # Imaging risk mapping for timeline visualization.
    if level >= 5:
        return "high"
    if level == 4:
        if suffix.lower() in {"c"}:
            return "high"
        return "moderate"
    if level == 3:
        return "moderate"
    return "low"


def extract_events(text: str, source: str) -> list[ClinicalEvent]:
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    events: list[ClinicalEvent] = []
    context_year = _extract_context_year(text)
    last_date: str | None = None

    for line_idx, line in enumerate(lines):
        event_date, date_explicit = _resolve_event_date(
            line, context_year=context_year, last_date=last_date
        )
        if date_explicit:
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

        birads_match = BIRADS_PATTERN.search(line)
        if birads_match:
            level = int(birads_match.group(1))
            suffix = (birads_match.group(2) or "").upper()
            birads_value = f"{level}{suffix}" if suffix else str(level)
            events.append(
                ClinicalEvent(
                    event_id=_build_event_id(source, line_idx, "BI-RADS", event_date),
                    source=source,
                    date=event_date,
                    event_type="imaging",
                    metric="BI-RADS",
                    value=birads_value,
                    unit="-",
                    risk_tag=_risk_from_birads(level, suffix),
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


_LLM_EXTRACT_SYSTEM_PROMPT = """\
你是一个临床病历结构化抽取引擎。你的任务是从医疗文本中提取所有临床事件，输出为 JSON 数组。

## 输出格式
严格输出一个 JSON 数组，每个元素包含以下字段：
{
  "date": "YYYY-MM-DD 或 未知日期",
  "event_type": "lab | stage | imaging | treatment | pathology | observation",
  "metric": "指标名称（如 CEA、ER、HER2、BI-RADS、TNM、手术、化疗方案等）",
  "value": "指标值或事件描述（简洁）",
  "unit": "单位，无则填 -",
  "risk_tag": "high | moderate | low | info | unknown"
}

## 事件类型说明与 risk_tag 判断规则

### lab（实验室检验）
提取所有可量化的检验指标，包括但不限于：
- 肿瘤标志物：CEA(≥10=high,≥5=moderate)、CA19-9(≥100=high,≥37=moderate)、CA72-4(≥10=high,≥6=moderate)、CA125(≥100=high,≥35=moderate)、CA15-3(≥50=high,≥25=moderate)、AFP
- 血常规：WBC(≥11或≤3=moderate)、Hb/HGB(<90=high,<120=moderate)、PLT(≥400=moderate)、ANC/中性粒细胞
- 炎症指标：CRP(≥30=high,≥10=moderate)、ESR、PCT
- 肝肾功能：ALT、AST、TBIL、Cr、BUN
- 凝血指标：PT、APTT、D-dimer
- 其他检验值

### pathology（病理）
- ER受体状态：阳性(+)=info，阴性(-)=moderate
- PR受体状态：阳性(+)=info，阴性(-)=moderate
- HER2状态：3+=high, 2+=moderate（需FISH确认），0/1+=info
- Ki-67指数：>30%=high, >14%=moderate, ≤14%=low
- 分子分型：Luminal A=low, Luminal B=moderate, HER2过表达型=high, 三阴性/Basal-like=high
- 淋巴结转移：有转移=high, 无转移=low
- 切缘状态：阳性=high, 阴性=low
- 病理分级：G3=high, G2=moderate, G1=low

### stage（分期）
- TNM 分期：含M1=high，其余=moderate
- 临床分期：IV期=high, III期=high, II期=moderate, I期=low

### imaging（影像）
- BI-RADS分级：5-6=high, 4C=high, 4A/4B=moderate, 3=moderate, 0-2=low
- 影像发现：肿块大小、钙化、淋巴结肿大等

### treatment（治疗/诊疗事件）
- 手术（保乳术、全乳切除、前哨淋巴结活检、腋窝清扫等）=info
- 化疗方案（AC-T、EC-T、TC、TAC、CMF等）=info
- 放疗 =info
- 靶向治疗（曲妥珠单抗、帕妥珠单抗等）=info
- 内分泌治疗（他莫昔芬、来曲唑、阿那曲唑等）=info
- 免疫治疗 =info
- 复发 =high
- 转移（远处转移）=high
- 复查/随访 =info

### observation（临床观察）
- 症状描述、体征、一般情况 =info
- 不良反应/副作用 =moderate

## 重要规则
1. **否定语义**：「未见转移」「未发现复发」「排除恶性」等否定表述，不要生成对应的高危事件。可以生成一个 risk_tag=low 的事件，表示排除。
2. **日期推断**：尽量从上下文推断事件日期。如果文本中有"2025年3月15日"，后续事件沿用该日期直到出现新日期。
3. **不要编造**：只提取文本中明确存在的信息，不要推测或添加文本中没有的内容。
4. **值要简洁**：value 字段不超过 60 个字符。
5. **只输出 JSON 数组**，不要输出任何其他内容（不要 markdown 代码块标记，不要解释）。
"""


def extract_events_with_llm(
    client: "OpenAI",
    model: str,
    text: str,
    source: str,
) -> list[ClinicalEvent]:
    """Use LLM to extract structured clinical events from medical text.

    Falls back to regex-based extraction on any failure.
    """
    if not text or not text.strip():
        return []

    # Truncate very long texts to avoid token limits; keep first ~6000 chars
    truncated = text[:6000] if len(text) > 6000 else text

    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": _LLM_EXTRACT_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": f"请从以下医疗文本中提取所有临床事件：\n\n{truncated}",
                },
            ],
            temperature=0.05,
        )
        raw_output = (resp.choices[0].message.content or "").strip()
    except Exception as exc:
        logger.warning("LLM event extraction failed, falling back to regex: %s", exc)
        return extract_events(text, source=source)

    # Parse JSON output
    events: list[ClinicalEvent] = []
    try:
        # Handle possible markdown code blocks wrapping
        cleaned = raw_output
        if cleaned.startswith("```"):
            # Strip ```json ... ``` wrapper
            lines = cleaned.splitlines()
            start = 1 if lines[0].strip().startswith("```") else 0
            end = len(lines) - 1 if lines[-1].strip() == "```" else len(lines)
            cleaned = "\n".join(lines[start:end])
        parsed = json.loads(cleaned)
        if not isinstance(parsed, list):
            raise ValueError("LLM output is not a JSON array")
    except (json.JSONDecodeError, ValueError) as exc:
        logger.warning("LLM output JSON parse failed (%s), falling back to regex. Output: %s", exc, raw_output[:300])
        return extract_events(text, source=source)

    for idx, item in enumerate(parsed):
        if not isinstance(item, dict):
            continue
        date = str(item.get("date", "未知日期")).strip() or "未知日期"
        event_type = str(item.get("event_type", "observation")).strip()
        metric = str(item.get("metric", "UNKNOWN")).strip()
        value = str(item.get("value", "")).strip()[:60]
        unit = str(item.get("unit", "-")).strip() or "-"
        risk_tag = str(item.get("risk_tag", "unknown")).strip()

        # Validate risk_tag
        if risk_tag not in {"high", "moderate", "low", "info", "unknown"}:
            risk_tag = "unknown"
        # Validate event_type
        if event_type not in {"lab", "stage", "imaging", "treatment", "pathology", "observation"}:
            event_type = "observation"

        event_id = _build_event_id(source, idx, metric, date)
        events.append(
            ClinicalEvent(
                event_id=event_id,
                source=source,
                date=date,
                event_type=event_type,
                metric=metric,
                value=value,
                unit=unit,
                risk_tag=risk_tag,
                raw_line=value,
            )
        )

    if not events:
        logger.info("LLM returned empty events, falling back to regex.")
        return extract_events(text, source=source)

    events.sort(key=lambda item: (item.date, item.event_id))
    logger.info("LLM extracted %d events from source '%s'", len(events), source)
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
    if "分期" in question or "stage" in q or "tnm" in q or "ctnm" in q:
        return "优先召回病理分期、影像分期和TNM相关指南。"
    if "化疗" in question or "靶向" in question or "免疫" in question:
        return "优先召回治疗路径、适应证、禁忌证和不良反应管理。"
    if risk == "high":
        return "优先召回高风险复发评估与强化随访建议。"
    return "优先召回基础指南、随访策略和检查建议。"


def events_to_dict(events: list[ClinicalEvent]) -> list[dict[str, str]]:
    return [event.as_dict() for event in events]
