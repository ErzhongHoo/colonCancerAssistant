from __future__ import annotations

import re
from dataclasses import dataclass


REDACTION_TOKENS = ("<ID_CARD>", "<PHONE>", "<EMAIL>", "<BANK_CARD>", "<NAME>", "<ADDRESS>", "<MEDICAL_RECORD>")


@dataclass
class RedactionStats:
    id_card: int = 0
    phone: int = 0
    email: int = 0
    bank_card: int = 0
    medical_record: int = 0
    name_field: int = 0
    address_field: int = 0

    def as_dict(self) -> dict[str, int]:
        return {
            "id_card": self.id_card,
            "phone": self.phone,
            "email": self.email,
            "bank_card": self.bank_card,
            "medical_record": self.medical_record,
            "name_field": self.name_field,
            "address_field": self.address_field,
        }


def _replace_with_count(pattern: re.Pattern[str], text: str, token: str) -> tuple[str, int]:
    count = 0

    def _sub(_match: re.Match[str]) -> str:
        nonlocal count
        count += 1
        return token

    return pattern.sub(_sub, text), count


def _replace_match_with_count(
    pattern: re.Pattern[str],
    text: str,
    repl,
) -> tuple[str, int]:
    count = 0

    def _sub(match: re.Match[str]) -> str:
        nonlocal count
        count += 1
        return repl(match)

    return pattern.sub(_sub, text), count


def _is_valid_id_card(digits: str) -> bool:
    """Validate a Chinese mainland 18-digit ID card number.

    Checks: (1) valid province prefix (11-82), (2) plausible date in positions
    6-14, and (3) the standard MOD-11 weighted checksum.
    """
    clean = re.sub(r"\s", "", digits)
    if len(clean) != 18:
        return False
    province = int(clean[:2])
    # Valid province codes: 11-15, 21-23, 31-37, 41-46, 50-54, 61-65, 71, 81, 82
    _VALID_PROVINCES = {
        11, 12, 13, 14, 15,
        21, 22, 23,
        31, 32, 33, 34, 35, 36, 37,
        41, 42, 43, 44, 45, 46,
        50, 51, 52, 53, 54,
        61, 62, 63, 64, 65,
        71, 81, 82,
    }
    if province not in _VALID_PROVINCES:
        return False
    # Basic date plausibility check (YYYYMMDD at positions 6-14).
    year_str, month_str, day_str = clean[6:10], clean[10:12], clean[12:14]
    try:
        year, month, day = int(year_str), int(month_str), int(day_str)
    except ValueError:
        return False
    if not (1900 <= year <= 2100 and 1 <= month <= 12 and 1 <= day <= 31):
        return False
    # MOD-11 weighted checksum.
    weights = [7, 9, 10, 5, 8, 4, 2, 1, 6, 3, 7, 9, 10, 5, 8, 4, 2]
    check_chars = "10X98765432"
    try:
        total = sum(int(clean[i]) * weights[i] for i in range(17))
    except (ValueError, IndexError):
        return False
    expected = check_chars[total % 11]
    return clean[17].upper() == expected


def redact_sensitive_info(text: str) -> tuple[str, RedactionStats]:
    content = text
    # Normalize common OCR spacing variants first.
    content = re.sub(r"姓\s*名", "姓名", content)
    content = re.sub(r"患\s*者\s*姓\s*名", "患者姓名", content)
    content = re.sub(r"身\s*份\s*证\s*号?", "身份证号", content)
    content = re.sub(r"手\s*机\s*(号|号码)?", "手机号", content)
    stats = RedactionStats()

    # Chinese mainland ID card number.
    # Two-pass approach:
    # 1. Context-aware: if "身份证" keyword appears nearby, treat matching
    #    18-digit numbers as ID cards even if the checksum is invalid (OCR
    #    commonly introduces checksum errors in scanned documents).
    # 2. Standalone: validate via MOD-11 checksum to avoid false positives
    #    on medical codes, ultrasound numbers, etc.
    _ID_CARD_PAT = re.compile(
        r"(?<![A-Za-z\d])"          # not preceded by letter or digit
        r"(\d{6}\s*\d{8}\s*[\dXx]{4})"
        r"(?!\d)"                    # not followed by digit
    )
    # Context pattern: "身份证" keyword within 12 chars before the number.
    _ID_CARD_CONTEXT_PAT = re.compile(
        r"身份证[号码]*\s*[:：]?\s*"
        r"(?P<num>\d{6}\s*\d{8}\s*[\dXx]{4})"
        r"(?!\d)"
    )

    def _looks_like_id_card(digits: str) -> bool:
        """Structural check (province + date) without checksum."""
        clean = re.sub(r"\s", "", digits)
        if len(clean) != 18:
            return False
        province = int(clean[:2])
        _VALID_PROVINCES = {
            11, 12, 13, 14, 15, 21, 22, 23,
            31, 32, 33, 34, 35, 36, 37,
            41, 42, 43, 44, 45, 46,
            50, 51, 52, 53, 54,
            61, 62, 63, 64, 65,
            71, 81, 82,
        }
        if province not in _VALID_PROVINCES:
            return False
        try:
            year, month, day = int(clean[6:10]), int(clean[10:12]), int(clean[12:14])
        except ValueError:
            return False
        return 1900 <= year <= 2100 and 1 <= month <= 12 and 1 <= day <= 31

    # Pass 1: context-aware replacement (keyword nearby → always redact).
    content, _ctx_count = _replace_match_with_count(
        _ID_CARD_CONTEXT_PAT,
        content,
        lambda m: m.group(0)[: m.start("num") - m.start()] + "<ID_CARD>",
    )

    # Pass 2: standalone checksum-validated replacement.
    def _id_card_sub(match: re.Match[str]) -> str:
        candidate = match.group(1)
        if _is_valid_id_card(candidate):
            return "<ID_CARD>"
        return match.group(0)  # leave unchanged

    content, _standalone_count = _replace_match_with_count(
        _ID_CARD_PAT,
        content,
        _id_card_sub,
    )
    # Count only those that were actually replaced across both passes.
    stats.id_card = content.count("<ID_CARD>") - text.count("<ID_CARD>")

    # Mainland phone number.
    content, stats.phone = _replace_with_count(
        re.compile(r"(?<!\d)(1[3-9]\d[\s-]?\d{4}[\s-]?\d{4})(?!\d)"),
        content,
        "<PHONE>",
    )
    content, stats.email = _replace_with_count(
        re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"),
        content,
        "<EMAIL>",
    )
    # Common bank card range — skip numbers that structurally look like
    # ID cards (valid province prefix + plausible date) even if their
    # checksum didn't pass, so they aren't misclassified as bank cards.
    def _bank_card_sub(match: re.Match[str]) -> str:
        candidate = re.sub(r"\s", "", match.group(1))
        if len(candidate) == 18 and _looks_like_id_card(candidate):
            return match.group(0)  # skip — likely an ID card with bad checksum
        return "<BANK_CARD>"

    _pre_bank = content.count("<BANK_CARD>")
    content, _bank_raw = _replace_match_with_count(
        re.compile(r"(?<![A-Za-z\d])(\d{16,19})(?!\d)"),
        content,
        _bank_card_sub,
    )
    stats.bank_card = content.count("<BANK_CARD>") - _pre_bank
    content, stats.medical_record = _replace_match_with_count(
        re.compile(
            r"((?:患者编号|病案号|住院号|门诊号|病例号|就诊号|检查号|样本号|申请单号|条码号?|编号)\s*[:：]?\s*)([A-Za-z0-9-]{6,})"
        ),
        content,
        lambda m: f"{m.group(1)}<MEDICAL_RECORD>",
    )
    # Field style replacement, e.g. "姓名: 张三".
    # Covers: 姓名、患者姓名、病史叙述者、送检医师、主管医师、责任护士、联系人、
    #          主治医师、经治医师、报告医师、审核医师、签名 etc.
    _NAME_PREFIX = (
        r"(?:"
        r"(?:患者)?姓名"
        r"|病史叙述者"
        r"|送检医师"
        r"|主管医师"
        r"|主治医师"
        r"|经治医师"
        r"|报告医师"
        r"|审核医师"
        r"|责任护士"
        r"|联系人"
        r"|签名"
        r"|记录(?:医师|者)"
        r"|(?:医师|护士|医生)签名?"
        r")"
    )
    _NAME_LOOKAHEAD = (
        r"(?="
        r"(?:\s|$|[，。,.\n\r]"
        r"|患者编号|性别|年龄|科别|科室|病区|床号|标本|条码|检验|送检|采样|地址|住址"
        r"|职业|民族|婚姻|入院|记录|可靠|诊断|出生|联系|电话|身份证"
        r"|住院号|门诊号|病案号|费用|日期|时间|编号|工号"
        r"|现住|籍贯|主诉|现病史|既往史|家族史|个人史|月经史"
        r")"
        r")"
    )
    content, stats.name_field = _replace_match_with_count(
        re.compile(
            rf"({_NAME_PREFIX}\s*[:：]\s*)([\u4e00-\u9fa5·]{{2,8}})"
            rf"{_NAME_LOOKAHEAD}"
        ),
        content,
        lambda m: f"{m.group(1)}<NAME>",
    )
    content, stats.address_field = _replace_match_with_count(
        re.compile(r"(住址|地址)\s*[:：]\s*([^\n]+)"),
        content,
        lambda m: f"{m.group(1)}: <ADDRESS>",
    )

    ocr_line_records = 0
    ocr_line_names = 0
    redacted_lines: list[str] = []
    for raw_line in content.splitlines(keepends=True):
        line_ending = ""
        line = raw_line
        if raw_line.endswith("\r\n"):
            line = raw_line[:-2]
            line_ending = "\r\n"
        elif raw_line.endswith("\n") or raw_line.endswith("\r"):
            line = raw_line[:-1]
            line_ending = raw_line[-1]

        if re.search(r"(?:^|\s)(?:男|女)(?:\s|$)|(?:^|\s)\d{1,3}岁(?:\s|$)", line):
            line, record_count = _replace_match_with_count(
                re.compile(r"^(\s*)([A-Za-z0-9-]{6,})(?=\s+[\u4e00-\u9fa5·]{2,4}(?:\s+(?:男|女|\d{1,3}岁)))"),
                line,
                lambda m: f"{m.group(1)}<MEDICAL_RECORD>",
            )
            ocr_line_records += record_count
            line, name_count = _replace_match_with_count(
                re.compile(r"(^|\s)([\u4e00-\u9fa5·]{2,4})(?=\s+(?:男|女)(?:\s|$)|\s+\d{1,3}岁(?:\s|$))"),
                line,
                lambda m: f"{m.group(1)}<NAME>",
            )
            ocr_line_names += name_count

        redacted_lines.append(line + line_ending)

    if ocr_line_records or ocr_line_names:
        content = "".join(redacted_lines)
        stats.medical_record += ocr_line_records
        stats.name_field += ocr_line_names
    return content, stats


def extract_redaction_preview(redacted_text: str, max_lines: int = 6, max_chars: int = 360) -> list[str]:
    if not redacted_text.strip():
        return []
    previews: list[str] = []
    for raw in redacted_text.splitlines():
        line = raw.strip()
        if not line:
            continue
        if not any(token in line for token in REDACTION_TOKENS):
            continue
        clipped = line if len(line) <= max_chars else (line[: max_chars - 1] + "...")
        previews.append(clipped)
        if len(previews) >= max_lines:
            break
    return previews


def detect_sensitive_types(text: str) -> list[str]:
    if not text.strip():
        return []
    _out, stats = redact_sensitive_info(text)
    stat_map = stats.as_dict()
    return [k for k, v in stat_map.items() if int(v or 0) > 0]
