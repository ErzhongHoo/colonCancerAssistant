from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass
class RedactionStats:
    id_card: int = 0
    phone: int = 0
    email: int = 0
    bank_card: int = 0
    name_field: int = 0
    address_field: int = 0

    def as_dict(self) -> dict[str, int]:
        return {
            "id_card": self.id_card,
            "phone": self.phone,
            "email": self.email,
            "bank_card": self.bank_card,
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


def redact_sensitive_info(text: str) -> tuple[str, RedactionStats]:
    content = text
    # Normalize common OCR spacing variants first.
    content = re.sub(r"姓\s*名", "姓名", content)
    content = re.sub(r"患\s*者\s*姓\s*名", "患者姓名", content)
    content = re.sub(r"身\s*份\s*证\s*号?", "身份证号", content)
    content = re.sub(r"手\s*机\s*(号|号码)?", "手机号", content)
    stats = RedactionStats()

    # Chinese mainland ID card number.
    content, stats.id_card = _replace_with_count(
        re.compile(r"(?<!\d)(\d{6}\s*\d{8}\s*[\dXx]{4})(?!\d)"),
        content,
        "<ID_CARD>",
    )
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
    # Common bank card range.
    content, stats.bank_card = _replace_with_count(
        re.compile(r"(?<!\d)(\d{16,19})(?!\d)"),
        content,
        "<BANK_CARD>",
    )
    # Field style replacement, e.g. "姓名: 张三".
    content, stats.name_field = _replace_match_with_count(
        re.compile(
            r"((?:患者)?姓名\s*[:：]\s*)([\u4e00-\u9fa5·]{2,8})"
            r"(?=(?:\s|$|患者编号|性别|年龄|科别|病区|床号|标本|条码|检验|送检|采样|地址|住址))"
        ),
        content,
        lambda m: f"{m.group(1)}<NAME>",
    )
    content, stats.address_field = _replace_match_with_count(
        re.compile(r"(住址|地址)\s*[:：]\s*([^\n]+)"),
        content,
        lambda m: f"{m.group(1)}: <ADDRESS>",
    )
    return content, stats
