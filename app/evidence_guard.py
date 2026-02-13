from __future__ import annotations

import math
import re
from dataclasses import asdict, dataclass
from typing import Callable


MEDICAL_CLAIM_KEYWORDS = (
    "建议",
    "分期",
    "风险",
    "化疗",
    "靶向",
    "免疫",
    "手术",
    "转移",
    "复发",
    "指标",
)


@dataclass
class ClaimCheck:
    claim: str
    cited_ranks: list[int]
    supported: bool
    reason: str
    support_score: float
    sufficiency: str

    def as_dict(self) -> dict[str, str | list[int] | bool]:
        return asdict(self)


@dataclass
class EvidenceGuardReport:
    coverage: float
    verified_claims: int
    unsupported_claims: int
    checks: list[ClaimCheck]

    def as_dict(self) -> dict[str, object]:
        return {
            "coverage": self.coverage,
            "verified_claims": self.verified_claims,
            "unsupported_claims": self.unsupported_claims,
            "checks": [item.as_dict() for item in self.checks],
        }


def _split_claims(answer: str) -> list[str]:
    parts = re.split(r"[\n。；;!?！？]+", answer)
    claims = []
    for part in parts:
        line = part.strip()
        if len(line) < 8:
            continue
        if line.startswith("#"):
            continue
        # Ignore markdown table separators and generic table rows.
        if line.startswith("|") and line.count("|") >= 2:
            if re.search(r"\|\s*:?-{3,}:?\s*\|", line):
                continue
            # Most table rows are formatting-heavy and produce noisy false negatives.
            continue
        if any(k in line for k in MEDICAL_CLAIM_KEYWORDS):
            claims.append(line)
    return claims


def _extract_cited_ranks(text: str) -> list[int]:
    cited: list[int] = []
    for match in re.finditer(r"\[证据#(\d+)\]", text):
        try:
            cited.append(int(match.group(1)))
        except Exception:
            continue
    deduped = sorted(set(cited))
    return deduped


def _extract_cited_pages(text: str) -> list[int]:
    pages: list[int] = []
    for match in re.finditer(r"\[[^\]]*?[Pp]\.?\s*(\d{1,4})\]", text):
        try:
            pages.append(int(match.group(1)))
        except Exception:
            continue
    for match in re.finditer(r"第\s*(\d{1,4})\s*页", text):
        try:
            pages.append(int(match.group(1)))
        except Exception:
            continue
    return sorted(set(pages))


def _has_any_bracket_citation(text: str) -> bool:
    return bool(re.search(r"\[[^\]]+\]", text))


def _token_overlap(claim: str, evidence: str) -> float:
    claim_text = re.sub(r"\[证据#\d+\]", "", claim)
    evidence_text = evidence

    claim_tokens = set(re.findall(r"[A-Za-z0-9]{2,}", claim_text))
    evidence_tokens = set(re.findall(r"[A-Za-z0-9]{2,}", evidence_text))
    claim_cn = re.sub(r"[^\u4e00-\u9fa5]", "", claim_text)
    evidence_cn = re.sub(r"[^\u4e00-\u9fa5]", "", evidence_text)

    def _ngrams(text: str, n: int = 2) -> set[str]:
        if len(text) < n:
            return {text} if text else set()
        return {text[i : i + n] for i in range(len(text) - n + 1)}

    claim_grams = _ngrams(claim_cn, 2) | _ngrams(claim_cn, 3)
    evidence_grams = _ngrams(evidence_cn, 2) | _ngrams(evidence_cn, 3)

    total_claim_units = claim_tokens | claim_grams
    total_evidence_units = evidence_tokens | evidence_grams
    if not total_claim_units or not total_evidence_units:
        return 0.0
    return len(total_claim_units & total_evidence_units) / max(len(total_claim_units), 1)


def _cosine_similarity(v1: list[float], v2: list[float]) -> float:
    if not v1 or not v2 or len(v1) != len(v2):
        return 0.0
    dot = 0.0
    n1 = 0.0
    n2 = 0.0
    for a, b in zip(v1, v2):
        dot += a * b
        n1 += a * a
        n2 += b * b
    denom = math.sqrt(n1) * math.sqrt(n2)
    if denom <= 1e-12:
        return 0.0
    return dot / denom


def verify_answer_with_evidence(
    answer: str,
    ranked_sources: list[dict[str, object]],
    embed_similarity_fn: Callable[[str], list[float]] | None = None,
) -> EvidenceGuardReport:
    claim_lines = _split_claims(answer)
    rank_to_evidence = {
        int(item.get("rank", -1)): str(item.get("evidence", ""))
        for item in ranked_sources
        if isinstance(item.get("rank"), int) or str(item.get("rank", "")).isdigit()
    }
    page_to_ranks: dict[int, list[int]] = {}
    for rank, evidence in rank_to_evidence.items():
        pages = _extract_cited_pages(evidence)
        for page in pages:
            page_to_ranks.setdefault(page, []).append(rank)
    available_ranks = sorted(rank_to_evidence.keys())

    checks: list[ClaimCheck] = []
    verified = 0
    for claim in claim_lines:
        rank_cites = _extract_cited_ranks(claim)
        page_cites = _extract_cited_pages(claim)

        resolved_ranks: list[int] = list(rank_cites)
        for page in page_cites:
            resolved_ranks.extend(page_to_ranks.get(page, []))
        resolved_ranks = sorted(set(resolved_ranks))

        if not resolved_ranks:
            if rank_cites:
                checks.append(
                    ClaimCheck(
                        claim=claim,
                        cited_ranks=rank_cites,
                        supported=False,
                        reason=f"引用编号越界，可用范围={available_ranks or []}",
                        support_score=0.0,
                        sufficiency="low",
                    )
                )
                continue
            if page_cites:
                checks.append(
                    ClaimCheck(
                        claim=claim,
                        cited_ranks=[],
                        supported=False,
                        reason=f"页码引用未命中检索证据，引用页={page_cites}",
                        support_score=0.0,
                        sufficiency="low",
                    )
                )
                continue
            if _has_any_bracket_citation(claim):
                checks.append(
                    ClaimCheck(
                        claim=claim,
                        cited_ranks=[],
                        supported=False,
                        reason="存在引用标记，但无法映射为证据编号",
                        support_score=0.0,
                        sufficiency="low",
                    )
                )
                continue
            checks.append(
                ClaimCheck(
                    claim=claim,
                    cited_ranks=[],
                    supported=False,
                    reason="缺少证据引用",
                    support_score=0.0,
                    sufficiency="low",
                )
            )
            continue

        best_overlap = 0.0
        for rank in resolved_ranks:
            evidence = rank_to_evidence.get(rank, "")
            if not evidence:
                continue
            best_overlap = max(best_overlap, _token_overlap(claim, evidence))

        best_embed = 0.0
        if best_overlap < 0.08 and embed_similarity_fn is not None:
            try:
                claim_emb = embed_similarity_fn(claim)
                for rank in resolved_ranks:
                    evidence = rank_to_evidence.get(rank, "")
                    if not evidence:
                        continue
                    ev_emb = embed_similarity_fn(evidence)
                    best_embed = max(best_embed, _cosine_similarity(claim_emb, ev_emb))
            except Exception:
                best_embed = 0.0

        support_score = max(best_overlap, best_embed)
        if best_overlap >= 0.08 or best_embed >= 0.7:
            verified += 1
            if best_embed >= 0.7 and best_overlap < 0.08:
                reason = f"embedding相似度={best_embed:.2f}"
            else:
                reason = f"语义重叠={best_overlap:.2f}"
            sufficiency = "high" if support_score >= 0.65 else "medium"
            checks.append(
                ClaimCheck(
                    claim=claim,
                    cited_ranks=resolved_ranks,
                    supported=True,
                    reason=reason,
                    support_score=round(support_score, 4),
                    sufficiency=sufficiency,
                )
            )
        else:
            sufficiency = "low"
            checks.append(
                ClaimCheck(
                    claim=claim,
                    cited_ranks=resolved_ranks,
                    supported=False,
                    reason=f"证据语义重叠偏低={best_overlap:.2f}, embedding={best_embed:.2f}",
                    support_score=round(support_score, 4),
                    sufficiency=sufficiency,
                )
            )

    unsupported = len(checks) - verified
    coverage = round(verified / len(checks), 4) if checks else 1.0
    return EvidenceGuardReport(
        coverage=coverage,
        verified_claims=verified,
        unsupported_claims=unsupported,
        checks=checks,
    )
