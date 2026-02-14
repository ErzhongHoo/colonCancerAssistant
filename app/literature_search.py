from __future__ import annotations

import json
import re
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass
from typing import Callable


USER_AGENT = "ColonCancerResearchBot/1.0 (+literature-search)"
ONCOLOGY_TERMS = {
    "oncology",
    "cancer",
    "tumor",
    "tumour",
    "neoplasm",
    "carcinoma",
    "adenocarcinoma",
    "colorectal",
    "colon",
    "rectal",
    "rectum",
    "metastasis",
    "metastatic",
    "chemotherapy",
    "radiotherapy",
    "immunotherapy",
    "targeted therapy",
    "tnm",
}
MEDICAL_TERMS = {
    "clinical",
    "trial",
    "patient",
    "disease",
    "treatment",
    "therapy",
    "medicine",
    "medical",
    "guideline",
    "diagnosis",
    "prognosis",
    "survival",
}


@dataclass
class LiteratureItem:
    title: str
    abstract: str
    year: int | None
    venue: str
    doi: str
    url: str
    source: str
    paper_type: str = "other"
    relevance: float = 0.0

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def _http_get_json(url: str, timeout: float = 8.0) -> dict:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        payload = resp.read().decode("utf-8", errors="ignore")
    return json.loads(payload)


def _clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "")).strip()


def _tokenize(text: str) -> set[str]:
    return set(re.findall(r"[A-Za-z0-9\u4e00-\u9fa5]{2,}", (text or "").lower()))


def _score_relevance(query: str, title: str, abstract: str, year: int | None, paper_type: str = "other") -> float:
    """Multi-signal relevance scoring.

    Combines:
    1. Token overlap F1 (harmonic mean of recall & precision).
    2. CJK bi-gram overlap for Chinese queries.
    3. Paper type bonus (guidelines & systematic reviews rank higher).
    4. Recency with gentler decay.
    """
    q_tokens = _tokenize(query)
    d_tokens = _tokenize(f"{title} {abstract}")

    # --- Signal 1: Token overlap F1 ---
    if q_tokens and d_tokens:
        intersection = len(q_tokens & d_tokens)
        recall = intersection / max(len(q_tokens), 1)
        precision = intersection / max(len(d_tokens), 1)
        f1 = 2 * recall * precision / max(recall + precision, 1e-8)
    else:
        f1 = 0.0

    # --- Signal 2: CJK bi-gram overlap ---
    q_cn = re.sub(r"[^\u4e00-\u9fa5]", "", query)
    d_cn = re.sub(r"[^\u4e00-\u9fa5]", "", f"{title} {abstract}")
    cjk_score = 0.0
    if len(q_cn) >= 2 and len(d_cn) >= 2:
        q_bigrams = {q_cn[i:i+2] for i in range(len(q_cn) - 1)}
        d_bigrams = {d_cn[i:i+2] for i in range(len(d_cn) - 1)}
        if q_bigrams:
            cjk_score = len(q_bigrams & d_bigrams) / max(len(q_bigrams), 1)

    # --- Signal 3: Paper type bonus ---
    type_bonus = {
        "guideline": 0.15,
        "meta_or_systematic_review": 0.10,
        "clinical_trial": 0.05,
    }.get(paper_type, 0.0)

    # --- Signal 4: Recency with gentler decay ---
    recency = 0.0
    if year:
        now_year = time.gmtime().tm_year
        age = max(now_year - year, 0)
        recency = max(0.0, 1.0 - (min(age, 15) / 15.0) ** 0.7)

    score = f1 * 0.40 + cjk_score * 0.20 + recency * 0.20 + type_bonus + 0.20 * max(f1, cjk_score)
    return round(min(score, 1.0), 4)


def _is_medical_oncology_item(title: str, abstract: str, venue: str) -> bool:
    text = f"{title} {abstract} {venue}".lower()
    has_oncology = any(term in text for term in ONCOLOGY_TERMS)
    has_medical = any(term in text for term in MEDICAL_TERMS)
    return has_oncology and has_medical


def _infer_paper_type(title: str, venue: str, abstract: str = "") -> str:
    t = f"{title} {venue} {abstract}".lower()
    if any(k in t for k in ["guideline", "consensus", "practice guideline", "指南", "共识"]):
        return "guideline"
    if any(k in t for k in ["meta-analysis", "systematic review", "meta analysis"]):
        return "meta_or_systematic_review"
    if any(k in t for k in ["randomized", "randomised", "phase iii", "phase ii", "trial"]):
        return "clinical_trial"
    if any(k in t for k in ["cohort", "retrospective", "prospective", "registry", "observational"]):
        return "observational_study"
    if any(k in t for k in ["conference", "proceedings", "abstract", "asco", "esmo"]):
        return "conference_abstract_or_proceeding"
    if any(k in t for k in ["review"]):
        return "review"
    return "other"


def _search_pubmed(query: str, limit: int, timeout: float) -> list[LiteratureItem]:
    term = urllib.parse.quote(query)
    esearch = (
        "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
        f"?db=pubmed&retmode=json&retmax={max(limit, 1)}&sort=relevance&term={term}"
    )
    search_data = _http_get_json(esearch, timeout=timeout)
    ids = search_data.get("esearchresult", {}).get("idlist", [])
    if not ids:
        return []

    id_arg = ",".join(ids)
    esummary = (
        "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"
        f"?db=pubmed&retmode=json&id={urllib.parse.quote(id_arg)}"
    )
    summary_data = _http_get_json(esummary, timeout=timeout)
    result = summary_data.get("result", {})
    abstract_map = _fetch_pubmed_abstracts(ids, timeout=timeout)

    items: list[LiteratureItem] = []
    for pid in ids:
        row = result.get(pid, {})
        title = _clean_text(str(row.get("title", "")))
        if not title:
            continue
        pubdate = str(row.get("pubdate", ""))
        year_match = re.search(r"(19|20)\d{2}", pubdate)
        year = int(year_match.group(0)) if year_match else None

        article_ids = row.get("articleids", []) or []
        doi = ""
        for aid in article_ids:
            if str(aid.get("idtype", "")).lower() == "doi":
                doi = str(aid.get("value", "")).strip()
                break

        venue = _clean_text(str(row.get("fulljournalname", "") or row.get("source", "")))
        url = f"https://pubmed.ncbi.nlm.nih.gov/{pid}/"
        abstract = _clean_text(abstract_map.get(pid, ""))
        items.append(
            LiteratureItem(
                title=title,
                abstract=abstract,
                year=year,
                venue=venue,
                doi=doi,
                url=url,
                source="pubmed",
                paper_type=_infer_paper_type(title, venue, abstract),
            )
        )
    return items


def _fetch_pubmed_abstracts(ids: list[str], timeout: float) -> dict[str, str]:
    if not ids:
        return {}
    id_arg = ",".join(ids)
    efetch = (
        "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
        f"?db=pubmed&retmode=xml&id={urllib.parse.quote(id_arg)}"
    )
    req = urllib.request.Request(efetch, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            payload = resp.read().decode("utf-8", errors="ignore")
    except Exception:
        return {}
    try:
        root = ET.fromstring(payload)
    except Exception:
        return {}
    out: dict[str, str] = {}
    for article in root.findall(".//PubmedArticle"):
        pmid_node = article.find(".//MedlineCitation/PMID")
        if pmid_node is None or not pmid_node.text:
            continue
        pmid = pmid_node.text.strip()
        abstract_parts = []
        for ab in article.findall(".//Abstract/AbstractText"):
            if ab.text:
                label = ab.attrib.get("Label", "").strip()
                if label:
                    abstract_parts.append(f"{label}: {ab.text.strip()}")
                else:
                    abstract_parts.append(ab.text.strip())
        out[pmid] = _clean_text(" ".join(abstract_parts))
    return out


def _search_openalex(query: str, limit: int, timeout: float) -> list[LiteratureItem]:
    params = urllib.parse.urlencode(
        {
            "search": query,
            "per-page": max(limit, 1),
            "sort": "relevance_score:desc",
        }
    )
    url = f"https://api.openalex.org/works?{params}"
    data = _http_get_json(url, timeout=timeout)

    items: list[LiteratureItem] = []
    for row in data.get("results", [])[:limit]:
        title = _clean_text(str(row.get("display_name", "")))
        if not title:
            continue
        year = row.get("publication_year")
        doi = str(row.get("doi", "") or "").replace("https://doi.org/", "")
        venue = _clean_text(str((row.get("primary_location") or {}).get("source", {}).get("display_name", "")))
        abstract = ""
        inv = row.get("abstract_inverted_index") or {}
        if isinstance(inv, dict) and inv:
            words = sorted(((pos, word) for word, poses in inv.items() for pos in poses), key=lambda x: x[0])
            abstract = _clean_text(" ".join(w for _, w in words))
        url_value = str((row.get("primary_location") or {}).get("landing_page_url", "") or row.get("id", ""))
        items.append(
            LiteratureItem(
                title=title,
                abstract=abstract,
                year=int(year) if isinstance(year, int) else None,
                venue=venue,
                doi=doi,
                url=url_value,
                source="openalex",
                paper_type=_infer_paper_type(title, venue, abstract),
            )
        )
    return items


def _search_semantic_scholar(query: str, limit: int, timeout: float) -> list[LiteratureItem]:
    params = urllib.parse.urlencode(
        {
            "query": query,
            "limit": max(limit, 1),
            "fields": "title,abstract,year,venue,url,externalIds",
        }
    )
    url = f"https://api.semanticscholar.org/graph/v1/paper/search?{params}"
    data = _http_get_json(url, timeout=timeout)

    items: list[LiteratureItem] = []
    for row in data.get("data", [])[:limit]:
        title = _clean_text(str(row.get("title", "")))
        if not title:
            continue
        ext = row.get("externalIds") or {}
        doi = str(ext.get("DOI", "") or "").strip()
        items.append(
            LiteratureItem(
                title=title,
                abstract=_clean_text(str(row.get("abstract", ""))),
                year=int(row["year"]) if isinstance(row.get("year"), int) else None,
                venue=_clean_text(str(row.get("venue", ""))),
                doi=doi,
                url=_clean_text(str(row.get("url", ""))),
                source="semanticscholar",
                paper_type=_infer_paper_type(
                    title,
                    _clean_text(str(row.get("venue", ""))),
                    _clean_text(str(row.get("abstract", ""))),
                ),
            )
        )
    return items


def _dedupe(items: list[LiteratureItem]) -> list[LiteratureItem]:
    seen: set[str] = set()
    out: list[LiteratureItem] = []
    for item in items:
        key = (item.doi or item.title).strip().lower()
        if not key:
            continue
        if key in seen:
            continue
        seen.add(key)
        out.append(item)
    return out


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
    if n1 <= 1e-12 or n2 <= 1e-12:
        return 0.0
    return dot / ((n1 ** 0.5) * (n2 ** 0.5))


def _embedding_rerank(
    query: str,
    items: list[LiteratureItem],
    embed_fn: Callable[[str], list[float]] | None,
) -> list[LiteratureItem]:
    if not items or embed_fn is None:
        return items
    try:
        qv = embed_fn(query)
    except Exception:
        return items
    rescored: list[LiteratureItem] = []
    for item in items:
        doc = f"{item.title}. {item.abstract}".strip()
        try:
            dv = embed_fn(doc)
            emb_score = _cosine_similarity(qv, dv)
        except Exception:
            emb_score = 0.0
        # Blend lexical/recent score and embedding score.
        item.relevance = round(item.relevance * 0.35 + emb_score * 0.65, 4)
        rescored.append(item)
    rescored.sort(key=lambda x: x.relevance, reverse=True)
    return rescored


def search_literature(
    query: str,
    providers: list[str],
    top_k: int = 5,
    timeout: float = 8.0,
    embed_fn: Callable[[str], list[float]] | None = None,
    medical_oncology_only: bool = True,
    min_relevance: float = 0.18,
) -> list[LiteratureItem]:
    if not query.strip():
        return []

    provider_map: dict[str, Callable[[str, int, float], list[LiteratureItem]]] = {
        "pubmed": _search_pubmed,
        "openalex": _search_openalex,
        "semanticscholar": _search_semantic_scholar,
    }

    merged: list[LiteratureItem] = []
    for p in providers:
        fn = provider_map.get(p.strip().lower())
        if fn is None:
            continue
        try:
            merged.extend(fn(query, max(top_k, 1), timeout))
        except Exception:
            continue

    deduped = _dedupe(merged)
    if medical_oncology_only:
        deduped = [
            item
            for item in deduped
            if _is_medical_oncology_item(item.title, item.abstract, item.venue)
        ]
    for item in deduped:
        item.relevance = _score_relevance(query, item.title, item.abstract, item.year, paper_type=item.paper_type)
    deduped.sort(key=lambda x: x.relevance, reverse=True)
    deduped = _embedding_rerank(query, deduped[: max(top_k * 3, top_k)], embed_fn=embed_fn)
    deduped = [item for item in deduped if item.relevance >= min_relevance]
    return deduped[: max(top_k, 1)]


def literature_to_context(items: list[object], max_items: int = 5) -> str:
    if not items:
        return "暂无外部学术文献。"
    lines: list[str] = []
    for idx, raw in enumerate(items[:max_items], start=1):
        if isinstance(raw, LiteratureItem):
            title = raw.title
            year = raw.year if raw.year is not None else "n/a"
            venue = raw.venue or raw.source
            doi = raw.doi or "n/a"
            source = raw.source
            ptype = raw.paper_type
        elif isinstance(raw, dict):
            title = str(raw.get("title", ""))
            y = raw.get("year")
            year = y if isinstance(y, int) else "n/a"
            venue = str(raw.get("venue", "") or raw.get("source", ""))
            doi = str(raw.get("doi", "") or "n/a")
            source = str(raw.get("source", ""))
            ptype = str(raw.get("paper_type", "other"))
        else:
            continue
        lines.append(
            f"[文献#{idx}] {title} | type={ptype} | year={year} | venue={venue} | doi={doi} | source={source}"
        )
    return "\n".join(lines)


def verify_literature_citations(
    answer: str,
    external_items: list[dict[str, object]],
    embed_fn: Callable[[str], list[float]] | None = None,
) -> dict[str, object]:
    claim_lines = [ln.strip() for ln in re.split(r"[\n。；;!?！？]+", answer) if ln.strip()]
    claim_lines = [ln for ln in claim_lines if "[文献#" in ln]
    lit_map = {
        idx + 1: f"{str(item.get('title', ''))}. {str(item.get('abstract', ''))}".strip()
        for idx, item in enumerate(external_items)
    }
    checks: list[dict[str, object]] = []
    verified = 0
    for claim in claim_lines:
        cited = []
        for m in re.finditer(r"\[文献#(\d+)\]", claim):
            try:
                cited.append(int(m.group(1)))
            except Exception:
                continue
        cited = sorted(set(cited))
        if not cited:
            checks.append(
                {
                    "claim": claim,
                    "cited_refs": [],
                    "supported": False,
                    "score": 0.0,
                    "reason": "缺少文献引用",
                }
            )
            continue
        texts = [lit_map.get(i, "") for i in cited if lit_map.get(i, "")]
        if not texts:
            checks.append(
                {
                    "claim": claim,
                    "cited_refs": cited,
                    "supported": False,
                    "score": 0.0,
                    "reason": "文献编号越界",
                }
            )
            continue
        score = 0.0
        if embed_fn is not None:
            try:
                q = embed_fn(claim)
                for t in texts:
                    score = max(score, _cosine_similarity(q, embed_fn(t)))
            except Exception:
                score = 0.0
        else:
            q_tok = _tokenize(claim)
            for t in texts:
                d_tok = _tokenize(t)
                score = max(score, len(q_tok & d_tok) / max(len(q_tok), 1))
        ok = score >= 0.52
        if ok:
            verified += 1
        checks.append(
            {
                "claim": claim,
                "cited_refs": cited,
                "supported": ok,
                "score": round(score, 4),
                "reason": f"claim-文献一致性={score:.2f}",
            }
        )
    total = len(checks)
    return {
        "coverage": round(verified / total, 4) if total else 1.0,
        "verified": verified,
        "unsupported": total - verified,
        "checks": checks,
    }
