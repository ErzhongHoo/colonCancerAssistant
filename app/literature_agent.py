from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Callable

from dotenv import load_dotenv

from app.rag import LocalVectorStore, split_text

load_dotenv()


ROOT = Path(__file__).resolve().parent.parent
LITERATURE_STORE_PATH = ROOT / "data" / "literature_store.jsonl"
LITERATURE_VECTOR_PATH = ROOT / "data" / "literature_vector_store.json"
LITERATURE_STATE_PATH = ROOT / "data" / "literature_agent_state.json"

USER_AGENT = "ColonCancerResearch-LiteratureAgent/1.0"
DEFAULT_TOPIC_QUERY = (
    "(breast cancer OR mammary carcinoma OR breast neoplasm)"
    " AND (clinical OR guideline OR trial OR treatment)"
)

ONCOLOGY_TERMS = (
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
    "metastasis",
    "metastatic",
    "chemotherapy",
    "radiotherapy",
    "immunotherapy",
)
MEDICAL_TERMS = (
    "clinical",
    "trial",
    "patient",
    "therapy",
    "treatment",
    "guideline",
    "diagnosis",
    "survival",
    "medicine",
    "disease",
)


def _http_get_json(url: str, timeout: float = 12.0) -> dict:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        payload = resp.read().decode("utf-8", errors="ignore")
    return json.loads(payload)


def _http_get_text(url: str, timeout: float = 12.0) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read().decode("utf-8", errors="ignore")


def _clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "")).strip()


def _infer_paper_type(title: str, venue: str, abstract: str) -> str:
    t = f"{title} {venue} {abstract}".lower()
    if any(k in t for k in ("guideline", "consensus", "practice guideline", "指南", "共识")):
        return "guideline"
    if any(k in t for k in ("meta-analysis", "systematic review", "meta analysis")):
        return "meta_or_systematic_review"
    if any(k in t for k in ("randomized", "randomised", "phase iii", "phase ii", "trial")):
        return "clinical_trial"
    if any(k in t for k in ("cohort", "retrospective", "prospective", "registry", "observational")):
        return "observational_study"
    if any(k in t for k in ("conference", "proceedings", "abstract", "asco", "esmo")):
        return "conference_abstract_or_proceeding"
    if "review" in t:
        return "review"
    return "other"


def _paper_type_to_evidence_level(paper_type: str) -> str:
    if paper_type == "guideline":
        return "A"
    if paper_type in {"meta_or_systematic_review", "clinical_trial"}:
        return "B"
    if paper_type in {"observational_study", "review"}:
        return "C"
    return "D"


def _topic_tags(title: str, abstract: str) -> list[str]:
    text = f"{title} {abstract}".lower()
    tags = []
    if "rectal" in text:
        tags.append("rectal")
    if "colon" in text:
        tags.append("colon")
    if "metast" in text:
        tags.append("metastatic")
    if "immun" in text:
        tags.append("immunotherapy")
    if "target" in text:
        tags.append("targeted")
    if "radi" in text:
        tags.append("radiotherapy")
    if "chemo" in text:
        tags.append("chemotherapy")
    if "surgery" in text or "resection" in text:
        tags.append("surgery")
    return sorted(set(tags))


def _is_medical_oncology_item(title: str, abstract: str, venue: str) -> bool:
    text = f"{title} {abstract} {venue}".lower()
    has_oncology = any(term in text for term in ONCOLOGY_TERMS)
    has_medical = any(term in text for term in MEDICAL_TERMS)
    return has_oncology and has_medical


def _freshness_score(year: int | None) -> float:
    if year is None:
        return 0.0
    now_year = time.gmtime().tm_year
    age = max(now_year - year, 0)
    return round(max(0.0, 1.0 - min(age, 10) / 10.0), 4)


def _doc_id(pmid: str, doi: str, title: str) -> str:
    if doi:
        normalized = doi.strip().lower()
        return f"doi:{normalized}"
    if pmid:
        return f"pmid:{pmid}"
    digest = hashlib.md5(title.encode("utf-8", errors="ignore")).hexdigest()[:16]
    return f"title:{digest}"


def _fetch_pubmed_ids(topic_query: str, since_date: str | None, max_results: int, timeout: float) -> list[str]:
    term = urllib.parse.quote(topic_query)
    url = (
        "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
        f"?db=pubmed&retmode=json&retmax={max(max_results,1)}&sort=pub+date&term={term}"
    )
    if since_date:
        encoded_date = urllib.parse.quote(since_date)
        url += f"&datetype=pdat&mindate={encoded_date}&maxdate=3000"
    data = _http_get_json(url, timeout=timeout)
    ids = data.get("esearchresult", {}).get("idlist", [])
    return [str(x).strip() for x in ids if str(x).strip()]


def _fetch_pubmed_abstracts(ids: list[str], timeout: float) -> dict[str, str]:
    if not ids:
        return {}
    id_arg = ",".join(ids)
    efetch = (
        "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
        f"?db=pubmed&retmode=xml&id={urllib.parse.quote(id_arg)}"
    )
    payload = _http_get_text(efetch, timeout=timeout)
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
        parts: list[str] = []
        for ab in article.findall(".//Abstract/AbstractText"):
            if ab.text:
                label = ab.attrib.get("Label", "").strip()
                if label:
                    parts.append(f"{label}: {ab.text.strip()}")
                else:
                    parts.append(ab.text.strip())
        out[pmid] = _clean_text(" ".join(parts))
    return out


def fetch_pubmed_raw(topic_query: str, since_date: str | None, max_results: int, timeout: float) -> list[dict[str, str]]:
    ids = _fetch_pubmed_ids(topic_query, since_date, max_results=max_results, timeout=timeout)
    if not ids:
        return []
    id_arg = ",".join(ids)
    esummary = (
        "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"
        f"?db=pubmed&retmode=json&id={urllib.parse.quote(id_arg)}"
    )
    summary_data = _http_get_json(esummary, timeout=timeout)
    result = summary_data.get("result", {})
    abstracts = _fetch_pubmed_abstracts(ids, timeout=timeout)

    rows: list[dict[str, str]] = []
    for pid in ids:
        row = result.get(pid, {}) or {}
        title = _clean_text(str(row.get("title", "")))
        if not title:
            continue
        pubdate = str(row.get("pubdate", ""))
        year_match = re.search(r"(19|20)\d{2}", pubdate)
        year = year_match.group(0) if year_match else ""
        doi = ""
        for aid in (row.get("articleids", []) or []):
            if str(aid.get("idtype", "")).lower() == "doi":
                doi = str(aid.get("value", "")).strip()
                break
        venue = _clean_text(str(row.get("fulljournalname", "") or row.get("source", "")))
        rows.append(
            {
                "pmid": str(pid),
                "doi": doi,
                "title": title,
                "abstract": abstracts.get(str(pid), ""),
                "year": year,
                "venue": venue,
                "url": f"https://pubmed.ncbi.nlm.nih.gov/{pid}/",
                "source": "pubmed",
            }
        )
    return rows


def normalize_records(raw_rows: list[dict[str, str]]) -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    now = int(time.time())
    for row in raw_rows:
        title = _clean_text(str(row.get("title", "")))
        abstract = _clean_text(str(row.get("abstract", "")))
        venue = _clean_text(str(row.get("venue", "")))
        if not title:
            continue
        if not _is_medical_oncology_item(title, abstract, venue):
            continue
        year_raw = str(row.get("year", "")).strip()
        year = int(year_raw) if year_raw.isdigit() else None
        paper_type = _infer_paper_type(title, venue, abstract)
        pmid = str(row.get("pmid", "")).strip()
        doi = str(row.get("doi", "")).strip().lower()
        record = {
            "doc_id": _doc_id(pmid, doi, title),
            "pmid": pmid,
            "doi": doi,
            "title": title,
            "abstract": abstract,
            "year": year,
            "venue": venue,
            "url": str(row.get("url", "")).strip(),
            "source": str(row.get("source", "pubmed")),
            "paper_type": paper_type,
            "evidence_level": _paper_type_to_evidence_level(paper_type),
            "topic_tags": _topic_tags(title, abstract),
            "freshness_score": _freshness_score(year),
            "updated_at": now,
        }
        out.append(record)
    return out


def load_records(path: Path = LITERATURE_STORE_PATH) -> list[dict[str, object]]:
    if not path.exists():
        return []
    rows: list[dict[str, object]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
            if isinstance(obj, dict):
                rows.append(obj)
        except Exception:
            continue
    return rows


def save_records(records: list[dict[str, object]], path: Path = LITERATURE_STORE_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps(item, ensure_ascii=False) for item in records]
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def merge_records(existing: list[dict[str, object]], incoming: list[dict[str, object]]) -> list[dict[str, object]]:
    merged: dict[str, dict[str, object]] = {}
    for item in existing:
        doc_id = str(item.get("doc_id", "")).strip()
        if doc_id:
            merged[doc_id] = item
    for item in incoming:
        doc_id = str(item.get("doc_id", "")).strip()
        if doc_id:
            merged[doc_id] = item
    rows = list(merged.values())
    rows.sort(key=lambda x: (x.get("year") or 0, x.get("updated_at") or 0), reverse=True)
    return rows


def _record_to_rag_text(record: dict[str, object]) -> str:
    title = str(record.get("title", ""))
    abstract = str(record.get("abstract", ""))
    year = record.get("year")
    venue = str(record.get("venue", ""))
    ptype = str(record.get("paper_type", "other"))
    level = str(record.get("evidence_level", "D"))
    doi = str(record.get("doi", ""))
    tags = ",".join(record.get("topic_tags", []) or [])
    return (
        f"[文献元数据] title={title}; year={year}; venue={venue}; type={ptype}; "
        f"evidence_level={level}; doi={doi}; tags={tags}\n"
        f"[摘要]\n{abstract}"
    ).strip()


def rebuild_literature_vector_store(
    records: list[dict[str, object]],
    embed_fn: Callable[[str], list[float]],
    vector_path: Path = LITERATURE_VECTOR_PATH,
) -> int:
    store = LocalVectorStore(vector_path)
    store.clear()
    total = 0
    for rec in records:
        doc_id = str(rec.get("doc_id", "")).strip()
        if not doc_id:
            continue
        text = _record_to_rag_text(rec)
        chunks = split_text(text, chunk_size=900, overlap=120)
        total += store.add_texts(source=f"literature/{doc_id}", texts=chunks, embed_fn=embed_fn)
    return total


def load_index(path: Path = LITERATURE_STORE_PATH) -> dict[str, dict[str, object]]:
    rows = load_records(path)
    return {str(r.get("doc_id", "")): r for r in rows if str(r.get("doc_id", ""))}


def _load_state(path: Path = LITERATURE_STATE_PATH) -> dict[str, object]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _save_state(payload: dict[str, object], path: Path = LITERATURE_STATE_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def run_incremental_update(
    topic_query: str,
    embed_fn: Callable[[str], list[float]],
    max_results: int = 120,
    timeout: float = 12.0,
) -> dict[str, object]:
    state = _load_state()
    since_date = str(state.get("last_success_date", "")).strip() or None
    raw = fetch_pubmed_raw(topic_query, since_date=since_date, max_results=max_results, timeout=timeout)
    normalized = normalize_records(raw)
    existing = load_records()
    merged = merge_records(existing, normalized)
    save_records(merged)
    chunk_count = rebuild_literature_vector_store(merged, embed_fn=embed_fn)
    today = time.strftime("%Y/%m/%d", time.gmtime())
    _save_state(
        {
            "last_success_date": today,
            "last_run_at": int(time.time()),
            "last_topic_query": topic_query,
            "fetched_raw": len(raw),
            "normalized_kept": len(normalized),
            "total_records": len(merged),
            "total_chunks": chunk_count,
        }
    )
    return {
        "ok": True,
        "since_date": since_date,
        "fetched_raw": len(raw),
        "normalized_kept": len(normalized),
        "total_records": len(merged),
        "total_chunks": chunk_count,
        "topic_query": topic_query,
    }


def search_from_local_store(
    query: str,
    embed_fn: Callable[[str], list[float]],
    top_k: int = 5,
    vector_path: Path = LITERATURE_VECTOR_PATH,
    index: dict[str, dict[str, object]] | None = None,
) -> list[dict[str, object]]:
    if not query.strip():
        return []
    store = LocalVectorStore(vector_path)
    if not store.chunks:
        return []
    scored = store.similarity_search_with_scores(query, embed_fn=embed_fn, k=max(top_k * 4, top_k))
    idx = index or load_index()

    out: list[dict[str, object]] = []
    seen_doc: set[str] = set()
    for score, chunk in scored:
        src = str(chunk.source)
        doc_id = src.removeprefix("literature/") if src.startswith("literature/") else src
        if doc_id in seen_doc:
            continue
        rec = idx.get(doc_id)
        if not rec:
            continue
        payload = dict(rec)
        payload["relevance"] = round(float(score), 4)
        out.append(payload)
        seen_doc.add(doc_id)
        if len(out) >= max(top_k, 1):
            break
    return out


def cli_main() -> None:
    parser = argparse.ArgumentParser(description="Run local literature agent incremental update.")
    parser.add_argument(
        "--topic",
        default=DEFAULT_TOPIC_QUERY,
        help="PubMed topic query for incremental crawling",
    )
    parser.add_argument(
        "--max-results",
        type=int,
        default=120,
        help="Max PubMed records to fetch in this run",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=12.0,
        help="HTTP timeout in seconds",
    )
    args = parser.parse_args()

    from app.llm import build_client, embed_text

    model = "text-embedding-v3"
    client = build_client()

    result = run_incremental_update(
        topic_query=args.topic,
        embed_fn=lambda text: embed_text(client, model, text),
        max_results=max(args.max_results, 20),
        timeout=max(args.timeout, 1.0),
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    cli_main()
