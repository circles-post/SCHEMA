from __future__ import annotations

from typing import Any

import requests

from .models import PaperRecord
from .utils import normalize_keyword, normalize_text, strip_markup, tokenize

CROSSREF_API_BASE_URL = "https://api.crossref.org"


class CrossrefClient:
    def __init__(self, config: dict[str, Any] | None = None, tool: str = "pubmed_graph_workflow"):
        cfg = dict(config or {})
        self.enabled = bool(cfg.get("enabled", False))
        self.base_url = str(cfg.get("base_url", CROSSREF_API_BASE_URL)).rstrip("/")
        self.mailto = str(cfg.get("mailto", "") or "").strip()
        self.timeout = float(cfg.get("timeout", 30.0))
        self.filter_has_abstract = bool(cfg.get("filter_has_abstract", False))
        self.query_mode = str(cfg.get("query_mode", "bibliographic") or "bibliographic").strip().lower()
        self.tool = tool
        self.session = requests.Session()
        user_agent = f"{self.tool}/1.0"
        if self.mailto:
            user_agent = f"{user_agent} (mailto:{self.mailto})"
        self.session.headers.update({"User-Agent": user_agent})

    def fetch_papers(
        self,
        keyword: str,
        retmax: int = 20,
        mindate: str | None = None,
        maxdate: str | None = None,
    ) -> list[PaperRecord]:
        if not self.enabled:
            return []
        if not self.mailto:
            raise ValueError("Crossref polite-pool usage requires crossref.mailto")
        params = self._build_params(keyword=keyword, retmax=retmax, mindate=mindate, maxdate=maxdate)
        response = self.session.get(f"{self.base_url}/works", params=params, timeout=self.timeout)
        response.raise_for_status()
        payload = response.json()
        items = payload.get("message", {}).get("items", []) if isinstance(payload, dict) else []
        if not isinstance(items, list):
            return []
        matched: list[PaperRecord] = []
        seen: set[str] = set()
        for item in items:
            if not isinstance(item, dict):
                continue
            record = self._to_paper_record(item)
            if not record.title:
                continue
            if self.filter_has_abstract and not record.abstract:
                continue
            if not self._matches_keyword(item, keyword, record):
                continue
            dedupe_key = normalize_keyword(record.doi or record.title)
            if dedupe_key and dedupe_key in seen:
                continue
            if dedupe_key:
                seen.add(dedupe_key)
            record.matched_keywords.append(keyword)
            record.source_queries.append(self._query_label(keyword))
            matched.append(record)
            if len(matched) >= retmax:
                break
        return matched

    def _build_params(
        self,
        keyword: str,
        retmax: int,
        mindate: str | None = None,
        maxdate: str | None = None,
    ) -> dict[str, str]:
        params = {
            "rows": str(max(int(retmax), 1)),
            "mailto": self.mailto,
        }
        if self.query_mode == "title":
            params["query.title"] = keyword
        else:
            params["query.bibliographic"] = keyword
        filters = []
        from_pub = self._to_crossref_date(mindate)
        until_pub = self._to_crossref_date(maxdate)
        if from_pub:
            filters.append(f"from-pub-date:{from_pub}")
        if until_pub:
            filters.append(f"until-pub-date:{until_pub}")
        filters.append("type:journal-article")
        if filters:
            params["filter"] = ",".join(filters)
        return params

    def _query_label(self, keyword: str) -> str:
        query_key = "query.title" if self.query_mode == "title" else "query.bibliographic"
        return f"crossref:{query_key}:{keyword}"

    @staticmethod
    def _to_crossref_date(value: str | None) -> str:
        raw = normalize_text(value)
        if not raw:
            return ""
        if "/" in raw:
            parts = raw.split("/")
            if len(parts) >= 3:
                return f"{parts[0]}-{parts[1]}-{parts[2]}"
            if len(parts) == 2:
                return f"{parts[0]}-{parts[1]}-01"
        if len(raw) == 4 and raw.isdigit():
            return f"{raw}-01-01"
        return raw

    @staticmethod
    def _normalize_doi(value: object) -> str:
        doi = normalize_text(value)
        if not doi:
            return ""
        lowered = doi.lower()
        if lowered.startswith("https://doi.org/"):
            doi = doi[16:]
        elif lowered.startswith("http://doi.org/"):
            doi = doi[15:]
        elif lowered.startswith("doi:"):
            doi = doi[4:]
        return normalize_text(doi).lower()

    @staticmethod
    def _authors(author_rows: Any) -> list[str]:
        if not isinstance(author_rows, list):
            return []
        authors: list[str] = []
        for item in author_rows:
            if not isinstance(item, dict):
                continue
            name = normalize_text(" ".join(part for part in [item.get("given", ""), item.get("family", "")] if normalize_text(part)))
            if not name:
                name = normalize_text(item.get("name", ""))
            if name and name not in authors:
                authors.append(name)
        return authors

    @staticmethod
    def _pick_year(item: dict[str, Any]) -> str | None:
        for field in ["issued", "published-online", "published-print", "published"]:
            payload = item.get(field, {})
            if not isinstance(payload, dict):
                continue
            date_parts = payload.get("date-parts", [])
            if isinstance(date_parts, list) and date_parts and isinstance(date_parts[0], list) and date_parts[0]:
                year = normalize_text(date_parts[0][0])
                if year:
                    return year
        return None

    def _to_paper_record(self, item: dict[str, Any]) -> PaperRecord:
        title_rows = item.get("title", [])
        journal_rows = item.get("container-title", [])
        abstract = strip_markup(item.get("abstract", ""))
        doi = self._normalize_doi(item.get("DOI", "")) or None
        title = normalize_text(title_rows[0] if isinstance(title_rows, list) and title_rows else item.get("title", ""))
        journal = normalize_text(journal_rows[0] if isinstance(journal_rows, list) and journal_rows else item.get("container-title", ""))
        return PaperRecord(
            pmid="",
            pmcid=None,
            doi=doi,
            title=title,
            abstract=abstract,
            journal=journal,
            publication_year=self._pick_year(item),
            authors=self._authors(item.get("author", [])),
            mesh_terms=[],
            has_pmc_fulltext=False,
            retrieval_source="crossref",
            preprint_server=None,
            preprint_category=None,
            jats_xml_path=None,
            published_doi=None,
            landing_page_url=normalize_text(item.get("URL", "")) or None,
        )

    @staticmethod
    def _matches_keyword(item: dict[str, Any], keyword: str, record: PaperRecord) -> bool:
        phrase = normalize_keyword(keyword)
        if not phrase:
            return False
        fields = [
            record.title,
            record.abstract,
            record.journal,
            " ".join(record.authors),
            item.get("publisher", ""),
            item.get("subtitle", [""])[0] if isinstance(item.get("subtitle", []), list) and item.get("subtitle", []) else "",
        ]
        combined = normalize_keyword(" ".join(str(value or "") for value in fields))
        if phrase in combined:
            return True
        phrase_tokens = tokenize(keyword)
        if not phrase_tokens:
            return False
        return phrase_tokens.issubset(tokenize(combined))
