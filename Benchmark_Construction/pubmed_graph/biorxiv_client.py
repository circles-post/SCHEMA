from __future__ import annotations

import tempfile
from typing import Any

import requests

from .models import PaperRecord
from .utils import normalize_keyword, normalize_text, tokenize

BIORXIV_API_BASE_URL = "https://api.biorxiv.org"
BIORXIV_JATS_BASE_URL = "https://www.biorxiv.org"


class BiorxivClient:
    def __init__(self, config: dict[str, Any] | None = None):
        cfg = dict(config or {})
        self.enabled = bool(cfg.get("enabled", False))
        self.base_url = str(cfg.get("base_url", BIORXIV_API_BASE_URL)).rstrip("/")
        self.jats_base_url = str(cfg.get("jats_base_url", BIORXIV_JATS_BASE_URL)).rstrip("/")
        self.timeout = float(cfg.get("timeout", 30.0))
        self.max_pages_per_keyword = max(int(cfg.get("max_pages_per_keyword", 5)), 1)
        self.page_size = max(int(cfg.get("page_size", 100)), 1)
        self.interval_mode = str(cfg.get("interval_mode", "date_range")).strip().lower()
        self.start_date = str(cfg.get("start_date", "") or "").strip()
        self.end_date = str(cfg.get("end_date", "") or "").strip()
        self.recent_count = max(int(cfg.get("recent_count", 500)), 1)
        self.recent_days = max(int(cfg.get("recent_days", 30)), 1)
        self.default_category = str(cfg.get("category", "") or "").strip()
        self.session = requests.Session()

    def fetch_papers(
        self,
        keyword: str,
        server: str = "biorxiv",
        retmax: int = 20,
        category: str | None = None,
    ) -> list[PaperRecord]:
        if not self.enabled:
            return []
        server = normalize_keyword(server) or "biorxiv"
        chosen_category = str(category or self.default_category).strip()
        interval = self._build_interval()
        matched: list[PaperRecord] = []
        seen_dois: set[str] = set()
        for page in range(self.max_pages_per_keyword):
            cursor = page * self.page_size
            payload = self._details(server=server, interval=interval, cursor=cursor, category=chosen_category)
            rows = payload.get("collection", [])
            if not isinstance(rows, list) or not rows:
                break
            for item in rows:
                if not isinstance(item, dict):
                    continue
                if not self._matches_keyword(item, keyword):
                    continue
                record = self._to_paper_record(item, server=server)
                dedupe_key = normalize_keyword(record.doi or "")
                if dedupe_key and dedupe_key in seen_dois:
                    continue
                if dedupe_key:
                    seen_dois.add(dedupe_key)
                record.matched_keywords.append(keyword)
                record.source_queries.append(f"{server}:{interval}")
                matched.append(record)
                if len(matched) >= retmax:
                    return matched
            if len(rows) < self.page_size:
                break
        return matched

    def fetch_jats_xml(self, path_or_url: str) -> str:
        target = str(path_or_url or "").strip()
        if not target:
            raise ValueError("Missing bioRxiv JATS XML path")
        url = target if target.startswith(("http://", "https://")) else f"{self.jats_base_url}/{target.lstrip('/')}"
        response = self.session.get(url, timeout=self.timeout)
        response.raise_for_status()
        return response.text

    def candidate_pdf_urls(self, paper: PaperRecord) -> list[str]:
        urls: list[str] = []
        if paper.doi:
            base = paper.doi.replace("doi.org/", "").strip("/")
            version = normalize_text(getattr(paper, "published_doi", "") or "")
            if version:
                version_suffix = version.split("/")[-1]
            else:
                version_suffix = ""
            urls.append(f"{self.jats_base_url}/content/{base}v1.full.pdf")
            urls.append(f"{self.jats_base_url}/content/{base}v2.full.pdf")
            urls.append(f"{self.jats_base_url}/content/{base}.full.pdf")
            if version_suffix:
                urls.append(f"{self.jats_base_url}/content/{version_suffix}.full.pdf")
        if paper.jats_xml_path:
            jats = paper.jats_xml_path.strip()
            if jats.endswith(".source.xml"):
                urls.append(jats[:-11] + ".full.pdf")
            if jats.endswith(".xml"):
                urls.append(jats[:-4] + ".full.pdf")
        seen: set[str] = set()
        ordered: list[str] = []
        for url in urls:
            if url and url not in seen:
                seen.add(url)
                ordered.append(url)
        return ordered

    def download_pdf(self, paper: PaperRecord, destination_path: str | None = None) -> str:
        errors: list[str] = []
        for url in self.candidate_pdf_urls(paper):
            try:
                response = self.session.get(url, timeout=self.timeout, stream=True)
                if response.status_code != 200:
                    errors.append(f"{url} -> HTTP {response.status_code}")
                    continue
                content_type = normalize_keyword(response.headers.get("content-type", ""))
                if "pdf" not in content_type and not url.lower().endswith(".pdf"):
                    errors.append(f"{url} -> unexpected content-type {content_type}")
                    continue
                if destination_path:
                    with open(destination_path, "wb") as handle:
                        for chunk in response.iter_content(chunk_size=8192):
                            if chunk:
                                handle.write(chunk)
                    return destination_path
                with tempfile.NamedTemporaryFile(prefix="biorxiv_", suffix=".pdf", delete=False) as handle:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            handle.write(chunk)
                    return handle.name
            except Exception as exc:
                errors.append(f"{url} -> {exc}")
        raise RuntimeError("; ".join(errors) if errors else "No candidate bioRxiv PDF URLs available")

    def _details(self, server: str, interval: str, cursor: int, category: str = "") -> dict[str, Any]:
        url = f"{self.base_url}/details/{server}/{interval}/{cursor}/json"
        params = {"category": category} if category else None
        response = self.session.get(url, params=params, timeout=self.timeout)
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, dict):
            raise ValueError("bioRxiv API returned non-dict JSON")
        return payload

    def _build_interval(self) -> str:
        if self.interval_mode == "recent_count":
            return str(self.recent_count)
        if self.interval_mode == "recent_days":
            return f"{self.recent_days}d"
        if not self.start_date or not self.end_date:
            raise ValueError("bioRxiv date_range mode requires start_date and end_date")
        return f"{self.start_date}/{self.end_date}"

    @staticmethod
    def _matches_keyword(item: dict[str, Any], keyword: str) -> bool:
        phrase = normalize_keyword(keyword)
        if not phrase:
            return False
        phrase_tokens = tokenize(keyword)
        fields = [
            item.get("title", ""),
            item.get("abstract", ""),
            item.get("category", ""),
            item.get("authors", ""),
        ]
        combined = normalize_keyword(" ".join(str(value or "") for value in fields))
        if phrase in combined:
            return True
        if not phrase_tokens:
            return False
        return phrase_tokens.issubset(tokenize(combined))

    @staticmethod
    def _split_authors(authors: str) -> list[str]:
        value = normalize_text(authors)
        if not value:
            return []
        if ";" in value:
            return [normalize_text(item) for item in value.split(";") if normalize_text(item)]
        return [normalize_text(item) for item in value.split(",") if normalize_text(item)]

    def _to_paper_record(self, item: dict[str, Any], server: str) -> PaperRecord:
        doi = normalize_text(item.get("doi", ""))
        title = normalize_text(item.get("title", ""))
        abstract = normalize_text(item.get("abstract", ""))
        date = normalize_text(item.get("date", ""))
        publication_year = date[:4] if len(date) >= 4 else None
        jats_xml_path = normalize_text(
            item.get("jats_xml_path")
            or item.get("jats xml path")
            or item.get("jatsxml")
            or item.get("jats_path")
            or ""
        )
        published = normalize_text(item.get("published", ""))
        return PaperRecord(
            pmid="",
            pmcid=None,
            doi=doi or None,
            title=title,
            abstract=abstract,
            journal=f"{server} preprint",
            publication_year=publication_year,
            authors=self._split_authors(str(item.get("authors", "") or "")),
            mesh_terms=[],
            has_pmc_fulltext=False,
            retrieval_source=server,
            preprint_server=server,
            preprint_category=normalize_text(item.get("category", "")) or None,
            jats_xml_path=jats_xml_path or None,
            published_doi=published or None,
        )
