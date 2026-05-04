from __future__ import annotations

import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path
from threading import Lock
from typing import Any

from .biorxiv_client import BiorxivClient
from .models import CachedArtifactRef, FullTextRecord, PaperCacheEntry, PaperRecord
from .pubmed_client import PubMedClient
from .sciverse_client import SciverseClient
from .utils import (
    clean_crossref_landing_text,
    compute_paper_cache_key,
    ensure_dir,
    extract_html_text,
    is_valid_crossref_landing_text,
    load_json,
    normalize_text,
    render_markdown_document,
    utc_now_iso,
    write_json,
)


class PDFParserToolkit:
    def __init__(self, config: dict[str, Any]):
        # Default ON: PDF parsing is the only way to recover full text for
        # publisher PDFs that have neither PMC XML nor preprint JATS. Without
        # it ~50% of kept papers degrade to abstract-only. Backends are tried
        # in order; failures are swallowed so a missing/broken backend just
        # falls through to the next one.
        self.enabled = config.get("enabled", True)
        self.backends = config.get("backends", ["pdfplumber", "pypdf", "pymupdf"])

    def parse(self, pdf_path: str) -> dict[str, Any]:
        if not self.enabled:
            return {"status": "disabled", "pdf_path": pdf_path, "backends": self.backends}
        errors = []
        for backend in self.backends:
            try:
                if backend == "pdfplumber":
                    import pdfplumber  # type: ignore

                    with pdfplumber.open(pdf_path) as pdf:
                        text = "\n".join((page.extract_text() or "") for page in pdf.pages)
                    return {"status": "ok", "backend": backend, "text": normalize_text(text)}
                if backend in {"pymupdf", "fitz"}:
                    import fitz  # type: ignore

                    doc = fitz.open(pdf_path)
                    text = "\n".join(page.get_text("text") for page in doc)
                    return {"status": "ok", "backend": "pymupdf", "text": normalize_text(text)}
                if backend == "pypdf":
                    from pypdf import PdfReader  # type: ignore

                    reader = PdfReader(pdf_path)
                    text = "\n".join((page.extract_text() or "") for page in reader.pages)
                    return {"status": "ok", "backend": backend, "text": normalize_text(text)}
            except Exception as exc:
                errors.append(f"{backend}:{exc}")
        return {"status": "error", "pdf_path": pdf_path, "errors": errors}


class PaperCache:
    def __init__(self, config: dict[str, Any] | None = None):
        cfg = dict(config or {})
        self.enabled = bool(cfg.get("enabled", False))
        self.cache_dir = Path(cfg.get("cache_dir") or "paper_cache")
        self.store_pdf = bool(cfg.get("store_pdf", True))
        self.store_xml = bool(cfg.get("store_xml", True))
        self.store_text = bool(cfg.get("store_text", True))
        self.store_landing_page = bool(cfg.get("store_landing_page", True))
        self.lock = Lock()
        self.stats = {
            "cache_hits": 0,
            "cache_misses": 0,
            "pdf_cache_hits": 0,
            "xml_cache_hits": 0,
            "text_cache_hits": 0,
            "network_fetches_avoided": 0,
        }
        if self.enabled:
            self.index_dir = ensure_dir(self.cache_dir / "index")
            self.pdf_dir = ensure_dir(self.cache_dir / "artifacts" / "pdf")
            self.xml_dir = ensure_dir(self.cache_dir / "artifacts" / "xml")
            self.text_dir = ensure_dir(self.cache_dir / "artifacts" / "text")
        else:
            self.index_dir = self.cache_dir / "index"
            self.pdf_dir = self.cache_dir / "artifacts" / "pdf"
            self.xml_dir = self.cache_dir / "artifacts" / "xml"
            self.text_dir = self.cache_dir / "artifacts" / "text"

    def key_for_paper(self, paper: PaperRecord) -> str:
        return compute_paper_cache_key(doi=paper.doi, pmcid=paper.pmcid, pmid=paper.pmid, title=paper.title)

    def _slug(self, cache_key: str) -> str:
        return compute_paper_cache_key(title=cache_key)

    def _slug_from_key(self, cache_key: str) -> str:
        from .utils import cache_slug

        return cache_slug(cache_key)

    def manifest_path(self, cache_key: str) -> Path:
        return self.index_dir / f"{self._slug_from_key(cache_key)}.json"

    def artifact_path(self, cache_key: str, artifact_type: str, suffix: str) -> Path:
        slug = self._slug_from_key(cache_key)
        if artifact_type == "pdf":
            return self.pdf_dir / f"{slug}{suffix}"
        if artifact_type == "xml":
            return self.xml_dir / f"{slug}{suffix}"
        return self.text_dir / f"{slug}{suffix}"

    def load_entry(self, cache_key: str) -> PaperCacheEntry | None:
        if not self.enabled:
            return None
        path = self.manifest_path(cache_key)
        if not path.exists():
            return None
        payload = load_json(path)
        artifacts = [CachedArtifactRef(**item) for item in payload.get("artifacts", [])]
        return PaperCacheEntry(
            cache_key=payload.get("cache_key", cache_key),
            doi=payload.get("doi"),
            pmid=payload.get("pmid"),
            pmcid=payload.get("pmcid"),
            title=payload.get("title", ""),
            source=payload.get("source", ""),
            artifacts=artifacts,
            updated_at=payload.get("updated_at", ""),
        )

    def save_entry(self, entry: PaperCacheEntry) -> None:
        if not self.enabled:
            return
        with self.lock:
            write_json(self.manifest_path(entry.cache_key), asdict(entry))

    def get_artifact(self, cache_key: str, artifact_type: str) -> CachedArtifactRef | None:
        entry = self.load_entry(cache_key)
        if entry is None:
            return None
        for artifact in entry.artifacts:
            if artifact.artifact_type == artifact_type and Path(artifact.path).exists():
                self._record_hit(artifact_type)
                return artifact
        return None

    def put_artifact(
        self,
        cache_key: str,
        paper: PaperRecord,
        *,
        artifact_type: str,
        path: Path,
        source: str,
        parse_status: str = "ok",
        parser_backend: str | None = None,
    ) -> CachedArtifactRef:
        artifact = CachedArtifactRef(
            artifact_type=artifact_type,
            path=str(path),
            source=source,
            parse_status=parse_status,
            parser_backend=parser_backend,
        )
        if not self.enabled:
            return artifact
        entry = self.load_entry(cache_key) or PaperCacheEntry(
            cache_key=cache_key,
            doi=paper.doi,
            pmid=paper.pmid or None,
            pmcid=paper.pmcid,
            title=paper.title,
            source=paper.retrieval_source,
            artifacts=[],
            updated_at=utc_now_iso(),
        )
        kept = [item for item in entry.artifacts if item.artifact_type != artifact_type]
        kept.append(artifact)
        entry.artifacts = kept
        entry.updated_at = utc_now_iso()
        self.save_entry(entry)
        return artifact

    def _record_hit(self, artifact_type: str) -> None:
        with self.lock:
            self.stats["cache_hits"] += 1
            self.stats["network_fetches_avoided"] += 1
            if artifact_type == "pdf":
                self.stats["pdf_cache_hits"] += 1
            elif artifact_type.endswith("xml") or artifact_type == "xml":
                self.stats["xml_cache_hits"] += 1
            else:
                self.stats["text_cache_hits"] += 1

    def record_miss(self) -> None:
        with self.lock:
            self.stats["cache_misses"] += 1

    def stats_snapshot(self) -> dict[str, int]:
        with self.lock:
            return dict(self.stats)


class PMCFullTextFetcher:
    def __init__(
        self,
        config: dict[str, Any],
        pubmed_client: PubMedClient,
        biorxiv_client: BiorxivClient | None = None,
    ):
        self.pubmed_client = pubmed_client
        self.biorxiv_client = biorxiv_client
        self.max_workers = max(int(config.get("max_workers", 4)), 1)
        self.prefer_pmc = bool(config.get("prefer_pmc", True))
        self.prefer_preprint_jats = bool(config.get("prefer_preprint_jats", True))
        self.prefer_preprint_pdf = bool(config.get("prefer_preprint_pdf", True))
        self.abstract_fallback = bool(config.get("abstract_fallback", True))
        self.prefer_crossref_landing_page = bool(config.get("prefer_crossref_landing_page", True))
        self.crossref_landing_page_timeout = float(config.get("crossref_landing_page_timeout", 20.0))
        self.crossref_landing_page_min_chars = max(int(config.get("crossref_landing_page_min_chars", 1200)), 200)
        self.pdf_parser = PDFParserToolkit(config.get("pdf_parsing", {}))
        self.cache = PaperCache(config.get("paper_cache", {}))
        self.sciverse = SciverseClient(config.get("sciverse", {}))

    def fetch(self, papers: list[PaperRecord]) -> tuple[list[FullTextRecord], dict[str, Any]]:
        pmc_map = self.pubmed_client.map_pubmed_to_pmc([paper.pmid for paper in papers if paper.pmid]) if self.prefer_pmc else {}
        for paper in papers:
            if not paper.pmcid and pmc_map.get(paper.pmid):
                paper.pmcid = pmc_map[paper.pmid][0]
                paper.has_pmc_fulltext = True
        records: list[FullTextRecord] = []
        fulltext_count = 0
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {executor.submit(self._fetch_one, paper): paper for paper in papers}
            for future in as_completed(futures):
                record = future.result()
                if record.has_fulltext:
                    fulltext_count += 1
                records.append(record)
        stats = {
            "input_papers": len(papers),
            "fulltext_records": fulltext_count,
            "abstract_only_records": len(records) - fulltext_count,
            **self.cache.stats_snapshot(),
        }
        return sorted(records, key=lambda r: r.doc_id), stats

    def _fetch_one(self, paper: PaperRecord) -> FullTextRecord:
        doc_id = f"PMID:{paper.pmid}" if paper.pmid else (paper.doi or paper.title[:32])
        cache_key = self.cache.key_for_paper(paper)
        note = "fulltext_unavailable"
        if self.prefer_preprint_jats and self.biorxiv_client is not None and paper.jats_xml_path:
            try:
                record = self._load_cached_structured_record(cache_key, paper, doc_id, "jats_text")
                if record is not None:
                    return record
                xml_artifact = self.cache.get_artifact(cache_key, "jats_xml")
                if xml_artifact is not None:
                    xml_text = Path(xml_artifact.path).read_text(encoding="utf-8")
                else:
                    self.cache.record_miss()
                    xml_text = self.biorxiv_client.fetch_jats_xml(paper.jats_xml_path)
                    xml_path = self.cache.artifact_path(cache_key, "xml", ".jats.xml")
                    xml_path.write_text(xml_text, encoding="utf-8")
                    self.cache.put_artifact(cache_key, paper, artifact_type="jats_xml", path=xml_path, source=paper.preprint_server or paper.retrieval_source)
                parsed = parse_pmc_xml(xml_text)
                preprint_source = f"{paper.preprint_server or paper.retrieval_source}_jats"
                return self._persist_structured_record(cache_key, paper, doc_id, parsed, source=preprint_source, artifact_type="jats_text", raw_path=paper.jats_xml_path)
            except Exception as exc:
                note = f"preprint_jats_failed={exc}"
        if self.prefer_preprint_pdf and self.biorxiv_client is not None and paper.retrieval_source in {"biorxiv", "medrxiv"}:
            try:
                record = self._load_cached_pdf_record(cache_key, paper, doc_id)
                if record is not None:
                    return record
                pdf_artifact = self.cache.get_artifact(cache_key, "pdf")
                if pdf_artifact is not None:
                    pdf_path = Path(pdf_artifact.path)
                else:
                    self.cache.record_miss()
                    pdf_path = self.cache.artifact_path(cache_key, "pdf", ".pdf")
                    self.biorxiv_client.download_pdf(paper, destination_path=str(pdf_path))
                    self.cache.put_artifact(cache_key, paper, artifact_type="pdf", path=pdf_path, source=paper.preprint_server or paper.retrieval_source)
                parsed_pdf = self.pdf_parser.parse(str(pdf_path))
                if parsed_pdf.get("status") == "ok" and normalize_text(parsed_pdf.get("text", "")):
                    pdf_text = normalize_text(parsed_pdf.get("text", ""))
                    md_text = render_markdown_document(paper.title, paper.abstract, [{"section": "PDF", "text": pdf_text}], pdf_text)
                    txt_path = self.cache.artifact_path(cache_key, "text", ".pdf.txt")
                    md_path = self.cache.artifact_path(cache_key, "text", ".pdf.md")
                    txt_path.write_text(pdf_text, encoding="utf-8")
                    md_path.write_text(md_text, encoding="utf-8")
                    self.cache.put_artifact(cache_key, paper, artifact_type="pdf_text", path=txt_path, source=paper.preprint_server or paper.retrieval_source, parser_backend=parsed_pdf.get("backend"))
                    self.cache.put_artifact(cache_key, paper, artifact_type="pdf_md", path=md_path, source=paper.preprint_server or paper.retrieval_source, parser_backend=parsed_pdf.get("backend"))
                    return FullTextRecord(
                        doc_id=doc_id,
                        pmid=paper.pmid or None,
                        pmcid=paper.pmcid,
                        doi=paper.doi,
                        title=paper.title,
                        journal=paper.journal,
                        publication_year=paper.publication_year,
                        source=f"{paper.preprint_server or paper.retrieval_source}_pdf",
                        has_fulltext=True,
                        abstract=paper.abstract,
                        text=pdf_text,
                        sections=[{"section": "PDF", "text": pdf_text}],
                        raw_path=str(pdf_path),
                        parse_status="ok",
                        cache_hit=pdf_artifact is not None,
                        cache_key=cache_key,
                    )
                note = f"preprint_pdf_parse_failed={parsed_pdf}"
            except Exception as exc:
                note = f"preprint_pdf_failed={exc}"
        if self.prefer_pmc and paper.pmcid:
            try:
                record = self._load_cached_structured_record(cache_key, paper, doc_id, "pmc_text")
                if record is not None:
                    return record
                xml_artifact = self.cache.get_artifact(cache_key, "pmc_xml")
                if xml_artifact is not None:
                    xml_text = Path(xml_artifact.path).read_text(encoding="utf-8")
                else:
                    self.cache.record_miss()
                    xml_text = self.pubmed_client.efetch_pmc_xml(paper.pmcid)
                    xml_path = self.cache.artifact_path(cache_key, "xml", ".pmc.xml")
                    xml_path.write_text(xml_text, encoding="utf-8")
                    self.cache.put_artifact(cache_key, paper, artifact_type="pmc_xml", path=xml_path, source="pmc")
                parsed = parse_pmc_xml(xml_text)
                return self._persist_structured_record(cache_key, paper, doc_id, parsed, source="pmc_xml", artifact_type="pmc_text")
            except Exception as exc:
                note = f"pmc_fetch_failed={exc}"
        if self.sciverse.enabled and normalize_text(paper.title):
            try:
                record = self._load_cached_pdf_record(
                    cache_key,
                    paper,
                    doc_id,
                    text_artifact_type="sciverse_pdf_text",
                    pdf_artifact_type="sciverse_pdf",
                    source="sciverse_pdf",
                )
                if record is not None:
                    return record
                pdf_artifact = self.cache.get_artifact(cache_key, "sciverse_pdf")
                if pdf_artifact is not None:
                    pdf_path = Path(pdf_artifact.path)
                    channel = "cache"
                    match_reason = "cache_hit"
                else:
                    self.cache.record_miss()
                    pdf_path = self.cache.artifact_path(cache_key, "pdf", ".sciverse.pdf")
                    result = self.sciverse.search_and_download(paper, str(pdf_path))
                    if not result.get("ok", False):
                        raise RuntimeError(str(result.get("error", "unknown")))
                    channel = normalize_text(result.get("channel", "")) or "sciverse"
                    match_reason = normalize_text(result.get("match_reason", "")) or "matched"
                    self.cache.put_artifact(
                        cache_key,
                        paper,
                        artifact_type="sciverse_pdf",
                        path=pdf_path,
                        source=f"sciverse:{channel}",
                    )
                parsed_pdf = self.pdf_parser.parse(str(pdf_path))
                if parsed_pdf.get("status") == "ok" and normalize_text(parsed_pdf.get("text", "")):
                    pdf_text = normalize_text(parsed_pdf.get("text", ""))
                    md_text = render_markdown_document(paper.title, paper.abstract, [{"section": "PDF", "text": pdf_text}], pdf_text)
                    txt_path = self.cache.artifact_path(cache_key, "text", ".sciverse_pdf.txt")
                    md_path = self.cache.artifact_path(cache_key, "text", ".sciverse_pdf.md")
                    txt_path.write_text(pdf_text, encoding="utf-8")
                    md_path.write_text(md_text, encoding="utf-8")
                    self.cache.put_artifact(
                        cache_key,
                        paper,
                        artifact_type="sciverse_pdf_text",
                        path=txt_path,
                        source=f"sciverse:{channel}",
                        parser_backend=parsed_pdf.get("backend"),
                    )
                    self.cache.put_artifact(
                        cache_key,
                        paper,
                        artifact_type="sciverse_pdf_md",
                        path=md_path,
                        source=f"sciverse:{channel}",
                        parser_backend=parsed_pdf.get("backend"),
                    )
                    return FullTextRecord(
                        doc_id=doc_id,
                        pmid=paper.pmid or None,
                        pmcid=paper.pmcid,
                        doi=paper.doi,
                        title=paper.title,
                        journal=paper.journal,
                        publication_year=paper.publication_year,
                        source="sciverse_pdf",
                        has_fulltext=True,
                        abstract=paper.abstract,
                        text=pdf_text,
                        sections=[{"section": "PDF", "text": pdf_text}],
                        raw_path=str(pdf_path),
                        parse_status="ok",
                        notes=f"channel={channel}; match={match_reason}",
                        cache_hit=pdf_artifact is not None,
                        cache_key=cache_key,
                    )
                note = f"sciverse_pdf_parse_failed={parsed_pdf}"
            except Exception as exc:
                note = f"sciverse_failed={exc}"
        if self.prefer_crossref_landing_page and paper.retrieval_source == "crossref" and paper.landing_page_url:
            try:
                cached_text = self.cache.get_artifact(cache_key, "landing_text")
                if cached_text is not None:
                    page_text = Path(cached_text.path).read_text(encoding="utf-8")
                    if is_valid_crossref_landing_text(page_text, min_chars=self.crossref_landing_page_min_chars, title=paper.title):
                        cleaned_text = clean_crossref_landing_text(page_text)
                        return FullTextRecord(
                            doc_id=doc_id,
                            pmid=paper.pmid or None,
                            pmcid=paper.pmcid,
                            doi=paper.doi,
                            title=paper.title,
                            journal=paper.journal,
                            publication_year=paper.publication_year,
                            source="crossref_landing_page",
                            has_fulltext=True,
                            abstract=paper.abstract,
                            text=cleaned_text,
                            sections=[{"section": "LandingPage", "text": cleaned_text}],
                            raw_path=paper.landing_page_url,
                            parse_status="ok",
                            cache_hit=True,
                            cache_key=cache_key,
                        )
                self.cache.record_miss()
                page_text = self._fetch_crossref_landing_page_text(paper.landing_page_url, title=paper.title)
                if page_text:
                    txt_path = self.cache.artifact_path(cache_key, "text", ".landing.txt")
                    md_path = self.cache.artifact_path(cache_key, "text", ".landing.md")
                    txt_path.write_text(page_text, encoding="utf-8")
                    md_path.write_text(render_markdown_document(paper.title, paper.abstract, [{"section": "LandingPage", "text": page_text}], page_text), encoding="utf-8")
                    self.cache.put_artifact(cache_key, paper, artifact_type="landing_text", path=txt_path, source="crossref")
                    self.cache.put_artifact(cache_key, paper, artifact_type="landing_md", path=md_path, source="crossref")
                    return FullTextRecord(
                        doc_id=doc_id,
                        pmid=paper.pmid or None,
                        pmcid=paper.pmcid,
                        doi=paper.doi,
                        title=paper.title,
                        journal=paper.journal,
                        publication_year=paper.publication_year,
                        source="crossref_landing_page",
                        has_fulltext=True,
                        abstract=paper.abstract,
                        text=page_text,
                        sections=[{"section": "LandingPage", "text": page_text}],
                        raw_path=paper.landing_page_url,
                        parse_status="ok",
                        cache_key=cache_key,
                    )
                note = "crossref_landing_page_rejected_or_too_short"
            except Exception as exc:
                note = f"crossref_landing_page_failed={exc}"
        if self.abstract_fallback:
            fallback_source = (
                f"{paper.preprint_server or paper.retrieval_source}_abstract"
                if paper.retrieval_source in {"biorxiv", "medrxiv"}
                else f"{paper.retrieval_source or 'unknown'}_abstract"
            )
            return FullTextRecord(
                doc_id=doc_id,
                pmid=paper.pmid or None,
                pmcid=paper.pmcid,
                doi=paper.doi,
                title=paper.title,
                journal=paper.journal,
                publication_year=paper.publication_year,
                source=fallback_source,
                has_fulltext=False,
                abstract=paper.abstract,
                text=paper.abstract,
                sections=[{"section": "Abstract", "text": paper.abstract}] if paper.abstract else [],
                parse_status="fallback",
                notes=note,
                cache_key=cache_key,
            )
        return FullTextRecord(
            doc_id=doc_id,
            pmid=paper.pmid or None,
            pmcid=paper.pmcid,
            doi=paper.doi,
            title=paper.title,
            journal=paper.journal,
            publication_year=paper.publication_year,
            source="unavailable",
            has_fulltext=False,
            parse_status="missing",
            notes=note,
            cache_key=cache_key,
        )

    def _fetch_crossref_landing_page_text(self, url: str, title: str = "") -> str:
        response = self.pubmed_client.session.get(url, timeout=self.crossref_landing_page_timeout, headers={"Accept": "text/html,application/xhtml+xml"})
        response.raise_for_status()
        content_type = normalize_text(response.headers.get("content-type", "")).lower()
        if "html" not in content_type and "xml" not in content_type:
            return ""
        text = clean_crossref_landing_text(response.text)
        return text if is_valid_crossref_landing_text(text, min_chars=self.crossref_landing_page_min_chars, title=title) else ""

    def _load_cached_structured_record(self, cache_key: str, paper: PaperRecord, doc_id: str, artifact_type: str) -> FullTextRecord | None:
        artifact = self.cache.get_artifact(cache_key, artifact_type)
        if artifact is None:
            return None
        payload = load_json(artifact.path)
        return FullTextRecord(
            doc_id=doc_id,
            pmid=paper.pmid or None,
            pmcid=paper.pmcid,
            doi=paper.doi,
            title=payload.get("title") or paper.title,
            journal=paper.journal,
            publication_year=paper.publication_year,
            source=payload.get("source") or artifact.source,
            has_fulltext=True,
            abstract=payload.get("abstract", paper.abstract),
            text=payload.get("text", ""),
            sections=payload.get("sections", []),
            raw_path=payload.get("raw_path"),
            parse_status=payload.get("parse_status", "ok"),
            cache_hit=True,
            cache_key=cache_key,
        )

    def _persist_structured_record(
        self,
        cache_key: str,
        paper: PaperRecord,
        doc_id: str,
        parsed: dict[str, Any],
        *,
        source: str,
        artifact_type: str,
        raw_path: str | None = None,
    ) -> FullTextRecord:
        payload = {
            "title": parsed.get("title") or paper.title,
            "abstract": parsed.get("abstract", paper.abstract),
            "text": parsed.get("text", ""),
            "sections": parsed.get("sections", []),
            "source": source,
            "raw_path": raw_path,
            "parse_status": "ok",
        }
        json_path = self.cache.artifact_path(cache_key, "text", f".{artifact_type}.json")
        txt_path = self.cache.artifact_path(cache_key, "text", f".{artifact_type}.txt")
        md_path = self.cache.artifact_path(cache_key, "text", f".{artifact_type}.md")
        write_json(json_path, payload)
        txt_path.write_text(normalize_text(payload.get("text", "")), encoding="utf-8")
        md_path.write_text(
            render_markdown_document(payload.get("title", paper.title), payload.get("abstract", paper.abstract), payload.get("sections", []), payload.get("text", "")),
            encoding="utf-8",
        )
        self.cache.put_artifact(cache_key, paper, artifact_type=artifact_type, path=json_path, source=source)
        self.cache.put_artifact(cache_key, paper, artifact_type=f"{artifact_type}_txt", path=txt_path, source=source)
        self.cache.put_artifact(cache_key, paper, artifact_type=f"{artifact_type}_md", path=md_path, source=source)
        return FullTextRecord(
            doc_id=doc_id,
            pmid=paper.pmid or None,
            pmcid=paper.pmcid,
            doi=paper.doi,
            title=payload.get("title") or paper.title,
            journal=paper.journal,
            publication_year=paper.publication_year,
            source=source,
            has_fulltext=True,
            abstract=payload.get("abstract", paper.abstract),
            text=payload.get("text", ""),
            sections=payload.get("sections", []),
            raw_path=raw_path,
            parse_status="ok",
            cache_key=cache_key,
        )

    def _load_cached_pdf_record(
        self,
        cache_key: str,
        paper: PaperRecord,
        doc_id: str,
        *,
        text_artifact_type: str = "pdf_text",
        pdf_artifact_type: str = "pdf",
        source: str | None = None,
    ) -> FullTextRecord | None:
        artifact = self.cache.get_artifact(cache_key, text_artifact_type)
        pdf_artifact = self.cache.get_artifact(cache_key, pdf_artifact_type)
        if artifact is None:
            return None
        pdf_text = Path(artifact.path).read_text(encoding="utf-8")
        return FullTextRecord(
            doc_id=doc_id,
            pmid=paper.pmid or None,
            pmcid=paper.pmcid,
            doi=paper.doi,
            title=paper.title,
            journal=paper.journal,
            publication_year=paper.publication_year,
            source=source or f"{paper.preprint_server or paper.retrieval_source}_pdf",
            has_fulltext=True,
            abstract=paper.abstract,
            text=pdf_text,
            sections=[{"section": "PDF", "text": pdf_text}],
            raw_path=pdf_artifact.path if pdf_artifact is not None else None,
            parse_status="ok",
            cache_hit=True,
            cache_key=cache_key,
        )


def parse_pmc_xml(xml_text: str) -> dict[str, Any]:
    root = ET.fromstring(xml_text)
    title = normalize_text(root.findtext(".//article-title") or "")
    abstract_parts = []
    for item in root.findall(".//abstract//p"):
        text = normalize_text(" ".join(item.itertext()))
        if text:
            abstract_parts.append(text)
    sections = []
    for sec in root.findall(".//body//sec"):
        sec_title = normalize_text(sec.findtext("./title") or "Body") or "Body"
        paras = []
        for p in sec.findall("./p"):
            text = normalize_text(" ".join(p.itertext()))
            if text:
                paras.append(text)
        if paras:
            sections.append({"section": sec_title, "text": "\n".join(paras)})
    if not sections:
        body_text = []
        for p in root.findall(".//body//p"):
            text = normalize_text(" ".join(p.itertext()))
            if text:
                body_text.append(text)
        if body_text:
            sections.append({"section": "Body", "text": "\n".join(body_text)})
    combined = "\n\n".join(item["text"] for item in sections if item.get("text"))
    return {
        "title": title,
        "abstract": "\n".join(abstract_parts),
        "sections": sections,
        "text": combined,
    }
