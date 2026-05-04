from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

from .biorxiv_client import BiorxivClient
from .crossref_client import CrossrefClient
from .embeddings import SapBERTScorer
from .models import PaperRecord
from .pubmed_client import PubMedClient
from .utils import normalize_keyword


class LiteratureScorer:
    def __init__(self, config: dict[str, Any]):
        self.sapbert = SapBERTScorer(config.get("sapbert", {}))
        self.impact_factors = {
            normalize_keyword(journal): float(score)
            for journal, score in config.get("journal_impact_factors", {}).items()
        }
        self.semantic_weight = float(config.get("semantic_weight", 0.7))
        self.impact_weight = float(config.get("impact_weight", 0.3))
        self.threshold = float(config.get("score_threshold", 0.2))

    def score_paper(self, matched_keywords: list[str], paper: PaperRecord) -> PaperRecord:
        semantic_scores = [self.sapbert.score(keyword, paper) for keyword in matched_keywords]
        semantic_score = max(semantic_scores) if semantic_scores else 0.0
        journal_score = self.impact_factors.get(normalize_keyword(paper.journal), 0.0)
        impact_score = min(journal_score / 100.0, 1.0) if journal_score > 0 else 0.0
        final_score = self.semantic_weight * semantic_score + self.impact_weight * impact_score
        paper.semantic_score = round(semantic_score, 4)
        paper.impact_score = round(impact_score, 4)
        paper.final_score = round(final_score, 4)
        paper.kept = final_score >= self.threshold
        return paper


class LiteratureRetrievalEngine:
    def __init__(
        self,
        config: dict[str, Any],
        pubmed_client: PubMedClient,
        biorxiv_client: BiorxivClient | None = None,
        biorxiv_config: dict[str, Any] | None = None,
        crossref_client: CrossrefClient | None = None,
        crossref_config: dict[str, Any] | None = None,
    ):
        self.pubmed_client = pubmed_client
        self.biorxiv_client = biorxiv_client
        self.crossref_client = crossref_client
        self.max_workers = max(int(config.get("max_workers", 8)), 1)
        self.retmax_per_keyword = max(int(config.get("retmax_per_keyword", 20)), 1)
        self.query_template = config.get(
            "query_template",
            '("{keyword}"[Title/Abstract] OR "{keyword}"[MeSH Terms])',
        )
        self.mindate = config.get("mindate")
        self.maxdate = config.get("maxdate")
        self.sleep_seconds = float(config.get("sleep_seconds", 0.0))
        self.allowed_journals = {normalize_keyword(name) for name in config.get("journal_allowlist", [])}
        self.related_expand_limit = max(int(config.get("related_expand_limit", 0)), 0)
        self.related_per_seed = max(int(config.get("related_per_seed", 3)), 0)
        bio_cfg = dict(biorxiv_config or {})
        self.biorxiv_enabled = bool(bio_cfg.get("enabled", False)) and biorxiv_client is not None
        self.biorxiv_servers = [normalize_keyword(item) for item in bio_cfg.get("servers", ["biorxiv"]) if normalize_keyword(item)]
        self.biorxiv_category = str(bio_cfg.get("category", "") or "").strip()
        cross_cfg = dict(crossref_config or {})
        query_mode = str(config.get("crossref_query_mode", cross_cfg.get("query_mode", "bibliographic")) or "bibliographic")
        if crossref_client is not None:
            crossref_client.query_mode = query_mode
        self.crossref_enabled = bool(cross_cfg.get("enabled", False)) and crossref_client is not None

    def build_query(self, keyword: str) -> str:
        safe_keyword = keyword.replace('"', "")
        return self.query_template.format(keyword=safe_keyword)

    def retrieve(self, keywords: list[str]) -> tuple[list[PaperRecord], dict[str, Any]]:
        deduped: dict[str, PaperRecord] = {}
        query_to_count: dict[str, int] = {keyword: 0 for keyword in keywords}
        source_result_counts: dict[str, int] = {}
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures: dict[Any, tuple[str, str]] = {}
            for keyword in keywords:
                futures[executor.submit(self._fetch_pubmed_keyword_batch, keyword)] = (keyword, "pubmed")
                if self.biorxiv_enabled:
                    for server in self.biorxiv_servers:
                        futures[executor.submit(self._fetch_biorxiv_keyword_batch, keyword, server)] = (keyword, server)
                if self.crossref_enabled:
                    futures[executor.submit(self._fetch_crossref_keyword_batch, keyword)] = (keyword, "crossref")
            for future in as_completed(futures):
                keyword, source_label = futures[future]
                try:
                    papers = future.result()
                except Exception as exc:
                    source_result_counts[f"{keyword} [{source_label}]"] = -1
                    print(f"[WARN] Retrieval failed for '{keyword}' via {source_label}: {exc}")
                    continue
                query_to_count[keyword] += len(papers)
                source_result_counts[f"{keyword} [{source_label}]"] = len(papers)
                for paper in papers:
                    if self.allowed_journals and normalize_keyword(paper.journal) not in self.allowed_journals:
                        continue
                    dedupe_key = paper.doi or paper.pmid or normalize_keyword(paper.title)
                    if not dedupe_key:
                        continue
                    existing = deduped.get(dedupe_key)
                    if existing is None:
                        if keyword not in paper.matched_keywords:
                            paper.matched_keywords.append(keyword)
                        deduped[dedupe_key] = paper
                    else:
                        self._merge_papers(existing, paper, keyword)
        if self.related_expand_limit > 0 and deduped:
            seed_pmids = [paper.pmid for paper in list(deduped.values())[: self.related_expand_limit] if paper.pmid]
            related_map = self.pubmed_client.elink_related(seed_pmids, max_links=self.related_per_seed)
            extra_pmids = []
            for paper in deduped.values():
                related = related_map.get(paper.pmid, [])
                paper.related_pmids = related
                for related_pmid in related:
                    if all(related_pmid != existing.pmid for existing in deduped.values()):
                        extra_pmids.append(related_pmid)
            if extra_pmids:
                for extra in self.pubmed_client.efetch_pubmed_xml(
                    extra_pmids[: self.related_expand_limit * self.related_per_seed]
                ):
                    key = extra.doi or extra.pmid
                    if key and key not in deduped:
                        extra.retrieval_source = "pubmed_related"
                        deduped[key] = extra
        stats = {
            "keyword_count": len(keywords),
            "retrieved_unique_papers": len(deduped),
            "query_result_counts": query_to_count,
            "query_result_counts_by_source": source_result_counts,
            "retrieval_source_breakdown": self._count_by_source(deduped.values()),
        }
        return list(deduped.values()), stats

    @staticmethod
    def _count_by_source(papers: Any) -> dict[str, int]:
        counts: dict[str, int] = {}
        for paper in papers:
            label = str(getattr(paper, "retrieval_source", "") or "unknown")
            counts[label] = counts.get(label, 0) + 1
        return counts

    @staticmethod
    def _merge_papers(existing: PaperRecord, paper: PaperRecord, keyword: str) -> None:
        if keyword not in existing.matched_keywords:
            existing.matched_keywords.append(keyword)
        for query in paper.source_queries:
            if query not in existing.source_queries:
                existing.source_queries.append(query)
        for author in paper.authors:
            if author not in existing.authors:
                existing.authors.append(author)
        for term in paper.mesh_terms:
            if term not in existing.mesh_terms:
                existing.mesh_terms.append(term)
        if not existing.abstract and paper.abstract:
            existing.abstract = paper.abstract
        if not existing.title and paper.title:
            existing.title = paper.title
        if not existing.journal and paper.journal:
            existing.journal = paper.journal
        if not existing.publication_year and paper.publication_year:
            existing.publication_year = paper.publication_year
        if not existing.doi and paper.doi:
            existing.doi = paper.doi
        if not existing.pmcid and paper.pmcid:
            existing.pmcid = paper.pmcid
            existing.has_pmc_fulltext = bool(paper.pmcid)
        if not existing.jats_xml_path and paper.jats_xml_path:
            existing.jats_xml_path = paper.jats_xml_path
        if not existing.preprint_server and paper.preprint_server:
            existing.preprint_server = paper.preprint_server
        if not existing.preprint_category and paper.preprint_category:
            existing.preprint_category = paper.preprint_category
        if not existing.published_doi and paper.published_doi:
            existing.published_doi = paper.published_doi
        if not existing.landing_page_url and paper.landing_page_url:
            existing.landing_page_url = paper.landing_page_url

    def _fetch_pubmed_keyword_batch(self, keyword: str) -> list[PaperRecord]:
        query = self.build_query(keyword)
        papers = self.pubmed_client.fetch_papers(
            query=query,
            retmax=self.retmax_per_keyword,
            mindate=self.mindate,
            maxdate=self.maxdate,
        )
        if self.sleep_seconds > 0:
            time.sleep(self.sleep_seconds)
        return papers

    def _fetch_biorxiv_keyword_batch(self, keyword: str, server: str) -> list[PaperRecord]:
        if not self.biorxiv_enabled or self.biorxiv_client is None:
            return []
        papers = self.biorxiv_client.fetch_papers(
            keyword=keyword,
            server=server,
            retmax=self.retmax_per_keyword,
            category=self.biorxiv_category,
        )
        if self.sleep_seconds > 0:
            time.sleep(self.sleep_seconds)
        return papers

    def _fetch_crossref_keyword_batch(self, keyword: str) -> list[PaperRecord]:
        if not self.crossref_enabled or self.crossref_client is None:
            return []
        papers = self.crossref_client.fetch_papers(
            keyword=keyword,
            retmax=self.retmax_per_keyword,
            mindate=self.mindate,
            maxdate=self.maxdate,
        )
        if self.sleep_seconds > 0:
            time.sleep(self.sleep_seconds)
        return papers
