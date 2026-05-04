from __future__ import annotations

import json
import os
import time
import xml.etree.ElementTree as ET
from typing import Any

import requests

from .models import PaperRecord
from .utils import normalize_text

PUBMED_BASE_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"

# NCBI E-utilities accepts POST for esummary/efetch/elink with id in the body.
# Empirically ~200 IDs per call stays well below the server's URL-length limit
# even when we fall back to GET, and avoids 414 errors on large PathVQA /
# PathVQA-like batches that can produce 1000+ candidate PMIDs.
_POST_ID_CHUNK = 200


class PubMedClient:
    def __init__(self, api_key: str | None = None, email: str | None = None, tool: str = "pubmed_graph_workflow"):
        self.api_key = api_key or os.getenv("NCBI_API_KEY") or os.getenv("PUBMED_API_KEY")
        self.email = email or os.getenv("NCBI_EMAIL", "your_email@example.com")
        self.tool = tool
        self.session = requests.Session()

    def _common_params(self, db: str = "pubmed") -> dict[str, str]:
        params = {"db": db, "tool": self.tool, "email": self.email}
        if self.api_key:
            params["api_key"] = self.api_key
        return params

    def _post_ids(self, endpoint: str, ids: list[str], extra: dict[str, str],
                  timeout: float = 60.0) -> list[str]:
        """POST each chunk of `ids` to an E-utilities endpoint.

        Returns a list of response bodies (one per chunk). Caller parses
        each element independently — we cannot concatenate raw XML without
        producing multiple roots.

        NCBI rate limits: 10 req/s with API key, 3 req/s without. We add a
        small inter-chunk sleep and retry on 429 to stay friendly under
        heavy-load paths (e.g. PathVQA overlay pulls 1000+ PMIDs per batch).
        """
        bodies: list[str] = []
        inter_chunk_sleep = 0.15 if self.api_key else 0.35
        for start in range(0, len(ids), _POST_ID_CHUNK):
            if start > 0 and inter_chunk_sleep > 0:
                time.sleep(inter_chunk_sleep)
            batch = ids[start:start + _POST_ID_CHUNK]
            data = {"id": ",".join(batch), **extra}
            backoff = 1.0
            last_exc: Exception | None = None
            for attempt in range(5):
                try:
                    response = self.session.post(
                        f"{PUBMED_BASE_URL}/{endpoint}",
                        data=data,
                        timeout=timeout,
                    )
                    if response.status_code == 429:
                        time.sleep(backoff)
                        backoff = min(backoff * 2, 16.0)
                        continue
                    response.raise_for_status()
                    bodies.append(response.text)
                    last_exc = None
                    break
                except requests.exceptions.RequestException as exc:
                    last_exc = exc
                    time.sleep(backoff)
                    backoff = min(backoff * 2, 16.0)
            else:
                if last_exc is not None:
                    raise last_exc
        return bodies

    def esearch(
        self,
        term: str,
        retmax: int = 20,
        sort: str = "relevance",
        mindate: str | None = None,
        maxdate: str | None = None,
    ) -> dict[str, Any]:
        params = self._common_params("pubmed")
        params.update({"term": term, "retmode": "json", "retmax": str(retmax), "sort": sort})
        if mindate or maxdate:
            params["datetype"] = "pdat"
        if mindate:
            params["mindate"] = mindate
        if maxdate:
            params["maxdate"] = maxdate
        response = self.session.get(f"{PUBMED_BASE_URL}/esearch.fcgi", params=params, timeout=30)
        response.raise_for_status()
        return response.json()

    def esummary(self, pmids: list[str]) -> dict[str, Any]:
        if not pmids:
            return {}
        extra = {**self._common_params("pubmed"), "retmode": "json"}
        merged_uids: list[str] = []
        merged_result: dict[str, Any] = {}
        for body in self._post_ids("esummary.fcgi", pmids, extra, timeout=30.0):
            try:
                payload = json.loads(body)
            except Exception:
                continue
            chunk_result = (payload.get("result") or {})
            for uid in chunk_result.get("uids", []) or []:
                if uid not in merged_uids:
                    merged_uids.append(uid)
                    merged_result[uid] = chunk_result.get(uid)
        return {"result": {"uids": merged_uids, **merged_result}}

    def efetch_pubmed_xml(self, pmids: list[str]) -> list[PaperRecord]:
        if not pmids:
            return []
        extra = {**self._common_params("pubmed"), "retmode": "xml"}
        records: list[PaperRecord] = []
        for body in self._post_ids("efetch.fcgi", pmids, extra, timeout=60.0):
            records.extend(self._parse_pubmed_xml(body))
        return records

    def efetch_pmc_xml(self, pmcid: str) -> str:
        clean_pmcid = pmcid.upper().replace("PMC", "")
        params = self._common_params("pmc")
        params.update({"id": clean_pmcid, "retmode": "xml"})
        response = self.session.get(f"{PUBMED_BASE_URL}/efetch.fcgi", params=params, timeout=60)
        response.raise_for_status()
        return response.text

    def elink_related(self, pmids: list[str], linkname: str = "pubmed_pubmed", max_links: int = 10) -> dict[str, list[str]]:
        if not pmids:
            return {}
        extra = {**self._common_params("pubmed"), "dbfrom": "pubmed", "db": "pubmed",
                 "linkname": linkname, "retmode": "xml"}
        mapping: dict[str, list[str]] = {}
        for body in self._post_ids("elink.fcgi", pmids, extra, timeout=30.0):
            try:
                root = ET.fromstring(body)
            except ET.ParseError:
                continue
            for linkset in root.findall(".//LinkSet"):
                source_id = normalize_text(linkset.findtext("./IdList/Id") or "")
                linked: list[str] = []
                for item in linkset.findall(".//Link/Id"):
                    linked_id = normalize_text(item.text or "")
                    if linked_id and linked_id != source_id and linked_id not in linked:
                        linked.append(linked_id)
                    if len(linked) >= max_links:
                        break
                if source_id:
                    mapping[source_id] = linked
        return mapping

    def map_pubmed_to_pmc(self, pmids: list[str]) -> dict[str, list[str]]:
        if not pmids:
            return {}
        extra = {**self._common_params("pubmed"), "dbfrom": "pubmed", "db": "pmc",
                 "retmode": "xml"}
        mapping: dict[str, list[str]] = {}
        for body in self._post_ids("elink.fcgi", pmids, extra, timeout=30.0):
            try:
                root = ET.fromstring(body)
            except ET.ParseError:
                continue
            for linkset in root.findall(".//LinkSet"):
                source_id = normalize_text(linkset.findtext("./IdList/Id") or "")
                pmc_ids: list[str] = []
                for item in linkset.findall(".//Link/Id"):
                    raw = normalize_text(item.text or "")
                    if raw:
                        value = raw if raw.upper().startswith("PMC") else f"PMC{raw}"
                        if value not in pmc_ids:
                            pmc_ids.append(value)
                if source_id:
                    mapping[source_id] = pmc_ids
        return mapping

    def fetch_papers(self, query: str, retmax: int, mindate: str | None = None, maxdate: str | None = None) -> list[PaperRecord]:
        search_result = self.esearch(query, retmax=retmax, mindate=mindate, maxdate=maxdate)
        id_list = search_result.get("esearchresult", {}).get("idlist", [])
        papers = self.efetch_pubmed_xml(id_list)
        for paper in papers:
            if query not in paper.source_queries:
                paper.source_queries.append(query)
        return papers

    def _parse_pubmed_xml(self, xml_text: str) -> list[PaperRecord]:
        root = ET.fromstring(xml_text)
        records: list[PaperRecord] = []
        for article in root.findall(".//PubmedArticle"):
            pmid = self._findtext(article, ".//MedlineCitation/PMID")
            title = self._flatten_text(article.find(".//ArticleTitle"))
            abstract = self._join_texts(article.findall(".//Abstract/AbstractText"))
            journal = self._findtext(article, ".//Journal/Title")
            year = self._findtext(article, ".//PubDate/Year") or self._findtext(article, ".//PubDate/MedlineDate")
            authors: list[str] = []
            for author in article.findall(".//AuthorList/Author"):
                last = self._findtext(author, "./LastName")
                fore = self._findtext(author, "./ForeName")
                collective = self._findtext(author, "./CollectiveName")
                if collective:
                    authors.append(collective)
                elif last or fore:
                    authors.append(" ".join(part for part in [fore, last] if part))
            doi = None
            pmcid = None
            for article_id in article.findall(".//PubmedData/ArticleIdList/ArticleId"):
                id_type = article_id.attrib.get("IdType")
                value = normalize_text(article_id.text or "")
                if id_type == "doi" and value and doi is None:
                    doi = value
                if id_type == "pmc" and value and pmcid is None:
                    pmcid = value if value.upper().startswith("PMC") else f"PMC{value}"
            mesh_terms = []
            for mesh in article.findall(".//MeshHeadingList/MeshHeading/DescriptorName"):
                label = normalize_text(" ".join(mesh.itertext()))
                if label and label not in mesh_terms:
                    mesh_terms.append(label)
            records.append(
                PaperRecord(
                    pmid=pmid or "",
                    pmcid=pmcid,
                    doi=doi,
                    title=title,
                    abstract=abstract,
                    journal=journal,
                    publication_year=year,
                    authors=authors,
                    mesh_terms=mesh_terms,
                    has_pmc_fulltext=bool(pmcid),
                )
            )
        return records

    @staticmethod
    def _findtext(element: ET.Element | None, path: str) -> str:
        if element is None:
            return ""
        child = element.find(path)
        return normalize_text(child.text if child is not None else "")

    @staticmethod
    def _flatten_text(element: ET.Element | None) -> str:
        if element is None:
            return ""
        return normalize_text(" ".join(element.itertext()))

    @staticmethod
    def _join_texts(elements: list[ET.Element]) -> str:
        parts = [normalize_text(" ".join(elem.itertext())) for elem in elements]
        return "\n".join(part for part in parts if part)
