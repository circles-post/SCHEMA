from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path
from typing import Any

from .models import PaperRecord
from .utils import normalize_keyword, normalize_text, tokenize


def _ensure_path(resolved: str | None, destination: str) -> str:
    """Materialize a downloaded PDF at `destination`.

    sciverse_toolkit.download_paper() may return a cached path that is not the
    caller's requested `destination` (channel=="Cache" returns the original
    cache entry). The PMCFullTextFetcher then tries to parse the file at
    `destination`, which doesn't exist. Here we copy the resolved file to
    `destination` whenever the two diverge so the caller contract holds.
    """
    if not resolved:
        return destination
    resolved_p = Path(resolved)
    destination_p = Path(destination)
    if resolved_p.resolve() == destination_p.resolve():
        return destination
    if not resolved_p.exists():
        return destination
    destination_p.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(resolved_p, destination_p)
    return destination

DEFAULT_SCIVERSE_TOOLKIT_ROOT = os.environ.get("SCIVERSE_DIR", "")


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


class SciverseClient:
    def __init__(self, config: dict[str, Any] | None = None):
        cfg = dict(config or {})
        self.enabled = bool(cfg.get("enabled", False))
        self.toolkit_root = str(cfg.get("toolkit_root") or DEFAULT_SCIVERSE_TOOLKIT_ROOT).strip()
        self.search_top_k = max(int(cfg.get("search_top_k", 5)), 1)
        self.language = normalize_text(cfg.get("language", "")) or None
        self.require_match = bool(cfg.get("require_match", True))
        self.min_title_token_overlap = float(cfg.get("min_title_token_overlap", 0.7))
        # When a DOI is available, skip the title-based sciverse API search and
        # call download_paper(title, doi, ...) directly. The toolkit's channels
        # (Sci-Hub / crossref / elsevier / ...) already resolve by DOI, and this
        # avoids fragile unicode-punctuation title mismatches on papers whose
        # titles contain non-ASCII dashes or mathematical symbols.
        self.prefer_doi_direct = bool(cfg.get("prefer_doi_direct", True))
        self._search_sciverse_papers = None
        self._download_paper = None

    def _ensure_imports(self) -> None:
        toolkit_root = Path(self.toolkit_root).expanduser().resolve()
        if not toolkit_root.exists():
            raise FileNotFoundError(f"Sciverse toolkit root not found: {toolkit_root}")
        toolkit_root_str = str(toolkit_root)
        if toolkit_root_str not in sys.path:
            sys.path.insert(0, toolkit_root_str)
        if self._search_sciverse_papers is None or self._download_paper is None:
            from sciverse_toolkit import download_paper, search_sciverse_papers

            self._search_sciverse_papers = search_sciverse_papers
            self._download_paper = download_paper

    def _title_overlap(self, title_a: str, title_b: str) -> float:
        tokens_a = tokenize(title_a)
        tokens_b = tokenize(title_b)
        if not tokens_a or not tokens_b:
            return 0.0
        return len(tokens_a & tokens_b) / max(len(tokens_a), 1)

    def _select_match(self, paper: PaperRecord, candidates: list[dict[str, Any]]) -> tuple[dict[str, Any] | None, str]:
        target_title = normalize_text(paper.title)
        target_title_key = normalize_keyword(target_title)
        target_doi = _normalize_doi(paper.doi)
        best_candidate: dict[str, Any] | None = None
        best_reason = ""
        best_score = -1.0

        for candidate in candidates:
            if not isinstance(candidate, dict):
                continue
            candidate_title = normalize_text(candidate.get("title", ""))
            if not candidate_title:
                continue
            candidate_title_key = normalize_keyword(candidate_title)
            candidate_doi = _normalize_doi(candidate.get("doi", ""))
            if target_doi and candidate_doi and target_doi == candidate_doi:
                return candidate, "doi_exact"
            if target_title_key and candidate_title_key == target_title_key:
                return candidate, "title_exact"
            overlap = self._title_overlap(target_title, candidate_title)
            if overlap > best_score:
                best_score = overlap
                best_candidate = candidate
                best_reason = f"title_overlap:{overlap:.3f}"

        if best_candidate is not None and best_score >= self.min_title_token_overlap:
            return best_candidate, best_reason
        return (None, f"no_match_above_threshold:{best_score:.3f}") if candidates else (None, "no_candidates")

    def search_and_download(self, paper: PaperRecord, destination_path: str) -> dict[str, Any]:
        if not self.enabled:
            return {"ok": False, "error": "disabled"}
        title = normalize_text(paper.title)
        doi = _normalize_doi(paper.doi)
        if not title and not doi:
            return {"ok": False, "error": "missing_title_and_doi"}

        self._ensure_imports()

        if self.prefer_doi_direct and doi:
            ok, channel, resolved_path = self._download_paper(title or doi, paper.doi, destination_path, verbose=False)
            if ok:
                final_path = _ensure_path(resolved_path, destination_path)
                return {
                    "ok": True,
                    "channel": channel,
                    "pdf_path": final_path,
                    "match_reason": "doi_direct",
                    "matched_title": title,
                    "num_candidates": 0,
                }
            doi_direct_failure = channel or "download_failed"
        else:
            doi_direct_failure = None

        if not title:
            return {"ok": False, "error": f"doi_direct_failed={doi_direct_failure}", "match_reason": "doi_direct"}

        candidates = self._search_sciverse_papers(title, page_size=self.search_top_k, language=self.language)
        matched, match_reason = self._select_match(paper, candidates or [])
        if matched is None and self.require_match:
            err = match_reason
            if doi_direct_failure:
                err = f"doi_direct_failed={doi_direct_failure};{match_reason}"
            return {
                "ok": False,
                "error": err,
                "match_reason": match_reason,
                "num_candidates": len(candidates or []),
            }

        selected_title = normalize_text((matched or {}).get("title", "")) or title
        ok, channel, resolved_path = self._download_paper(selected_title, paper.doi, destination_path, verbose=False)
        if not ok:
            err = "download_failed"
            if doi_direct_failure:
                err = f"doi_direct_failed={doi_direct_failure};{err}"
            return {
                "ok": False,
                "error": err,
                "match_reason": match_reason,
                "matched_title": selected_title,
                "num_candidates": len(candidates or []),
            }
        final_path = _ensure_path(resolved_path, destination_path)
        return {
            "ok": True,
            "channel": channel,
            "pdf_path": final_path,
            "match_reason": match_reason,
            "matched_title": selected_title,
            "num_candidates": len(candidates or []),
        }
