from __future__ import annotations

import hashlib
import json
import math
import os
import re
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

TOKEN_RE = re.compile(r"[A-Za-z0-9]+")
TAG_RE = re.compile(r"<[^>]+>")
LANDING_BOILERPLATE_RE = re.compile(
    r"(?:html\{|body\{|font-family|@media|display:|window\.|document\.|datalayer|gtm|"
    r"log in|\bmenu\b|find a journal|publish with us|submit manuscript|search by keyword or author|"
    r"your privacy choices|accessibility statement|terms and conditions|privacy policy|advertisement|"
    r"copy shareable link|saved research|\bcart\b|springer nature)",
    re.IGNORECASE,
)
ACADEMIC_SECTION_RE = re.compile(
    r"\b(?:abstract|introduction|methods?|results?|discussion|conclusion|references|author information|ethics declarations)\b",
    re.IGNORECASE,
)


def ensure_dir(path: str | Path) -> Path:
    target = Path(path)
    target.mkdir(parents=True, exist_ok=True)
    return target


def normalize_keyword(term: str) -> str:
    return re.sub(r"\s+", " ", (term or "").strip().lower())


def normalize_text(text: object) -> str:
    if text is None:
        return ""
    return re.sub(r"\s+", " ", str(text).replace("\x00", " ")).strip()


def strip_markup(text: object) -> str:
    if text is None:
        return ""
    value = str(text)
    value = TAG_RE.sub(" ", value)
    return normalize_text(value)


def extract_html_text(text: object) -> str:
    if text is None:
        return ""
    value = str(text)
    value = re.sub(r"<script[\s\S]*?</script>", " ", value, flags=re.IGNORECASE)
    value = re.sub(r"<style[\s\S]*?</style>", " ", value, flags=re.IGNORECASE)
    return strip_markup(value)


def tokenize(text: str) -> set[str]:
    return {token.lower() for token in TOKEN_RE.findall(text or "")}


def load_json(path: str | Path) -> Any:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: str | Path, payload: Any) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def write_jsonl(path: str | Path, rows: Iterable[Any]) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            if is_dataclass(row):
                row = asdict(row)
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def load_env_file(path: str | Path | None) -> dict[str, str]:
    if not path:
        return {}
    env_path = Path(path)
    if not env_path.exists():
        return {}
    loaded: dict[str, str] = {}
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)
        loaded[key] = value
    return loaded


def chunked(items: Sequence[Any], size: int) -> Iterator[Sequence[Any]]:
    size = max(int(size), 1)
    for idx in range(0, len(items), size):
        yield items[idx : idx + size]


def cosine_similarity(vec_a: Sequence[float], vec_b: Sequence[float]) -> float:
    if not vec_a or not vec_b or len(vec_a) != len(vec_b):
        return 0.0
    dot = sum(float(a) * float(b) for a, b in zip(vec_a, vec_b))
    norm_a = math.sqrt(sum(float(a) * float(a) for a in vec_a))
    norm_b = math.sqrt(sum(float(b) * float(b) for b in vec_b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


def resolve_config_paths(config: dict[str, Any], config_path: str | Path) -> dict[str, Any]:
    base_dir = Path(config_path).resolve().parent
    resolved = dict(config)
    env_file = resolved.get("env_file")
    if env_file and not Path(env_file).is_absolute():
        resolved["env_file"] = str((base_dir / env_file).resolve())
    benchmark_cfg = dict(resolved.get("benchmark_seed_source", {}))
    script_path = benchmark_cfg.get("benchmark_script_path")
    if script_path and not Path(script_path).is_absolute():
        benchmark_cfg["benchmark_script_path"] = str((base_dir / script_path).resolve())
    cache_dir = benchmark_cfg.get("cache_dir")
    if cache_dir and not Path(cache_dir).is_absolute():
        benchmark_cfg["cache_dir"] = str((base_dir / cache_dir).resolve())
    if benchmark_cfg:
        resolved["benchmark_seed_source"] = benchmark_cfg
    paper_cache_cfg = dict(resolved.get("paper_cache", {}))
    paper_cache_dir = paper_cache_cfg.get("cache_dir")
    if paper_cache_dir and not Path(paper_cache_dir).is_absolute():
        paper_cache_cfg["cache_dir"] = str((base_dir / paper_cache_dir).resolve())
    if paper_cache_cfg:
        resolved["paper_cache"] = paper_cache_cfg
    sciverse_cfg = dict(resolved.get("sciverse", {}))
    toolkit_root = sciverse_cfg.get("toolkit_root")
    if toolkit_root and not Path(toolkit_root).is_absolute():
        sciverse_cfg["toolkit_root"] = str((base_dir / toolkit_root).resolve())
    if sciverse_cfg:
        resolved["sciverse"] = sciverse_cfg
    return resolved


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_text(value: str) -> str:
    return hashlib.sha256((value or "").encode("utf-8")).hexdigest()


def compute_paper_cache_key(*, doi: str | None = None, pmcid: str | None = None, pmid: str | None = None, title: str | None = None) -> str:
    normalized_doi = normalize_keyword(doi or "")
    if normalized_doi:
        return f"doi::{normalized_doi}"
    normalized_pmcid = normalize_keyword(pmcid or "")
    if normalized_pmcid:
        return f"pmcid::{normalized_pmcid}"
    normalized_pmid = normalize_keyword(pmid or "")
    if normalized_pmid:
        return f"pmid::{normalized_pmid}"
    normalized_title = normalize_keyword(title or "")
    if normalized_title:
        return f"title::{sha256_text(normalized_title)}"
    raise ValueError("Cannot compute cache key without DOI, PMCID, PMID, or title")


def cache_slug(cache_key: str, length: int = 16) -> str:
    return sha256_text(cache_key)[: max(int(length), 8)]


def render_markdown_document(title: str, abstract: str = "", sections: Sequence[dict[str, str]] | None = None, text: str = "") -> str:
    parts: list[str] = []
    safe_title = normalize_text(title) or "Untitled"
    parts.append(f"# {safe_title}")
    if normalize_text(abstract):
        parts.append("## Abstract\n\n" + normalize_text(abstract))
    clean_sections = list(sections or [])
    for section in clean_sections:
        heading = normalize_text(section.get("section", "Body")) or "Body"
        body = normalize_text(section.get("text", ""))
        if body:
            parts.append(f"## {heading}\n\n{body}")
    if not clean_sections and normalize_text(text):
        parts.append("## Body\n\n" + normalize_text(text))
    return "\n\n".join(parts).strip() + "\n"


def clean_crossref_landing_text(text: object) -> str:
    raw = extract_html_text(text)
    if not raw:
        return ""
    normalized = str(raw).replace(" | ", "\n")
    normalized = re.sub(r"(?i)(abstract|introduction|methods?|results?|discussion|references|author information|ethics declarations)", lambda m: f"\n{m.group(1)}\n", normalized)
    pieces = [normalize_text(piece) for piece in re.split(r"[\n\r]+", normalized)]
    kept: list[str] = []
    for piece in pieces:
        if not piece:
            continue
        if LANDING_BOILERPLATE_RE.search(piece):
            continue
        if len(piece) < 40 and not ACADEMIC_SECTION_RE.search(piece):
            continue
        kept.append(piece)
    deduped: list[str] = []
    seen: set[str] = set()
    for piece in kept:
        key = normalize_keyword(piece)
        if key and key not in seen:
            seen.add(key)
            deduped.append(piece)
    return "\n".join(deduped).strip()


def is_valid_crossref_landing_text(text: object, *, min_chars: int = 1200, title: str = "") -> bool:
    value = normalize_text(text)
    if len(value) < max(int(min_chars), 200):
        return False
    lower = value.lower()
    boilerplate_hits = len(LANDING_BOILERPLATE_RE.findall(value))
    if boilerplate_hits >= 8:
        return False
    if lower.count("window.") + lower.count("document.") + lower.count("font-family") + lower.count("@media") >= 2:
        return False
    sentence_like = re.findall(r"[A-Z][^.?!]{40,}[.?!]", value)
    if len(sentence_like) < 2:
        return False
    title_norm = normalize_keyword(title)
    if title_norm:
        title_tokens = [token for token in title_norm.split() if len(token) > 3]
        if title_tokens:
            overlap = sum(1 for token in title_tokens if token in lower)
            if overlap == 0:
                return False
    if not ACADEMIC_SECTION_RE.search(value) and len(sentence_like) < 3:
        return False
    return True
