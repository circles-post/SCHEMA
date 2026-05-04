from __future__ import annotations

import importlib
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd

from .llm import InternChatClient
from .utils import chunked, normalize_keyword, normalize_text

BIO_SUFFIX_TERMS = {
    "protein",
    "proteins",
    "domain",
    "domains",
    "complex",
    "complexes",
    "system",
    "pathway",
    "pathways",
    "enzyme",
    "enzymes",
    "biology",
    "resistance",
    "period",
    "kinetics",
    "affinity",
    "specificity",
    "conductance",
    "dynamics",
    "chromatography",
    "mutation",
    "mutations",
    "ligand",
    "ligands",
}

GENERIC_CANDIDATES = {
    "protein",
    "proteins",
    "protein complex",
    "protein complexes",
    "protein structure",
    "protein function",
    "protein mechanism",
    "protein interaction",
    "protein binding",
    "binding",
    "interaction",
    "mechanism",
    "structure",
    "function",
    "inter-domain interface",
    "interface",
    "according",
    "based",
    "how",
    "what",
    "which",
    "why",
    "when",
    "type",
    "types",
    "his",
    "its",
    "their",
    "study",
    "study of protein",
    "document",
    "document segment",
    "provided document segment",
    "researchers",
    "recent study",
    "various environments",
    "structural and functional studies",
    "synthetic biology applications",
    "plants",
}
BAD_START_WORDS = {
    "and",
    "as",
    "at",
    "among",
    "between",
    "bound",
    "conditions",
    "derived",
    "do",
    "does",
    "effect",
    "impact",
    "interaction",
    "introducing",
    "researchers",
    "role",
    "substrate",
    "to",
    "how",
    "what",
    "which",
    "why",
    "when",
    "where",
}

QUESTION_PREFIXES = (
    "how do ",
    "how did ",
    "how can ",
    "how were ",
    "which of the following statements about ",
    "which of the following statements best describes ",
    "according to the provided document segment, what is ",
    "according to the document, what can be inferred about ",
    "according to a recent study, what is ",
    "according to a study on ",
    "what is ",
    "what can be inferred about ",
    "what factor might contribute to ",
    "what is the impact of ",
)
LEADING_NOISE_RE = re.compile(
    r"^(?:"
    r"role of(?: the)?|effect of|impact of|activity of|structure of|function of|"
    r"conditions for|interaction between|binding energy with|binding affinity of|"
    r"binding specificity of|researchers (?:purify|analyze|determine)|"
    r"substrate interact with|which domain of|among different classes of|"
    r"between two molecules of|derived for|does |and activity of|bound to|"
    r"co-expressing|removing|introducing|do "
    r")\s+",
    re.IGNORECASE,
)
TRAILING_NOISE_RE = re.compile(
    r"\s+(?:when bound to|in this study|using .*|for .*|as described.*|on .*|in .*|with .*|compared to .*|according to .*|differ from .*|presented.*)$",
    re.IGNORECASE,
)

UPPER_ENTITY_RE = re.compile(r"\b[A-Za-z0-9]*[A-Z][A-Za-z0-9]*(?:[/:+-][A-Za-z0-9]+)*\b")
FOCUS_NODE_RE = re.compile(r"focus node:\s*([^\n]+)", re.IGNORECASE)
TAXON_RE = re.compile(r"\b(?:[A-Z]\.\s*[a-z]+|[A-Z][a-z]+\s+[a-z]+)\b")
SCIENTIFIC_PHRASE_RE = re.compile(
    r"\b([A-Za-z0-9][A-Za-z0-9/:+-]*(?:\s+[A-Za-z0-9][A-Za-z0-9/:+-]*){0,4}\s+"
    r"(?:protein|proteins|domain|domains|complex|complexes|system|pathway|pathways|"
    r"enzyme|enzymes|biology|resistance|period|kinetics|affinity|specificity|"
    r"conductance|dynamics|chromatography|mutation|mutations|ligand|ligands))\b",
    re.IGNORECASE,
)
COMMON_SCIENCE_PHRASE_RE = re.compile(
    r"\b([A-Za-z][A-Za-z0-9-]*(?:\s+[A-Za-z][A-Za-z0-9-]*){0,3}\s+"
    r"(?:synthetic biology|antibiotic resistance|electron tomography|circadian period|"
    r"binding affinity|binding specificity|oligomeric state|colistin resistance|"
    r"gel filtration chromatography|protein structure|protein function))\b",
    re.IGNORECASE,
)
STOPWORD_RE = re.compile(r"^(?:a|an|the|of|in|on|for|to|with|by|from|into|during|after|before|under|between|among|and|or|that|this|these|those|is|are|was|were|be|been|being|as|it|its|their|other|another|different|recent|provided|document|study|researchers)$")
SHORT_ACRONYM_RE = re.compile(r"^[A-Z]{3,5}$")
GENERIC_SUFFIX_EXPANSION_RE = re.compile(
    r"\b(?:structure|function|mechanism|interaction|binding|mutation)\b$",
    re.IGNORECASE,
)
PROMISCUOUS_PHRASE_RE = re.compile(
    r"\b(?:and its|and their|and other|another|different|provided|document segment|recent study|what|which|how|why|when|where)\b",
    re.IGNORECASE,
)
BENCHMARK_SEED_SYSTEM_PROMPT = """You extract literature-retrieval seed keywords from life-science benchmark questions.

Return strict JSON only with one key: "seed_keywords".

Rules:
- extract concise biomedical seed keywords or short phrases useful for PubMed search
- prioritize named proteins, genes, domains, complexes, enzymes, mutations, ligands, pathways, species, diseases, assays, and biological processes
- prefer canonical entity names over question wording
- do not output question fragments like "how does", "what is", "role of", "effect of", "according to"
- do not output overly generic phrases like "protein function", "protein structure", "protein complexes", "inter-domain interface", or broad category words like "plants"
- each item should usually be 1 to 5 words
- avoid duplicates
- reject short opaque all-caps acronyms unless they are clearly canonical biomedical symbols
- reject sentence fragments, pronoun-heavy phrases, and generic "X structure/function/mechanism/interaction" expansions
- if a question contains multiple important entities, include multiple seed keywords
- return at most the requested number of items

Good examples: "IMP1 protein", "KH1 domain", "Smc5/6 complex", "LOV domain", "Min system", "E. coli".
Bad examples: "what structural feature", "protein function", "inter-domain interface", "recent study", "plants".
"""


class LocalBenchmarkDataset:
    def __init__(self, path: str | Path):
        self.path = Path(path).resolve()

    def load(self) -> pd.DataFrame:
        suffix = self.path.suffix.lower()
        if suffix == ".csv":
            return pd.read_csv(self.path)
        if suffix in {".tsv", ".txt"}:
            return pd.read_csv(self.path, sep="\t")
        raise ValueError(f"Unsupported local benchmark file format: {self.path}")


def _load_dataset(config: dict[str, Any]) -> Any:
    local_path = config.get("benchmark_local_path")
    if local_path:
        resolved_local = Path(local_path).resolve()
        if not resolved_local.exists():
            raise FileNotFoundError(f"Local benchmark file not found: {resolved_local}")
        return LocalBenchmarkDataset(resolved_local)
    script_path = Path(config["benchmark_script_path"]).resolve()
    module_root = script_path.parent
    if str(module_root) not in sys.path:
        sys.path.insert(0, str(module_root))
    module = importlib.import_module("dataset.life_science_datasets")
    get_dataset = getattr(module, "get_dataset")
    dataset_name = config.get("dataset_name", "ProteinLMBench")
    cache_dir = config.get("cache_dir")
    return get_dataset(dataset_name, cache_dir=cache_dir)


def _clean_candidate(text: str) -> str:
    value = normalize_text(text)
    value = re.sub(r"\s+", " ", value)
    value = value.strip(" ,.;:!?()[]{}\"'")
    lower = value.lower()
    for prefix in QUESTION_PREFIXES:
        if lower.startswith(prefix):
            value = value[len(prefix) :].strip()
            lower = value.lower()
    value = LEADING_NOISE_RE.sub("", value).strip()
    value = TRAILING_NOISE_RE.sub("", value).strip()
    value = re.sub(r"^(.+?)\s+and\s+(?:its|their)\s+(protein|gene|domain|complex|pathway|enzyme)$", r"\1 \2", value, flags=re.IGNORECASE)
    value = re.sub(r"^([A-Za-z0-9/:+_.-]+)\s+and\s+other\s+.+?\s+(protein|gene|domain|complex|pathway|enzyme)$", r"\1 \2", value, flags=re.IGNORECASE)
    value = re.sub(r"\bimpact\s+(?:its|their)\s+binding affinity\b", "", value, flags=re.IGNORECASE)
    value = re.sub(r"\b(?:its|their|other|another|different)\b", " ", value, flags=re.IGNORECASE)
    if " from " in value.lower():
        left, right = re.split(r"\s+from\s+", value, maxsplit=1, flags=re.IGNORECASE)
        right = right.strip()
        if right and _has_biomedical_signal(right) and len(right) >= max(len(left.strip()) - 2, 1):
            value = right
    value = re.sub(r"\b(?:the|a|an)\s+", "", value, count=1, flags=re.IGNORECASE)
    value = re.sub(r"\s+", " ", value).strip()
    phrase_matches = []
    for match in SCIENTIFIC_PHRASE_RE.finditer(value):
        phrase = match.group(1).strip()
        if phrase:
            phrase_matches.append(phrase)
    if phrase_matches:
        value = max(
            phrase_matches,
            key=lambda item: (
                len(item.split()),
                sum(ch.isupper() for ch in item),
                len(item),
            ),
        )
    words = value.split()
    for idx, token in enumerate(words):
        stripped = token.strip(",.;:!?()[]{}\"'")
        has_signal = (
            any(ch.isdigit() for ch in stripped)
            or any(ch.isupper() for ch in stripped[1:])
            or any(sym in stripped for sym in "/:-+")
        )
        if has_signal and idx > 0:
            value = " ".join(words[idx:])
            break
    value = value.strip(" ,.;:!?()[]{}\"'")
    return value


def _split_compound_candidate(text: str) -> list[str]:
    value = _clean_candidate(text)
    if not value:
        return []
    variants = [value]
    suffix_match = re.match(
        r"^(.+?)\s+and\s+(?:the\s+)?(.+?)\s+"
        r"(domain|domains|protein|proteins|enzyme|enzymes|complex|complexes|gene|genes|mutation|mutations)$",
        value,
        re.IGNORECASE,
    )
    if suffix_match:
        left = suffix_match.group(1).strip()
        right = suffix_match.group(2).strip()
        suffix = suffix_match.group(3).lower()
        singular = suffix[:-1] if suffix.endswith("s") else suffix
        variants.extend(
            [
                f"{left} {singular}".strip(),
                f"{right} {singular}".strip(),
            ]
        )
    else:
        bare_match = re.match(r"^([A-Za-z0-9/:+_.-]+)\s+and\s+([A-Za-z0-9/:+_.-]+)$", value)
        if bare_match:
            variants.extend([bare_match.group(1).strip(), bare_match.group(2).strip()])
    cleaned: list[str] = []
    seen: set[str] = set()
    for item in variants:
        normalized = normalize_keyword(_clean_candidate(item))
        if normalized and normalized not in seen:
            seen.add(normalized)
            cleaned.append(_clean_candidate(item))
    return cleaned


def _canonicalize_seed_candidate(text: str) -> str:
    value = _clean_candidate(text)
    if not value:
        return ""
    replacements = [
        (r"\bprotein complexes\b", "complex"),
        (r"\bprotein complex\b", "complex"),
        (r"\bcomplexes\b", "complex"),
        (r"\bproteins\b", "protein"),
        (r"\bdomains\b", "domain"),
        (r"\benzymes\b", "enzyme"),
        (r"\bgenes\b", "gene"),
        (r"\bmutations\b", "mutation"),
        (r"\bvariants\b", "variant"),
        (r"\bpathways\b", "pathway"),
        (r"\bligands\b", "ligand"),
    ]
    for pattern, replacement in replacements:
        value = re.sub(pattern, replacement, value, flags=re.IGNORECASE)
    value = re.sub(r"\s+", " ", value).strip(" ,.;:!?()[]{}\"'")
    return value


def _canonical_seed_key(text: str) -> str:
    return normalize_keyword(_canonicalize_seed_candidate(text))


def _has_biomedical_signal(value: str) -> bool:
    if not value:
        return False
    if TAXON_RE.search(value):
        return True
    if any(ch.isdigit() for ch in value):
        return True
    if any(ch.isupper() for ch in value[1:]):
        return True
    if any(sym in value for sym in "/:-+"):
        return True
    lowered = value.lower()
    if any(lowered.endswith(term) for term in BIO_SUFFIX_TERMS):
        return True
    if SCIENTIFIC_PHRASE_RE.search(value) or COMMON_SCIENCE_PHRASE_RE.search(value):
        return True
    tokens = lowered.split()
    if len(tokens) == 1 and len(tokens[0]) >= 6 and not tokens[0].endswith("s"):
        return True
    return False


def _looks_like_sentence_fragment(text: str) -> bool:
    tokens = text.split()
    if len(tokens) > 6:
        return True
    if PROMISCUOUS_PHRASE_RE.search(text):
        return True
    stopwords = sum(1 for token in tokens if STOPWORD_RE.match(token.lower()))
    if tokens and stopwords / len(tokens) >= 0.45:
        return True
    return False


def _looks_like_short_noisy_acronym(text: str) -> bool:
    value = text.strip()
    return bool(SHORT_ACRONYM_RE.fullmatch(value)) and not any(ch.isdigit() for ch in value)


def _looks_like_generic_suffix_expansion(text: str) -> bool:
    lower = normalize_keyword(text)
    if not GENERIC_SUFFIX_EXPANSION_RE.search(lower):
        return False
    return len(lower.split()) >= 2


def _should_keep_ranked_candidate(text: str, frequency: int) -> bool:
    if _looks_like_sentence_fragment(text):
        return False
    if _looks_like_generic_suffix_expansion(text):
        return False
    if _looks_like_short_noisy_acronym(text) and frequency < 2:
        return False
    return True


def _is_useful_candidate(text: str) -> bool:
    value = _clean_candidate(text)
    normalized = normalize_keyword(value)
    if not normalized or normalized in GENERIC_CANDIDATES:
        return False
    if len(normalized) < 3:
        return False
    if normalized.isdigit():
        return False
    if normalized in {"how", "what", "which", "study generated"}:
        return False
    if normalized.startswith(("in ", "of ", "for ", "with ", "using ", "according to ", "based on ")):
        return False
    first_word = normalized.split()[0]
    if first_word in BAD_START_WORDS:
        return False
    if normalized.endswith((" study", " studies", " question", " questions")):
        return False
    if _looks_like_sentence_fragment(value):
        return False
    if _looks_like_generic_suffix_expansion(value):
        return False
    noisy_substrings = (
        "according to",
        " as ",
        "differ from",
        "experimental evidence",
        "researchers analyze",
        "researchers determine",
        "researchers purify",
        "binding of different",
        "presented",
    )
    if any(fragment in normalized for fragment in noisy_substrings):
        return False
    if normalized.startswith(("1 and ", "2 ", "511 ")):
        return False
    if " and the " in normalized:
        return False

    if not _has_biomedical_signal(value):
        return False
    return True


def _candidate_score(text: str, frequency: int) -> tuple[int, int, int, int, str]:
    score = 0
    if any(ch.isdigit() for ch in text):
        score += 4
    if any(ch.isupper() for ch in text[1:]):
        score += 3
    if any(sym in text for sym in "/:-+"):
        score += 2
    if len(text.split()) >= 2:
        score += min(len(text.split()), 4)
    if any(text.lower().endswith(term) for term in BIO_SUFFIX_TERMS):
        score += 3
    if normalize_keyword(text) == "synthetic biology":
        score += 1
    if _looks_like_short_noisy_acronym(text):
        score -= 6
    if _looks_like_generic_suffix_expansion(text):
        score -= 6
    return (-score, -frequency, len(text.split()), -len(text), text.lower())


def _llm_extract_batch_seed_keywords(
    client: InternChatClient,
    questions: list[str],
    max_terms: int,
) -> list[str]:
    numbered_questions = "\n".join(
        f"{idx + 1}. {normalize_text(question)}" for idx, question in enumerate(questions) if normalize_text(question)
    )
    payload = client.chat_json(
        [
            {"role": "system", "content": BENCHMARK_SEED_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"Questions:\n{numbered_questions}\n\n"
                    f"Return at most {max_terms} benchmark seed keywords."
                ),
            },
        ]
    )
    if isinstance(payload, dict):
        items = payload.get("seed_keywords", [])
    else:
        items = payload
    if not isinstance(items, list):
        return []
    return [str(item).strip() for item in items if str(item).strip()]


def _resolve_llm_seed_keywords(questions: list[str], config: dict[str, Any]) -> tuple[list[str], dict[str, Any]]:
    llm_cfg = dict(config.get("llm", {}))
    if not llm_cfg.get("enabled", False):
        return [], {"enabled": False, "batch_count": 0, "raw_candidate_count": 0}
    llm_cfg.setdefault("model", "intern-s1-pro")
    llm_cfg.setdefault("thinking_mode", False)
    llm_cfg.setdefault("temperature", 0.0)
    llm_cfg.setdefault("max_tokens", 1200)
    batch_size = max(int(llm_cfg.get("question_batch_size", 25)), 1)
    max_terms_per_batch = max(int(llm_cfg.get("max_terms_per_batch", 20)), 1)
    client = InternChatClient(llm_cfg)

    cleaned: list[str] = []
    seen: set[str] = set()
    batch_count = 0
    raw_candidate_count = 0
    for batch in chunked(questions, batch_size):
        batch_count += 1
        try:
            items = _llm_extract_batch_seed_keywords(client, list(batch), max_terms=max_terms_per_batch)
        except Exception:
            continue
        raw_candidate_count += len(items)
        for item in items:
            for candidate in _split_compound_candidate(item):
                candidate = _canonicalize_seed_candidate(candidate)
                normalized = _canonical_seed_key(candidate)
                if not _is_useful_candidate(candidate) or normalized in seen:
                    continue
                seen.add(normalized)
                cleaned.append(candidate)
    return cleaned, {
        "enabled": True,
        "model": llm_cfg.get("model", "intern-s1-pro"),
        "thinking_mode": bool(llm_cfg.get("thinking_mode", False)),
        "batch_size": batch_size,
        "batch_count": batch_count,
        "raw_candidate_count": raw_candidate_count,
        "accepted_candidate_count": len(cleaned),
    }


def _extract_from_question(question: str) -> list[str]:
    text = normalize_text(question)
    candidates: list[str] = []

    focus_match = FOCUS_NODE_RE.search(text)
    if focus_match:
        candidates.append(focus_match.group(1))

    for match in SCIENTIFIC_PHRASE_RE.finditer(text):
        candidates.append(match.group(1))

    for match in COMMON_SCIENCE_PHRASE_RE.finditer(text):
        candidates.append(match.group(1))

    for match in TAXON_RE.finditer(text):
        candidates.append(match.group(0))

    for match in UPPER_ENTITY_RE.finditer(text):
        token = match.group(0)
        if token.lower() in GENERIC_CANDIDATES:
            continue
        if any(ch.isdigit() for ch in token) or any(ch.isupper() for ch in token[1:]) or any(sym in token for sym in "/:-+"):
            candidates.append(token)

    lower_text = text.lower()
    for marker in [
        "role of ",
        "effect of ",
        "impact of ",
        "binding affinity of ",
        "binding specificity of ",
        "mechanism of action of ",
        "oligomeric state of ",
        "conductance of ",
        "dynamics of ",
        "activity of ",
    ]:
        if marker in lower_text:
            start = lower_text.index(marker) + len(marker)
            remainder = text[start:]
            stop_match = re.search(r"\b(?: in| on| with| for| using| compared to| as described| according to)\b|\?", remainder, re.IGNORECASE)
            phrase = remainder[: stop_match.start()] if stop_match else remainder
            candidates.append(phrase)

    cleaned: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        for item in _split_compound_candidate(candidate):
            item = _canonicalize_seed_candidate(item)
            normalized = _canonical_seed_key(item)
            if not _is_useful_candidate(item) or normalized in seen:
                continue
            seen.add(normalized)
            cleaned.append(item)
    return cleaned


def resolve_seed_keywords(config: dict[str, Any]) -> tuple[list[str], dict[str, Any] | None]:
    benchmark_cfg = config.get("benchmark_seed_source", {})
    base_seeds = [str(item).strip() for item in config.get("seed_keywords", []) if str(item).strip()]
    if not benchmark_cfg or not benchmark_cfg.get("enabled", False):
        return base_seeds, None

    dataset = _load_dataset(benchmark_cfg)
    df = dataset.load().copy()
    question_limit = int(benchmark_cfg.get("question_limit", 0))
    if question_limit > 0:
        df = df.head(question_limit).copy()
    questions = [str(question) for question in df.get("question", []).tolist()]

    counts: Counter[str] = Counter()
    examples: dict[str, str] = {}
    for question in questions:
        for candidate in _extract_from_question(str(question)):
            candidate = _canonicalize_seed_candidate(candidate)
            normalized = _canonical_seed_key(candidate)
            counts[normalized] += 1
            if normalized not in examples or len(candidate) < len(examples[normalized]):
                examples[normalized] = candidate

    llm_seed_keywords, llm_summary = _resolve_llm_seed_keywords(questions, benchmark_cfg)
    llm_counter: Counter[str] = Counter()
    for candidate in llm_seed_keywords:
        candidate = _canonicalize_seed_candidate(candidate)
        normalized = _canonical_seed_key(candidate)
        llm_counter[normalized] += 1
        if normalized not in examples or len(candidate) < len(examples[normalized]):
            examples[normalized] = candidate
        counts[normalized] += 2

    min_frequency = max(int(benchmark_cfg.get("min_frequency", 1)), 1)
    max_seed_keywords = max(int(benchmark_cfg.get("max_seed_keywords", 50)), 1)
    ranked = [
        (normalized, count, examples[normalized])
        for normalized, count in counts.items()
        if count >= min_frequency and _should_keep_ranked_candidate(examples[normalized], count)
    ]
    ranked.sort(key=lambda item: _candidate_score(item[2], item[1]))
    benchmark_seeds = [item[2] for item in ranked[:max_seed_keywords]]

    mode = str(benchmark_cfg.get("mode", "replace")).lower()
    if mode == "append":
        merged: list[str] = []
        seen: set[str] = set()
        for item in base_seeds + benchmark_seeds:
            normalized = normalize_keyword(item)
            if normalized and normalized not in seen:
                seen.add(normalized)
                merged.append(item)
        final_seeds = merged
    else:
        final_seeds = benchmark_seeds or base_seeds

    summary = {
        "dataset_name": benchmark_cfg.get("dataset_name", "ProteinLMBench"),
        "question_count": int(len(df)),
        "candidate_count": int(len(counts)),
        "selected_seed_count": int(len(final_seeds)),
        "mode": mode,
        "top_benchmark_seeds": benchmark_seeds[:20],
        "llm_seed_summary": llm_summary,
        "llm_seed_examples": llm_seed_keywords[:20],
    }
    return final_seeds, summary
