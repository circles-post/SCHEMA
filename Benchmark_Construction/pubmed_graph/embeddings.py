from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests

from .models import PaperRecord
from .utils import cosine_similarity, normalize_keyword, tokenize

try:
    import torch
    from transformers import AutoModel, AutoTokenizer
except Exception as exc:  # pragma: no cover
    torch = None
    AutoModel = None
    AutoTokenizer = None
    TRANSFORMER_IMPORT_ERROR = exc
else:  # pragma: no cover
    TRANSFORMER_IMPORT_ERROR = None


@dataclass
class FusedNodeGroup:
    canonical_name: str
    member_names: list[str]
    node_type: str | None = None


class RemoteEmbeddingClient:
    def __init__(self, config: dict[str, Any], default_model: str):
        self.service_url = str(config.get("service_url") or "").strip().rstrip("/")
        self.default_model = str(config.get("remote_model") or default_model).strip() or default_model
        self.embed_path = str(config.get("embed_path") or "/embed")
        self.score_path = str(config.get("score_path") or "/score")
        self.timeout = float(config.get("request_timeout", 120.0))
        self.verify_ssl = bool(config.get("verify_ssl", True))
        self.auth_token = str(config.get("auth_token") or "").strip()
        self.session = requests.Session()
        if self.auth_token:
            self.session.headers.update({"Authorization": f"Bearer {self.auth_token}"})

    @property
    def enabled(self) -> bool:
        return bool(self.service_url)

    def _url(self, path: str) -> str:
        if path.startswith("http://") or path.startswith("https://"):
            return path
        path = path if path.startswith("/") else f"/{path}"
        return f"{self.service_url}{path}"

    def _post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        response = self.session.post(
            self._url(path),
            json=payload,
            timeout=self.timeout,
            verify=self.verify_ssl,
        )
        response.raise_for_status()
        data = response.json()
        if not isinstance(data, dict):
            raise ValueError("Remote embedding service returned non-dict JSON")
        if data.get("error"):
            raise ValueError(str(data["error"]))
        return data

    def embed_texts(self, texts: list[str], model_name: str | None = None) -> list[list[float]]:
        payload = {
            "model": model_name or self.default_model,
            "texts": [str(text or "") for text in texts],
        }
        data = self._post(self.embed_path, payload)
        embeddings = data.get("embeddings", [])
        if not isinstance(embeddings, list):
            raise ValueError("Remote embedding service returned invalid embeddings payload")
        return embeddings

    def score_paper(self, keyword: str, paper: PaperRecord) -> float:
        payload = {
            "keyword": keyword,
            "title": paper.title,
            "abstract": paper.abstract,
        }
        data = self._post(self.score_path, payload)
        return max(0.0, min(1.0, float(data.get("score", 0.0) or 0.0)))


class HFTextEmbedder:
    def __init__(
        self,
        model_path: str,
        enabled: bool = False,
        batch_size: int = 8,
        device: str = "cpu",
        max_length: int = 512,
        instruction: str | None = None,
    ):
        self.model_path = model_path
        self.enabled = enabled
        self.batch_size = max(int(batch_size), 1)
        self.device = device
        self.max_length = max(int(max_length), 32)
        self.instruction = instruction
        self._tokenizer = None
        self._model = None

    def _lazy_load(self) -> None:
        if not self.enabled:
            return
        if self._tokenizer is not None and self._model is not None:
            return
        if TRANSFORMER_IMPORT_ERROR is not None or torch is None or AutoTokenizer is None or AutoModel is None:
            raise ImportError(
                f"transformers/torch are required for local embedding models: {TRANSFORMER_IMPORT_ERROR}"
            )
        if not Path(self.model_path).exists():
            raise FileNotFoundError(f"Model path does not exist: {self.model_path}")
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_path, local_files_only=True)
        self._model = AutoModel.from_pretrained(self.model_path, local_files_only=True)
        self._model.eval()
        self._model.to(self.device)

    def _prepare_text(self, text: str) -> str:
        text = (text or "").strip()
        if self.instruction:
            return f"{self.instruction}{text}"
        return text

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not self.enabled:
            return []
        self._lazy_load()
        assert self._tokenizer is not None and self._model is not None and torch is not None
        prepared = [self._prepare_text(text) for text in texts]
        all_vectors: list[list[float]] = []
        with torch.no_grad():
            for start in range(0, len(prepared), self.batch_size):
                batch = prepared[start : start + self.batch_size]
                encoded = self._tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                )
                encoded = {k: v.to(self.device) for k, v in encoded.items()}
                outputs = self._model(**encoded)
                hidden = outputs.last_hidden_state
                mask = encoded["attention_mask"].unsqueeze(-1).expand(hidden.size()).float()
                pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
                pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
                all_vectors.extend(pooled.cpu().numpy().tolist())
        return all_vectors


class SapBERTScorer:
    def __init__(self, config: dict[str, Any]):
        self.enabled = bool(config.get("enabled", False))
        self.model_path = config.get("model_path") or config.get(
            "model_name", "cambridgeltl/SapBERT-from-PubMedBERT-fulltext"
        )
        self.batch_size = int(config.get("batch_size", 8))
        self.device = config.get("device", "cpu")
        self.remote_client = RemoteEmbeddingClient(config, default_model="sapbert")
        self.embedder = HFTextEmbedder(
            model_path=self.model_path,
            enabled=self.enabled and not self.remote_client.enabled,
            batch_size=self.batch_size,
            device=self.device,
        )

    def score(self, keyword: str, paper: PaperRecord) -> float:
        text = f"{paper.title} {paper.abstract}".strip()
        if not text:
            return 0.0
        if self.enabled and self.remote_client.enabled:
            return self.remote_client.score_paper(keyword, paper)
        if self.enabled:
            vectors = self.embedder.embed_texts([keyword, text])
            if len(vectors) == 2:
                return max(0.0, min(1.0, cosine_similarity(vectors[0], vectors[1])))
        normalized_keyword = normalize_keyword(keyword)
        normalized_text = normalize_keyword(text)
        if normalized_keyword and normalized_keyword in normalized_text:
            return 1.0
        keyword_tokens = tokenize(keyword)
        paper_tokens = tokenize(text)
        if not keyword_tokens or not paper_tokens:
            return 0.0
        overlap = len(keyword_tokens & paper_tokens)
        if overlap == 0:
            return 0.0
        recall_like = overlap / len(keyword_tokens)
        jaccard = overlap / len(keyword_tokens | paper_tokens)
        return max(recall_like, jaccard)


class BGELargeEmbedder:
    def __init__(self, config: dict[str, Any]):
        self.enabled = bool(config.get("enabled", False))
        self.model_path = config.get("model_path") or config.get("model_name", "BAAI/bge-large-en-v1.5")
        self.instruction = config.get("instruction_prefix", "")
        self.remote_client = RemoteEmbeddingClient(config, default_model="bge")
        self.local_embedder = HFTextEmbedder(
            model_path=self.model_path,
            enabled=self.enabled and not self.remote_client.enabled,
            batch_size=int(config.get("batch_size", 8)),
            device=config.get("device", "cpu"),
            instruction=self.instruction,
        )

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not self.enabled:
            return []
        if self.remote_client.enabled:
            prepared = [f"{self.instruction}{(text or '').strip()}" if self.instruction else str(text or "") for text in texts]
            return self.remote_client.embed_texts(prepared, model_name=self.remote_client.default_model)
        return self.local_embedder.embed_texts(texts)


class NodeFusionEngine:
    def __init__(self, embedder: BGELargeEmbedder, threshold: float = 0.9, compatible_types_only: bool = True):
        self.embedder = embedder
        self.threshold = float(threshold)
        self.compatible_types_only = compatible_types_only

    @staticmethod
    def _canonical_name(member_names: list[str]) -> str:
        if not member_names:
            return ""
        return max(
            member_names,
            key=lambda name: (len(normalize_keyword(name).split()), len(name), name.lower()),
        )

    def fuse(self, nodes: list[dict[str, Any]]) -> list[FusedNodeGroup]:
        if not nodes:
            return []
        parent = list(range(len(nodes)))

        def find(idx: int) -> int:
            while parent[idx] != idx:
                parent[idx] = parent[parent[idx]]
                idx = parent[idx]
            return idx

        def union(a: int, b: int) -> None:
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[rb] = ra

        normalized = [normalize_keyword(node.get("text", "")) for node in nodes]
        for i in range(len(nodes)):
            for j in range(i + 1, len(nodes)):
                if normalized[i] and normalized[i] == normalized[j]:
                    union(i, j)
        if self.embedder.enabled:
            embeddings = self.embedder.embed_texts([node.get("text", "") for node in nodes])
            for i in range(len(nodes)):
                for j in range(i + 1, len(nodes)):
                    type_i = nodes[i].get("type")
                    type_j = nodes[j].get("type")
                    if self.compatible_types_only and type_i and type_j and type_i != type_j:
                        continue
                    if cosine_similarity(embeddings[i], embeddings[j]) >= self.threshold:
                        union(i, j)
        groups: dict[int, list[dict[str, Any]]] = {}
        for idx, node in enumerate(nodes):
            groups.setdefault(find(idx), []).append(node)
        fused = []
        for members in groups.values():
            member_names = sorted({node.get("text", "") for node in members if node.get("text")})
            types = [node.get("type") for node in members if node.get("type")]
            fused.append(
                FusedNodeGroup(
                    canonical_name=self._canonical_name(member_names),
                    member_names=member_names,
                    node_type=types[0] if types else None,
                )
            )
        return fused
