from __future__ import annotations

from .models import ChunkRecord, FullTextRecord
from .utils import normalize_text


def split_documents_into_chunks(
    documents: list[FullTextRecord], chunk_words: int = 250, overlap_words: int = 40
) -> list[ChunkRecord]:
    chunk_words = max(int(chunk_words), 50)
    overlap_words = max(min(int(overlap_words), chunk_words - 1), 0)
    chunks: list[ChunkRecord] = []
    for document in documents:
        fallback_sections = [{"section": "Body", "text": document.text}]
        for sec_idx, section in enumerate(document.sections or fallback_sections):
            section_name = normalize_text(section.get("section", "Body"))
            text = normalize_text(section.get("text", ""))
            if not text:
                continue
            words = text.split()
            start = 0
            chunk_no = 0
            while start < len(words):
                end = min(start + chunk_words, len(words))
                chunk_text = " ".join(words[start:end]).strip()
                if chunk_text:
                    chunks.append(
                        ChunkRecord(
                            doc_id=document.doc_id,
                            chunk_id=f"{document.doc_id}::chunk_{sec_idx:02d}_{chunk_no:04d}",
                            title=document.title,
                            section=section_name,
                            text=chunk_text,
                            start_offset=start,
                            end_offset=end,
                        )
                    )
                    chunk_no += 1
                if end >= len(words):
                    break
                start = max(end - overlap_words, start + 1)
    return chunks
