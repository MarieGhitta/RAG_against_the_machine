import json
from pathlib import Path
from src.models import Chunk
from rank_bm25 import BM25Okapi
import re


def _convert_json_in_chunks(datas: list[dict[str, object]]) -> list[Chunk]:
    """Convert json element in chunk object."""
    chunks = []
    for data in datas:
        chunks.append(Chunk.model_validate(data))
    return chunks


def _load_chunks(index_path: str | Path) -> list[Chunk]:
    """Load chunks from the JSON index."""
    index_path = Path(index_path)
    with index_path.open("r", encoding="utf-8") as file:
        data = json.load(file)
    return _convert_json_in_chunks(data)


def _tokenize(text: str) -> list[str]:
    """Tokenize text while preserving technical identifiers."""
    tokens = re.findall(
        r"[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*|\d+",
        text)
    result = []
    for token in tokens:
        result.append(token.lower())
        parts = re.split(r"[_.]+", token)
        for part in parts:
            subtokens = re.findall(
                r"[A-Z]+(?=[A-Z][a-z]|\d|$)|"
                r"[A-Z]?[a-z]+|[A-Z]+|\d+",
                part)
            result.extend(subtoken.lower()
                          for subtoken in subtokens
                          if subtoken.lower() != token.lower())
    return result


class Retriever:
    """Seach relevant chunks unsing BM25."""

    def __init__(self, index_path: str | Path):
        self.chunks = _load_chunks(index_path)
        self.words_list = self._tokenized_chunk()
        self.bm25 = BM25Okapi(self.words_list) if self.words_list else None

    def _tokenized_chunk(self) -> list[list[str]]:
        words_list = []
        for chunk in self.chunks:
            tokens = _tokenize(chunk.text)
            words_list.append(tokens)
        return words_list

    def search(self, query: str, k: int = 5) -> list[Chunk]:
        """Return the k most relevant chunks for a query."""
        if not query.strip() or k <= 0 or not self.chunks or self.bm25 is None:
            return []
        query_tokens = _tokenize(query)
        scores = self.bm25.get_scores(query_tokens)
        ranked_indices = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True)
        results = []
        for i in ranked_indices[:min(k, len(self.chunks))]:
            results.append(self.chunks[i])
        return results
