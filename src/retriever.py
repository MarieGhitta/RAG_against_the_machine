import json
from pathlib import Path
from src.models import Chunk
from rank_bm25 import BM25Okapi


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


class Retriever:
    """Seach relevant chunks unsing BM25."""

    def __init__(self, index_path: str | Path):
        self.chunks = _load_chunks(index_path)
        self.words_list = self._tokenized_chunk()
        self.bm25 = BM25Okapi(self.words_list)

    def _tokenized_chunk(self) -> list[list[str]]:
        words_list = []
        for chunk in self.chunks:
            words_list.append(chunk.text.lower().split())
        return words_list


    def search(self, query: str, k: int = 5) -> list[Chunk]:
        """Return the k most relevant chunks for a query."""
        if not query.strip() or k <= 0 or not self.chunks:
            return []
        query_tokens = query.lower().split()
        scores = self.bm25.get_scores(query_tokens)
        ranked_indices = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True)
        return [
            self.chunks[i] for i in ranked_indices[:min(k, len(self.chunks))]
            ]

