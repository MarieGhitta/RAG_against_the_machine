import json
from pathlib import Path
from src.models import Chunk
from rank_bm25 import BM25Okapi
import re
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
import nltk



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
        self.bm25 = BM25Okapi(self.tokenized_corpus) if self.tokenized_corpus else None
        self.tokenized_corpus = [self._preprocess(self.words_list) for word in self.words_list]

    nltk.download('punkt')
    stemmer = PorterStemmer()
    def _preprocess(self):
        tokens = word_tokenize(self)
        return [stemmer.stem(token) for token in tokens]

    def _tokenized_chunk(self) -> list[list[str]]:
        words_list = []
        for chunk in self.chunks:
            tokens = re.findall(r"\w+", chunk.text.lower())
            words_list.append(tokens)
        return words_list

    def search(self, query: str, k: int = 5) -> list[Chunk]:
        """Return the k most relevant chunks for a query."""
        if not query.strip() or k <= 0 or not self.chunks or self.bm25 is None:
            return []
        query_tokens = re.findall(r"\w+", query.lower())
        scores = self.bm25.get_scores(query_tokens)
        ranked_indices = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True)
        results = []
        for i in ranked_indices[:min(k, len(self.chunks))]:
            results.append(self.chunks[i])
        return results
