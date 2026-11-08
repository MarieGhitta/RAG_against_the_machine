import heapq
import json
import math
import pickle
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any
from src.models import Chunk


CACHE_VERSION = 1
DEFAULT_K1 = 0.5
DEFAULT_B = 0.5
BM25_EPSILON = 0.25


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


def _markdown_headings(file_path: str) -> list[tuple[int, int, str]]:
    """Return md headings with their character line position and levels."""
    path = Path(file_path)
    if path.suffix.lower() != ".md" or not path.is_file():
        return []
    try:
        content = path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return []
    headings = []
    line_char_pos = 0
    for line in content.splitlines(keepends=True):
        match = re.match(
            r"^\s{0,3}(#{1,6})\s+(.+?)\s*#*\s*(?:\r?\n)?$",
            line)
        if match:
            headings.append(
                (line_char_pos, len(match.group(1)), match.group(2).strip()))
        line_char_pos += len(line)
    return headings


def _active_headings_context(headings: list[tuple[int, int, str]],
                             character_index: int) -> str:
    """Return the Markdown heading hierarchy at a character offset."""
    active: list[tuple[int, str]] = []
    for line_char_pos, level, title in headings:
        if line_char_pos > character_index:
            break
        active = [item for item in active if item[0] < level]
        active.append((level, title))
    return " ".join(title for _, title in active)


def _cache_path(index_path: str | Path) -> Path:
    """Return the sidecar path used for the precomputed BM25 index."""
    return Path(index_path).with_suffix(".bm25.pkl")


def _build_documents(chunks: list[Chunk]) -> list[list[str]]:
    """Tokenize all chunks and enrich documentation with heading context."""
    documents: list[list[str]] = []
    headings_cache: dict[str, list[tuple[int, int, str]]] = {}
    for chunk in chunks:
        tokens = _tokenize(f"{chunk.context} {chunk.text}")
        tokens.extend(_tokenize(chunk.file_path) * 3)
        if (chunk.file_path.lower().endswith("md")
                and "/docs/" in chunk.file_path):
            if chunk.file_path not in headings_cache:
                headings_cache[chunk.file_path] = _markdown_headings(
                    chunk.file_path)
            context = _active_headings_context(
                headings_cache[chunk.file_path],
                chunk.first_character_index)
            tokens.extend(_tokenize(context) * 3)
        documents.append(tokens)
    return documents


def _build_postings(
        documents: list[list[str]]) -> tuple[
            dict[str, dict[int, int]], list[int], dict[str, float], float]:
    """Build inverted postings and BM25 statistics."""
    postings: dict[str, dict[int, int]] = defaultdict(dict)
    document_lengths: list[int] = []
    document_count = len(documents)
    for document_id, tokens in enumerate(documents):
        document_lengths.append(len(tokens))
        frequencies = Counter(tokens)
        for term, frequency in frequencies.items():
            postings[term][document_id] = frequency
    average_length = (
        sum(document_lengths) / document_count if document_count else 0.0)
    raw_idf: dict[str, float] = {}
    for term, term_postings in postings.items():
        document_frequency = len(term_postings)
        raw_idf[term] = (
            math.log(document_count - document_frequency + 0.5)
            - math.log(document_frequency + 0.5))
    average_idf = (
        sum(raw_idf.values()) / len(raw_idf) if raw_idf else 0.0)
    idf = {
        term: (BM25_EPSILON * average_idf if value < 0 else value)
        for term, value in raw_idf.items()}
    return dict(postings), document_lengths, idf, average_length


def save_bm25_cache(chunks: list[Chunk],
                    index_path: str | Path,
                    k1: float = DEFAULT_K1,
                    b: float = DEFAULT_B) -> Path:
    """Precompute and persist the inverted BM25 index."""
    if k1 <= 0 or not 0 <= b <= 1:
        raise ValueError("BM25 requires k1 > 0 and 0 <= b <= 1.")
    documents = _build_documents(chunks)
    postings, lengths, idf, average_length = _build_postings(documents)
    cache = {
        "version": CACHE_VERSION,
        "chunks": chunks,
        "postings": postings,
        "document_lengths": lengths,
        "idf": idf,
        "average_length": average_length,
        "k1": k1,
        "b": b}
    destination = _cache_path(index_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    with temporary.open("wb") as file:
        pickle.dump(cache, file, protocol=pickle.HIGHEST_PROTOCOL)
    temporary.replace(destination)
    return destination


class Retriever:
    """Seach relevant chunks unsing BM25."""

    def __init__(self, index_path: str | Path):
        self.index_path = Path(index_path)
        cache_path = _cache_path(self.index_path)
        cache = self._load_cache(cache_path)
        if cache is None:
            self.chunks = _load_chunks(self.index_path)
            documents = _build_documents(self.chunks)
            (self.postings, self.document_lengths,
             self.idf, self.average_length) = _build_postings(documents)
            self.k1 = DEFAULT_K1
            self.b = DEFAULT_B
        else:
            self.chunks = cache["chunks"]
            self.postings = cache["postings"]
            self.document_lengths = cache["document_lengths"]
            self.idf = cache["idf"]
            self.average_length = cache["average_length"]
            self.k1 = cache["k1"]
            self.b = cache["b"]

    def _load_cache(self, cache_path: Path) -> dict[str, Any] | None:
        """Load a compatible precomputed cache, if one exists."""
        if not cache_path.is_file():
            return None
        try:
            with cache_path.open("rb") as file:
                cache = pickle.load(file)
            if (isinstance(cache, dict)
                    and cache.get("version") == CACHE_VERSION):
                return cache
        except (OSError, pickle.PickleError, EOFError, AttributeError,
                ImportError, ValueError):
            return None
        return None

    def search(self, query: str, k: int = 5) -> list[Chunk]:
        """Return the k most relevant chunks for a query."""
        if (not query or not query.strip() or k <= 0
                or not self.chunks or self.average_length <= 0):
            return []
        query_terms = Counter(_tokenize(query))
        if not query_terms:
            return []
        scores: dict[int, float] = defaultdict(float)
        for term, query_frequency in query_terms.items():
            term_postings = self.postings.get(term)
            term_idf = self.idf.get(term)
            if not term_postings or term_idf is None:
                continue
            for document_id, term_frequency in term_postings.items():
                document_length = self.document_lengths[document_id]
                denominator = (term_frequency + self.k1 * (
                    1 - self.b + self.b * document_length
                    / self.average_length))
                contribution = (
                    term_idf * term_frequency * (self.k1 + 1)
                    / denominator)
                scores[document_id] += query_frequency * contribution
        if not scores:
            return []
        best = heapq.nlargest(
            min(k, len(scores)),
            scores.items(),
            key=lambda item: (item[1], -item[0]))
        return [self.chunks[document_id] for document_id, _ in best]
