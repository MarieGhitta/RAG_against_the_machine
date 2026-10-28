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
    active = []
    for line_char_pos, level, title in headings:
        if line_char_pos > character_index:
            break
        active = [item for item in active if item[0] < level]
        active.append((level, title))
    return " ".join(title for _, title in active)


class Retriever:
    """Seach relevant chunks unsing BM25."""

    def __init__(self, index_path: str | Path):
        self.chunks = _load_chunks(index_path)
        self.words_list = self._tokenized_chunk()
        self.bm25 = BM25Okapi(self.words_list, k1=2, b=0.5
                              ) if self.words_list else None

    def _tokenized_chunk(self) -> list[list[str]]:
        words_list = []
        headings_cache = {}
        for chunk in self.chunks:
            tokens = _tokenize(chunk.text)
            path_tokens = _tokenize(chunk.file_path)
            tokens.extend(path_tokens * 4)
            if (chunk.file_path.lower().endswith(".md")
               and "/docs/" in chunk.file_path):
                if chunk.file_path not in headings_cache:
                    headings_cache[chunk.file_path] = _markdown_headings(
                        chunk.file_path)
                context = _active_headings_context(
                    headings_cache[chunk.file_path],
                    chunk.first_character_index)
                tokens.extend(_tokenize(context) * 2)
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
