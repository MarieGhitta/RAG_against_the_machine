import ast
from src.models import Chunk


def _line_pos_char(content: str) -> list[int]:
    """Return the character position of the beginning of each line."""
    pos_char = [0]
    for line in content.splitlines(keepends=True):
        pos_char.append(pos_char[-1] + len(line))
    return pos_char


def _split_text(
        text: str,
        start: int,
        max_chunk_size: int) -> list[tuple[str, int, int]]:
    """Split text into chunks of at most max_chunk_size characters."""
    chunks = []
    for position in range(0, len(text), max_chunk_size):
        end = min(position + max_chunk_size, len(text))
        chunks.append(
            (text[position:end],
             start + position,
             start + end))
    return chunks


def _create_chunks(
        content: str,
        file_path: str,
        ranges: list[tuple[int, int]],
        max_chunk_size: int) -> list[Chunk]:
    chunks = []
    for start, end in ranges:
        text = content[start:end]
        if len(text) <= max_chunk_size:
            chunks.append(
                Chunk(
                    text=text,
                    file_path=file_path,
                    first_character_index=start,
                    last_character_index=end))
        else:
            for chunk_text, chunk_start, chunk_end in _split_text(
                    text, start, max_chunk_size):
                chunks.append(
                    Chunk(
                        text=chunk_text,
                        file_path=file_path,
                        first_character_index=chunk_start,
                        last_character_index=chunk_end))
    return chunks


def chunk_python(
        content: str,
        file_path: str,
        max_chunk_size: int) -> list[Chunk]:
    """Split Python code according to its AST structure."""
    if not content:
        return []
    if max_chunk_size <= 0:
        raise ValueError("max_chunk_size must be greater than 0")
    try:
        tree = ast.parse(content)
    except SyntaxError:
        return []
    line_pos_char = _line_pos_char(content)
    ranges = []
    for node in tree.body:
        if not hasattr(node, "lineno"):
            continue
        start = line_pos_char[node.lineno - 1] + node.col_offset
        end = line_pos_char[node.end_lineno - 1] + node.end_col_offset
        ranges.append((start, end))
    return _create_chunks(
        content,
        file_path,
        ranges,
        max_chunk_size)


def _text_blocks(content: str) -> list[tuple[int, int]]:
    """Return start and end character positions of non-empty text blocks."""
    ranges = []
    start = 0
    for block in content.split("\n\n"):
        end = start + len(block)
        if block.strip():
            ranges.append((start, end))
        start = end + len("\n\n")
    return ranges


def _merge_blocks(
        ranges: list[tuple[int, int]],
        max_chunk_size: int) -> list[tuple[int, int]]:
    """Merge consecutive blocks without exceeding max_chunk_size."""
    if not ranges:
        return []
    merged = []
    current_start, current_end = ranges[0]
    for start, end in ranges[1:]:
        if end - current_start <= max_chunk_size:
            current_end = end
        else:
            merged.append((current_start, current_end))
            current_start = start
            current_end = end
    merged.append((current_start, current_end))
    return merged


def chunk_text(
        content: str,
        file_path: str,
        max_chunk_size: int) -> list[Chunk]:
    """Split Markdown or text into paragraph-based chunks."""
    if not content:
        return []
    if max_chunk_size <= 0:
        raise ValueError("max_chunk_size must be greater than 0.")
    ranges = _text_blocks(content)
    ranges = _merge_blocks(ranges, max_chunk_size)
    return _create_chunks(
        content,
        file_path,
        ranges,
        max_chunk_size)

