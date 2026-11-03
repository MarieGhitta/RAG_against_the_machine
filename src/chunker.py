import ast
from src.models import Chunk
import re


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
        ranges: list[tuple[int, int, str]],
        max_chunk_size: int) -> list[Chunk]:
    """Create chunks from character ranges and their context."""
    chunks = []
    for start, end, context in ranges:
        text = content[start:end]
        if len(text) <= max_chunk_size:
            chunks.append(
                Chunk(
                    text=text,
                    file_path=file_path,
                    first_character_index=start,
                    last_character_index=end,
                    context=context))
        else:
            for chunk_text, chunk_start, chunk_end in _split_text(
                    text, start, max_chunk_size):
                chunks.append(
                    Chunk(
                        text=chunk_text,
                        file_path=file_path,
                        first_character_index=chunk_start,
                        last_character_index=chunk_end,
                        context=context))
    return chunks


def _node_range(node: ast.stmt,
                content_lines: list[str],
                line_pos_char: list[int]) -> tuple[int, int]:
    """Return the character range of an AST node."""
    def byte_to_char(line: str, byte_offset: int) -> int:
        """Convert a UTF-8 byte offset into a character offset."""
        prefix = line.encode("utf-8")[:byte_offset]
        return len(prefix.decode("utf-8"))

    if node.end_lineno is None or node.end_col_offset is None:
        raise ValueError("AST node has no end position")
    start_line = content_lines[node.lineno - 1]
    end_line = content_lines[node.end_lineno - 1]
    start = (line_pos_char[node.lineno - 1]
             + byte_to_char(start_line, node.col_offset))
    end = (line_pos_char[node.end_lineno - 1]
           + byte_to_char(end_line, node.end_col_offset))
    return start, end


def _class_ranges(node: ast.ClassDef,
                  content_lines: list[str],
                  line_pos_char: list[int]
                  ) -> list[tuple[int, int, str]]:
    """Split a class into ranges based on ites direct members."""
    ranges: list[tuple[int, int, str]] = []
    context = f"class {node.name}"
    for member in node.body:
        if not hasattr(member, "lineno"):
            continue
        start, end = _node_range(member,
                                 content_lines,
                                 line_pos_char)
        ranges.append((start, end, context))
    return ranges


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
    content_lines = content.splitlines(keepends=True)
    ranges: list[tuple[int, int, str]] = []
    for node in tree.body:
        if not hasattr(node, "lineno"):
            continue
        if isinstance(node, ast.ClassDef):
            ranges.extend(_class_ranges(node,
                                        content_lines,
                                        line_pos_char))
        else:
            start, end = _node_range(node,
                                     content_lines,
                                     line_pos_char)
            ranges.append((start, end, ""))
    return _create_chunks(
        content,
        file_path,
        ranges,
        max_chunk_size)


def _text_blocks(content: str) -> list[tuple[int, int]]:
    """Return start and end character positions of non-empty text blocks."""
    ranges = []
    start = 0
    for separator in re.finditer(
            r"\r?\n[ \t]*\r?\n+", content):
        end = separator.start()
        if content[start:end].strip():
            ranges.append((start, end))
        start = separator.end()
    if content[start:].strip():
        ranges.append((start, len(content)))
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
    text_ranges = _text_blocks(content)
    merged_ranges = _merge_blocks(text_ranges, max_chunk_size)
    ranges: list[tuple[int, int, str]] = [(start, end, "")
                                          for start, end in merged_ranges]
    return _create_chunks(
        content,
        file_path,
        ranges,
        max_chunk_size)
