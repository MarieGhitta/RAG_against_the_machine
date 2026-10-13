import json
from pathlib import Path
from tqdm import tqdm
from src.chunker import chunk_python, chunk_text
from src.models import Chunk

PYTHON_EXTENSIONS = {".py", ".pyi"}

TEXT_EXTENSIONS = {
    ".md",
    ".txt",
    ".json",
    ".yaml",
    ".yml",
    ".toml",
    ".cmake",
    ".sh",
    ".c",
    ".h",
    ".cpp",
    ".hpp",
    ".cu",
    ".cuh"
}

MAX_CHUNK_SIZE = 2000


def _get_supported_files(raw_dir: Path) -> list[Path]:
    """Return supported source files found recursively in raw_dir."""
    extensions = PYTHON_EXTENSIONS | TEXT_EXTENSIONS
    return sorted(
        path for path in raw_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in extensions
    )


def _chunk_file(path: Path, file_path: str,
                max_chunk_size: int) -> list[Chunk]:
    """Read a source file and split it into chunks."""
    content = path.read_text(encoding="utf-8")
    if path.suffix.lower() in PYTHON_EXTENSIONS:
        return chunk_python(content, file_path, max_chunk_size)
    return chunk_text(content, file_path, max_chunk_size)


def index_corpus(raw_dir: str | Path = "data/raw",
                 output_path: str | Path = "data/processed/index.json",
                 max_chunk_size: int = MAX_CHUNK_SIZE) -> int:
    """Build and save a JSON index of the supported source files."""
    if not 0 < max_chunk_size <= MAX_CHUNK_SIZE:
        raise ValueError(
            f"max_chunk_size must be between 1 and {MAX_CHUNK_SIZE}")
    project_root = Path.cwd().resolve()
    raw_path = Path(raw_dir)
    if not raw_path.is_absolute():
        raw_path = project_root / raw_path
    raw_path = raw_path.resolve()
    try:
        raw_path.relative_to(project_root)
    except ValueError as error:
        raise ValueError(
            "The source directory must be inside the project root."
            ) from error

    if not raw_path.is_dir():
        raise NotADirectoryError(
            f"Source directory does not exist: {raw_path}"
        )
    destination = Path(output_path)
    if not destination.is_absolute():
        destination = project_root / destination
    files = _get_supported_files(raw_path)
    chunks: list[Chunk] = []
    indexed_files = 0
    skipped_files = 0
    for path in tqdm(files, desc="Indexing files", unit="files"):
        try:
            file_path = path.relative_to(project_root).as_posix()
            file_chunks = _chunk_file(path, file_path, max_chunk_size)
            chunks.extend(file_chunks)
            indexed_files += 1
        except (OSError, UnicodeError, ValueError) as error:
            skipped_files += 1
            tqdm.write(f"Skipping {path}; {error}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        with destination.open("w", encoding="utf-8") as file:
            json.dump(
                [chunk.model_dump() for chunk in chunks],
                file,
                ensure_ascii=False,
                indent=2
            )
    except OSError as error:
        raise OSError(
            f"Could not save index to {destination}: {error}"
        ) from error
    print(f"Indexed {len(chunks)} chunks from {indexed_files} files.")
    print(f"Skipped {skipped_files} files out of {len(files)} candidates.")
    print(f"Index saved to {destination}")
    return len(chunks)