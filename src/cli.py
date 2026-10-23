import json
from pathlib import Path
from tqdm import tqdm
from src.indexer import index_corpus
from src.models import (
    MinimalAnswer,
    MinimalSearchResults,
    MinimalSource,
    RagDataset,
    StudentSearchResults,
    StudentSearchResultsAndAnswer,
    Chunk)
from src.pipeline import Pipeline
from src.retriever import Retriever


DEFAULT_INDEX_PATH = "data/processed/index.json"
DEFAULT_MODEL_NAME = "Qwen/Qwen3-0.6B"


def _load_json(path: str | Path) -> object:
    """Load JSON data from a file."""
    with Path(path).open("r", encoding="utf-8") as file:
        return json.load(file)


def _save_json(data: object, path: str | Path) -> Path:
    """Save JSON data to a file, creating its parent directory."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)
    return destination


def _output_path(dataset_path: str | Path,
                 save_directory: str | Path) -> Path:
    """Build the output filename from the dataset filename."""
    return Path(save_directory) / Path(dataset_path).name


def _source_from_chunk(chunk: Chunk) -> MinimalSource:
    """Convert a retrieved chunk to its source location."""
    return MinimalSource(
        file_path=chunk.file_path,
        first_character_index=chunk.first_character_index,
        last_character_index=chunk.last_character_index)


def _read_sources(sources: list[MinimalSource]) -> list[str]:
    """Read the exact text ranges represented by source locations."""
    texts = []
    for source in sources:
        path = Path(source.file_path)
        with path.open("r", encoding="utf-8") as file:
            content = file.read()
        texts.append(
            content[
                source.first_character_index:
                source.last_character_index])
    return texts


class CLI:
    """Expose the RAG system through command-line operations."""

    def index(self,
              max_chunk_size: int = 2000,
              raw_dir: str = "data/raw",
              output_path: str = "data/processed/index.json") -> None:
        """Index the source corpus."""
        index_corpus(raw_dir, output_path, max_chunk_size)

    def search(self,
               query: str,
               k: int = 5,
               index_path: str = str(DEFAULT_INDEX_PATH)) -> None:
        """Search for the most relevant source chunks."""
        if not query.strip():
            print("Error: query cannot be empty.")
            return
        if k <= 0:
            print("Error: k must be greater than 0.")
            return
        try:
            retriever = Retriever(index_path)
            chunks = retriever.search(query, k)
            results = [_source_from_chunk(chunk).model_dump()
                       for chunk in chunks]
            print(json.dumps(results, ensure_ascii=False, indent=2))
        except (OSError, ValueError, json.JSONDecodeError) as error:
            print(f"Error: {error}")

    def search_dataset(self,
                       dataset_path: str,
                       k: int = 5,
                       save_directory: str = "data/output/search_results",
                       index_path: str = str(DEFAULT_INDEX_PATH)) -> None:
        """Search every question in a dataset and save the results."""
        if k <= 0:
            print("Error: k must be greater than 0.")
            return 
        try:
            dataset = RagDataset.model_validate(_load_json(dataset_path))
            retriever = Retriever(index_path)
            results = []
            for question in tqdm(
                 dataset.rag_questions,
                 desc="Searching questions",
                 unit="questions"):
                chunks = retriever.search(question.question, k)
                results.append(
                    MinimalSearchResults(
                        question_id=question.question_id,
                        question=question.question,
                        retrieved_sources=[
                            _source_from_chunk(chunk)
                            for chunk in chunks]))
            output = StudentSearchResults(
                search_results=results,
                k=k)
            destination = _output_path(dataset_path, save_directory)
            _save_json(output.model_dump(), destination)
            print(f"Saved search results to {destination}")
        except (
             OSError,
             ValueError,
             json.JSONDecodeError) as error:
            print(f"Error: {error}")