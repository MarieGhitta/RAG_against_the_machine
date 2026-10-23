from pathlib import Path
from src.generator import Generator
from src.models import Chunk
from src.retriever import Retriever


DEFAULT_INDEX_PATH = "data/processed/index.json"
DEFAULT_MODEL_NAME = "Qwen/Qwen3-0.6B"


class Pipeline:
    """Orchestrate retrieval and answer generation."""

    def __init__(
            self,
            index_path: str | Path = DEFAULT_INDEX_PATH,
            model_name: str = DEFAULT_MODEL_NAME) -> None:
        """Initialize the retriever and the language model."""
        self.retriever = Retriever(index_path)
        self.generator = Generator(model_name)

    def answer(self, query: str, k: int = 5) -> tuple[str, list[Chunk]]:
        """Generate an answer and return it with its source chunks."""
        if not query.strip():
            return "Question cannot be empty.", []
        if k <= 0:
            return "k must be greater than 0.", []
        chunks = self.retriever.search(query, k)
        if not chunks:
            return (
                "Insufficient context to answer the question.", [])
        answer = self.generator.generate(query, chunks)
        return answer, chunks