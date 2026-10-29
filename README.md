# RAG Against the Machine

A Retrieval-Augmented Generation (RAG) system for answering questions about a codebase. It indexes source files, retrieves relevant snippets with BM25, and uses a small local language model to generate answers grounded in the retrieved context.

## How it works

The pipeline has four stages:

1. **Indexing** — reads supported files from `data/raw`, splits them into chunks, and saves a JSON index.
2. **Retrieval** — ranks chunks for a question using BM25 lexical search.
3. **Generation** — sends the retrieved context and question to `Qwen/Qwen3-0.6B` to generate an answer.
4. **Evaluation** — compares retrieved source locations with the reference sources and reports `Recall@k`.

### Main components

- `src/chunker.py` — splits Python files using their AST structure and Markdown/text files using paragraph blocks.
- `src/indexer.py` — scans the corpus and writes the index.
- `src/retriever.py` — tokenizes queries and chunks, then ranks chunks with BM25. Technical identifiers are preserved, and Markdown headings and file paths can contribute to retrieval.
- `src/generator.py` — loads the tokenizer and language model and generates an answer from the retrieved context.
- `src/pipeline.py` — connects retrieval and generation.
- `src/models.py` — defines the Pydantic models used to validate chunks, datasets, sources, and output.
- `src/cli.py` — exposes the command-line operations.

## Requirements

- Python **3.10 or later**
- [`uv`](https://docs.astral.sh/uv/)
- Enough memory and disk space to load the `Qwen/Qwen3-0.6B` model and its dependencies. The model is downloaded on first use.

Dependencies are declared in `pyproject.toml`. They include Pydantic, Python Fire, tqdm, rank-bm25, Transformers, PyTorch, Accelerate, and NLTK. Development dependencies include Flake8, mypy, and pytest.

## Installation

From the project root:

```bash
uv sync
```

This creates the project environment and installs the declared dependencies.

## Prepare the corpus

Place the codebase to index inside `data/raw/`. By default, the indexer scans this directory recursively.

Supported extensions include:

- Python: `.py`, `.pyi`
- Text and configuration: `.md`, `.txt`, `.json`, `.yaml`, `.yml`, `.toml`, `.cmake`, `.sh`
- C/C++ and CUDA: `.c`, `.h`, `.cpp`, `.hpp`, `.cu`, `.cuh`

The default maximum chunk size is **2,000 characters**. The indexer does not accept a value above this limit.

## Usage

Run commands from the project root. The CLI is built with Python Fire, so methods of the `CLI` class are exposed as commands.

### 1. Build the index

```bash
uv run python -m src index
```

With custom options:

```bash
uv run python -m src index --max_chunk_size=2000 --raw_dir=data/raw --output_path=data/processed/index.json
```

The default index path is `data/processed/index.json`.

### 2. Search for relevant snippets

Search the index for one question:

```bash
uv run python -m src search --query="How does the scheduler work?" --k=5
```

The command prints the retrieved source locations, including each file path and character range.

### 3. Generate an answer

```bash
uv run python -m src answer --query="How does the scheduler work?" --k=5
```

By default, the pipeline uses:

- Index: `data/processed/index.json`
- Model: `Qwen/Qwen3-0.6B`

You can provide `--index_path` or `--model_name` to override these defaults.

### 4. Search a dataset of questions

The dataset should follow the project’s Pydantic dataset structure, with a `rag_questions` list. Search results are saved as JSON under `data/output/search_results/` by default.

```bash
uv run python -m src search_dataset --dataset_path=data/datasets/public/AnsweredQuestions/dataset_docs_public.json --k=5
```

To evaluate the retrieval results against a reference dataset:

```bash
uv run python -m src evaluate --student_search_results_path=data/output/search_results/dataset_docs_public.json --dataset_path=data/datasets/public/AnsweredQuestions/dataset_docs_public.json
```

The evaluator reports `Recall@k`. Reference and retrieved sources are considered a match when their file paths are identical and their character ranges overlap.

### 5. Generate answers for a dataset

This command consumes previously generated search results, reads the corresponding source ranges, generates answers, and writes the results to JSON.

```bash
uv run python -m src answer_dataset --student_search_results_path=data/output/search_results/dataset_docs_public.json
```

Generated results are saved under `data/output/search_results_and_answer/` by default.

> **Note:** The dataset paths above are examples. Adjust them to match the dataset files available in your checkout. Run commands from the repository root so that relative paths resolve correctly.

## Makefile commands

The project includes these convenience targets:

| Command | Purpose |
| --- | --- |
| `make install` | Install dependencies with `uv sync` |
| `make run` | Run the CLI |
| `make debug` | Run the CLI through Python's debugger (see note below) |
| `make clean` | Remove Python cache directories and mypy cache |
| `make lint` | Run Flake8 and mypy with the project's required flags |
| `make lint-strict` | Run Flake8 and mypy in strict mode |

Use:

```bash
make install
make run
make lint
```

## Evaluation

The retrieval metric is **Recall@k**: it measures how much of the reference source evidence is recovered among the top `k` retrieved snippets. The project specification sets these minimum targets on the reference datasets:

- **Documentation questions:** at least 80% Recall@5
- **Code questions:** at least 50% Recall@5

Run evaluation separately for documentation and code datasets, using the matching reference dataset for each set of search results. Scores depend on the corpus, chunking strategy, retrieval configuration, and value of `k`.

## Output and data formats

Search operations write a JSON object containing:

- `search_results`
- `k`

Each search result includes the question ID, question text, and a list of retrieved sources. A source contains:

- `file_path`
- `first_character_index`
- `last_character_index`

Answer operations additionally include the generated `answer`.

File paths and character offsets must correspond to the indexed corpus. Keep the corpus paths stable between indexing and evaluation.

## Development checks

Run the configured checks with:

```bash
make lint
```

Run tests with:

```bash
uv run pytest
```

## Limitations

- Retrieval uses lexical BM25 matching; questions that use very different wording from the source may be harder to match.
- The generated answer is limited by the retrieved context and the capabilities of the small local model. It can be incomplete or incorrect, so answers should be checked against the source code.
- The model may need to be downloaded the first time generation is run.

## Project goal

This project demonstrates an end-to-end RAG workflow for codebase question answering: indexing a repository, retrieving evidence, generating a context-grounded response, and evaluating retrieval quality with Recall@k.

# Ressouces:

## Web resources: 
- https://huggingface.co/Qwen/Qwen3-0.6B
- https://dataaspirant.com/blog/bm25-explained/

## AI ressources:
AI was used throughout the project to supplement various documentation and sources, to help understand the many concepts related to this project. AI was also used to conduct code reviews and to provide ideas for breaking through certain impasses. 