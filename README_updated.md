*This project has been created as part of the 42 curriculum by mghitta.*

# RAG Against the Machine

## Description

RAG Against the Machine is a Retrieval-Augmented Generation (RAG) system
for answering questions about the vLLM codebase. It indexes source
files, retrieves relevant snippets using BM25, and uses the local
`Qwen/Qwen3-0.6B` language model to generate answers grounded in the
retrieved context.

The goal is to locate the source-code or documentation passages that
answer a question, then provide a concise answer based on those
passages.

### System architecture

The pipeline consists of four stages:

1.  **Indexing** --- reads supported files from `data/raw/`, splits
    their contents into chunks, and saves the resulting index under
    `data/processed/`.
2.  **Retrieval** --- tokenizes a question and the indexed chunks, then
    ranks candidate chunks using BM25.
3.  **Generation** --- sends the retrieved context and the question to
    `Qwen/Qwen3-0.6B`, which generates an answer based on that context.
4.  **Evaluation** --- compares retrieved source locations with
    reference source locations and reports `Recall@k`.

### Main components

-   `src/chunker.py` --- splits Python files using their AST structure
    and Markdown/text files using paragraph blocks.
-   `src/indexer.py` --- scans the corpus, creates chunks, and writes
    the index.
-   `src/retriever.py` --- tokenizes queries and chunks, builds an
    inverted index for BM25, and ranks matching chunks. Technical
    identifiers are preserved; file paths and Markdown heading context
    can contribute to retrieval.
-   `src/generator.py` --- loads the tokenizer and language model and
    generates an answer from the retrieved context.
-   `src/pipeline.py` --- connects retrieval and generation.
-   `src/models.py` --- defines the Pydantic models used to validate
    chunks, datasets, sources, and outputs.
-   `src/cli.py` --- exposes the command-line interface.

### Chunking strategy

Python files and documentation are segmented differently:

-   **Python files:** the parser uses Python's AST to identify code
    structures and their source ranges.
-   **Markdown and text files:** content is split into paragraph-based
    blocks.

Chunks are limited to 2,000 characters by default. The maximum chunk
size can be configured when indexing. Character offsets are stored with
each chunk so that retrieved sources can be traced back to their
original locations.

### Retrieval method

The retrieval stage uses **BM25**, a lexical ranking method. Text is
tokenized before indexing and querying. Technical identifiers are
preserved, while file paths and Markdown heading context can contribute
to matching.

During index construction, the retriever builds an **inverted index**
(`postings`): each term maps to the document IDs in which it appears and
its frequency in each document. It also precomputes the document
lengths, average document length, and IDF values required by BM25.
At query time, this allows the retriever to score only chunks containing
query terms instead of scanning the entire corpus.

For Markdown files located under `/docs/`, the active heading hierarchy
at each chunk's character offset is added to the retrieval tokens. This
provides topical context even when a chunk does not contain the heading
itself. File-path tokens are also included in the retrieval text.

The retriever returns the top `k` source chunks for each question.

### Design decisions

-   **BM25:** provides a lexical retrieval baseline without requiring a
    separate embedding model.
-   **Different chunking strategies:** Python syntax has a structure
    that can be identified with the AST, while documentation is more
    naturally segmented into text blocks.
-   **Source offsets:** storing character positions makes it possible to
    return exact locations in the original corpus and compare them with
    reference sources.
-   **Local generation model:** `Qwen/Qwen3-0.6B` is used to generate
    answers from retrieved context.

### Challenges and solutions

-   **Different source formats:** Python code and Markdown      documentation require different segmentation strategies, so separate chunking approaches are used.
-   **Character offsets in Python AST:** AST column offsets are
    expressed in UTF-8 bytes. They must be converted to character
    offsets to keep source positions consistent with the character-index
    format expected by the project.
-   **Retrieval vocabulary:** questions may use different wording from the source code. Preserving technical identifiers and including
    file paths in retrieval text helps match both natural-language
    descriptions and code terminology. For Markdown files under
    `/docs/`, the active heading hierarchy is also added as context to
    the tokens of each chunk.
-   **Retrieval performance:** initially, each query required scoring
    the corpus chunks. Building an inverted index (`postings`) makes it
    possible to visit only the chunks containing at least one query
    term. The top results are selected with `heapq.nlargest()` rather
    than sorting every scored chunk.
-   **Index startup time:** building the inverted index and BM25
    statistics is expensive, so these data are saved in a sidecar
    pickle cache (`index.bm25.pkl`). The retriever can load the
    precomputed data on subsequent runs instead of rebuilding them.
    If the cache is missing or incompatible, it can be rebuilt from
    the JSON chunk index.
-   **BM25 parameter tuning:** retrieval quality depends on the balance
    between `k1`, which controls term-frequency saturation, and `b`,
    which controls document-length normalization. These parameters were
    tuned against the evaluation dataset to reach the required
    documentation Recall@5 threshold.
-   **Grounded generation:** the generator is instructed to answer
    using the retrieved context and to state when the context is
    insufficient.

## Instructions

### Requirements

-   Python **3.10 or later**
-   [`uv`](https://docs.astral.sh/uv/)
-   Enough memory and disk space to install the dependencies and load
    `Qwen/Qwen3-0.6B`. The model is downloaded from Hugging Face on
    first use.

Dependencies are declared in `pyproject.toml`. They include Pydantic,
Python Fire, tqdm, rank-bm25, Transformers, PyTorch, Accelerate, and
NLTK. Development dependencies include Flake8, mypy, and pytest.

### Installation

Run these commands from the repository root:

``` bash
uv sync
```

### Prepare the corpus

Place the vLLM repository inside `data/raw/`. The indexer scans this
directory recursively.

Supported extensions include:

-   Python: `.py`, `.pyi`
-   Text and configuration: `.md`, `.txt`, `.json`, `.yaml`, `.yml`,
    `.toml`, `.cmake`, `.sh`
-   C/C++ and CUDA: `.c`, `.h`, `.cpp`, `.hpp`, `.cu`, `.cuh`

The default maximum chunk size is **2,000 characters**. The indexer does
not accept a value above this limit.

### Example usage

Run commands from the project root. The CLI is built with Python Fire
and is invoked with `uv run python -m src <command>`.

#### 1. Build the index

``` bash
uv run python -m src index --max_chunk_size 2000
```

The default index is saved to `data/processed/index.json`. You can
override the input and output paths using the available CLI options.
The BM25 data are also saved automatically in the sidecar cache
`data/processed/index.bm25.pkl` to speed up subsequent retriever
initialization. The cache is generated from the JSON index and can be
recreated if it is deleted; it should not be treated as the source
corpus or as a replacement for `index.json`.

#### 2. Search for one question

``` bash
uv run python -m src search "How does the scheduler work?" --k 5
```

The command prints the retrieved source locations, including each file
path and character range.

#### 3. Generate an answer for one question

``` bash
uv run python -m src answer "How does the scheduler work?" --k 5
```

By default, the pipeline uses the index at `data/processed/index.json`
and the model `Qwen/Qwen3-0.6B`. The `--index_path` and `--model_name`
options can be used to override these defaults.

#### 4. Search a dataset of questions

The dataset must follow the project's Pydantic dataset structure and
contain a `rag_questions` list.

``` bash
uv run python -m src search_dataset \
  --dataset_path data/datasets/UnansweredQuestions/dataset_docs_public.json \
  --k 5 \
  --save_directory data/output/search_results/UnansweredQuestions
```

Search results are written as a `StudentSearchResults` JSON file. Keep
the output directory scoped by dataset (`UnansweredQuestions` or
`AnsweredQuestions`) to avoid overwriting results from another dataset.

#### 5. Evaluate retrieval results

``` bash
uv run python -m src evaluate \
  --student_search_results_path data/output/search_results/UnansweredQuestions/dataset_docs_public.json \
  --dataset_path data/datasets/AnsweredQuestions/dataset_docs_public.json
```

The evaluator reports `Recall@k` against a reference dataset. A
retrieved source matches a reference source when the file paths are
identical and their character ranges have an intersection-over-union
(IoU) of at least 0.05.

#### 6. Generate answers for a dataset

This command consumes previously generated search results, reads the
corresponding source ranges, generates answers, and writes a
`StudentSearchResultsAndAnswer` JSON file.

``` bash
uv run python -m src answer_dataset \
  --student_search_results_path data/output/search_results/UnansweredQuestions/dataset_docs_public.json \
  --save_directory data/output/search_results_and_answer/UnansweredQuestions
```

### Makefile commands

The project includes these convenience targets:

  Command              Purpose
  -------------------- -------------------------------------------------------
  `make install`       Install dependencies with `uv sync`
  `make run`           Run the CLI
  `make debug`         Run the CLI through Python's debugger
  `make clean`         Remove Python cache directories and mypy cache
  `make lint`          Run Flake8 and mypy with the project's required flags
  `make lint-strict`   Run Flake8 and mypy in strict mode

Run the checks with:

``` bash
make install
make run
make lint
make lint-strict
```

## Performance analysis

The project specification sets these minimum retrieval targets on the
reference datasets:

-   **Documentation questions:** at least 80% Recall@5
-   **Code questions:** at least 50% Recall@5

The latest documentation evaluation reported by the author reached
**80% Recall@5**, meeting the required documentation threshold. The
retrieval speed reached at least **27 questions per second** in the
tested environment, comfortably above the required throughput.

Results depend on the evaluation dataset and execution environment.
The official evaluation during the defense is performed by the
provided Moulinette executable.

## Output formats

Search operations write a JSON object conforming to the
`StudentSearchResults` model. It contains:

-   `search_results`
-   `k`

Each search result includes the question ID, question text, and a list
of retrieved sources. Each source contains:

-   `file_path`
-   `first_character_index`
-   `last_character_index`

Answer operations additionally include the generated `answer` and
conform to the `StudentSearchResultsAndAnswer` model.

File paths and character offsets must correspond to the indexed corpus.
Keep the corpus paths stable between indexing and evaluation.

## Resources

### References

-   *Qwen:* 
    https://huggingface.co/Qwen/Qwen3-0.6B
-   *BM25:*
    https://dataaspirant.com/blog/bm25-explained/
    https://dev.to/pykpyky/comprendre-et-implementer-lalgo-de-score-bm25-47af
-   *Python AST:* 
    https://docs.python.org/3/library/ast.html
    https://earthly.dev/blog/python-ast/
-   *Pydantic:*
    https://docs.pydantic.dev/
-   *Pipeline:*
    https://huggingface.co/docs/transformers/en/pipeline_tutorial
-   *Rag:*
    https://huggingface.co/blog/ngxson/make-your-own-rag

### Use of AI

AI tools were used as supplementary assistance during the project. They
helped with understanding concepts and documentation related to RAG,
BM25, Python AST parsing, and language-model generation; exploring and
reviewing chunking and retrieval approaches; investigating errors and
debugging; reviewing code and type-checking issues; and improving the
project documentation.

AI-generated suggestions were reviewed, tested, and adapted as part of
the implementation. The project code and design choices remain the
responsibility of the author.
