
from src.retriever import Retriever
import re

retriever = Retriever("data/processed/index.json")

queries = [
    "How do you achieve reproducible results in vLLM?",
    "What hardware platforms does vLLM support?",
    "What is the default value of trust_remote_code?",
    "What is the default value of trust_remote_code in LLM?",
    "What is the default value of trust_remote_code in LLM.__init__?",
    "trust_remote_code bool False",
    "LLM __init__ trust_remote_code"
]

for query in queries:
    print(f"\nQuestion : {query}")

    results = retriever.search(query, k=5)

    for i, chunk in enumerate(results, start=1):
        print(f"\n{i}. {chunk.file_path}")
        print(chunk.text[:300])

    scores = retriever.bm25.get_scores(query.lower().split())

    ranked_indices = sorted(
        range(len(scores)),
        key=lambda i: scores[i],
        reverse=True,
    )

    print("\n--- Classement des chunks de llm.py ---")

    count = 0
    for i in ranked_indices:
        chunk = retriever.chunks[i]

        if (
            chunk.file_path.endswith("vllm/entrypoints/llm.py")
            and "trust_remote_code" in chunk.text
        ):
            print(f"\nScore : {scores[i]:.4f}")
            print(chunk.text[:500])
            count += 1

            if count == 3:
                break
   
    query = "trust_remote_code bool False"
    scores = retriever.bm25.get_scores(
        re.findall(r"\w+", query.lower())
    )

    ranked_indices = sorted(
        range(len(scores)),
        key=lambda i: scores[i],
        reverse=True,
    )

    for rank, i in enumerate(ranked_indices, start=1):
        chunk = retriever.chunks[i]
        if (
            chunk.file_path.endswith("vllm/entrypoints/llm.py")
            and "trust_remote_code: bool = False" in chunk.text
        ):
            print(f"\nRang global : {rank}")
            print(f"Score : {scores[i]:.4f}")
            print(f"Fichier : {chunk.file_path}")
            break

