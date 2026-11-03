from transformers import (AutoModelForCausalLM,
                          AutoTokenizer,
                          BatchEncoding)
from src.models import Chunk
from typing import cast, Any


class Generator:
    """Generate answers from retrieved context using a language model."""
    def __init__(self, model_name: str):
        """Initialize the tokenizer and language model."""
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
                        model_name,
                        torch_dtype="auto",
                        device_map="auto")

    def generate(self, query: str, chunks: list[Chunk]) -> str:
        """Generate an answer using the retrieved chunks as context."""
        if not chunks:
            return "Insufficient context to answer the question."
        context = "\n\n".join(chunk.text for chunk in chunks)
        messages = [{
            "role": "system",
            "content": (
                "Answer the question using only the provided context.\n"
                "If the context is insufficient, say so clearly.\n"
                "Do not invent information or use external knowledge.\n"
                "Be concise and answer the question directly.\n"
                "/no_think"
            )},
            {
            "role": "user",
            "content": f"Context:\n{context}\n\nQuestion: {query}"}]
        inputs = cast(BatchEncoding,
                      self.tokenizer.apply_chat_template(
                        messages,
                        tokenize=True,
                        add_generation_prompt=True,
                        return_tensors="pt",
                        return_dict=True,
                        enable_thinking=False))
        inputs = inputs.to(self.model.device)
        model = cast(Any, self.model)
        outputs = model.generate(**inputs,
                                 max_new_tokens=300)
        input_length = inputs["input_ids"].shape[1]
        generated_tokens = outputs[0][input_length:]
        decoded = self.tokenizer.decode(
            generated_tokens,
            skip_special_tokens=True)
        if isinstance(decoded, list):
            answer = " ".join(decoded)
        else:
            answer = decoded
        answer = answer.strip()
        if "</think>" in answer:
            answer = answer.split("</think>", 1)[-1].strip()
        return answer
