from transformers import (AutoModelForCausalLM,
                          AutoTokenizer,
                          BatchEncoding)
from src.models import Chunk
from typing import cast, Any


class Generator:
    def __init__(self, model_name: str):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
                        model_name,
                        torch_dtype="auto",
                        device_map="auto")

    def generate(self, query: str, chunks: list[Chunk]) -> str:
        if not chunks:
            return "Insufficient context to answer the question."
        context = "\n\n".join(chunk.text for chunk in chunks)
        messages = [{
            "role": "system",
            "content": (
                "Answer the question using only the provided context."
                "If the context is insufficient, say so."
                "Do not repeat the answer."
                "Do not repeat the answer. /no_think"
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
                        return_dict=True))
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
        return answer.strip()
