from typing import Any

class LlamaModel:
    # we could define Config later
    config: Any
    
    def __new__(cls, checkpoint: str, debug: bool = False) -> LlamaModel:
        ...

class Random:
    def __new__(cls) -> Random:
        ...

class Tokenizer:
    def __new__(cls, filename: str) -> Tokenizer:
        ...

    def bpe_encode(self, text: str) -> list[int]:
        ...

def generate(
    model: LlamaModel,
    tokenizer: Tokenizer,
    prompt: str,
    steps: int,
    random: Random,
    temperature: float = 0.0,
    print_tokens: bool = False
) -> tuple[int, list[str]]:
    ...