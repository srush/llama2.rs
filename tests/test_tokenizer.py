import torch
from transformers import AutoTokenizer

import llama2_rs

model_name_or_path = "TheBloke/llama-2-13B-Guanaco-QLoRA-GPTQ"
model_basename = "model" 

def test_tokenizer_1():
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)
    tokenizer_rs = llama2_rs.Tokenizer("tokenizer.bin")

    # NOTE! this doesn't work if there's a space at the beginning or end of the prompt
    TEST_PROMPTS = [
        "The climate change need to be properly addressed through the",
        "Please give me 200 words about deep learning.",
        # " Can you teach me about lifetimes in Rust using examples?",
        # "I want to learn about the Rust borrow checker. ",
    ]
    for prompt in TEST_PROMPTS:
        tokens_python = tokenizer(prompt, return_tensors='pt').input_ids.view(-1)
        tokens_rs = torch.tensor(tokenizer_rs.bpe_encode(" " + prompt.strip()))
        assert (tokens_python[1:] == tokens_rs).all(), prompt