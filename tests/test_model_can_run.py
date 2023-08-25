"""
NOTE! This test is meant to be run locally, since there's no obvious way to distribute the large binaries that
are required at the moment. In the future this will change.

Mostly this code is here as an example for users of the Python interface.
"""
import llama2_rs

def test_llama2_13b_4_128act_can_generate():
    model = llama2_rs.LlamaModel("llama2_13b_4_128act.bin", False)
    tokenizer = llama2_rs.Tokenizer("tokenizer.bin")
    random = llama2_rs.Random()
    response = llama2_rs.generate(
        model,
        tokenizer,
        "Tell me zero-cost abstractions in Rust ",
        50,
        random, 
        0.0
    )
    
    # mostly we're asserting that we didn't crash so far.
    assert isinstance(response[0], int)
    assert isinstance(response[1], list)