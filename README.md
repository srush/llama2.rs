# llama2.rs 🤗

This is a Rust implementation of Llama2 inference on CPU

The goal is to be as fast as possible. 

It has the following features:

* Support for 4-bit GPT-Q Quantization 
* Batched prefill of prompt tokens
* SIMD support for fast CPU inference
* Memory mapping, loads 70B instantly.
* Static size checks for safety
* Support for Grouped Query Attention (needed for big Llamas)
* Python calling API 

Can run up on *1 tok/s* 70B Llama2 and *9 tok/s* 7B Llama2. (on my intel i9 desktop) 

<img src="https://github.com/srush/llama2.rs/assets/35882/dac9a285-b141-409f-bb46-c81a28516cd1" width=300px>

To build, you'll need the nightly toolchain, which is used by default:

```bash
> rustup toolchain install nightly # to get nightly
> ulimit -s 10000000 # Increase your stack memory limit. 
```

You can load models from the Hugging Face hub. For example this creates a version of a [70B quantized](https://huggingface.co/TheBloke/llama-2-70b-Guanaco-QLoRA-GPTQ)) model with 4 bit quant and 64 sized groups:

Make sure you pick a GPTQ model in the GGUF format. Take note of the parameter size, size and if the model is quantized or not.

```
> pip install -r requirements.export.txt
> python export.py l70b.act64.bin TheBloke/llama-2-70b-Guanaco-QLoRA-GPTQ gptq-4bit-64g-actorder_True
```

The library needs to be *recompiled* to match the model. You can do this with cargo.

To run:

```
> cargo run --release --features 70B,group_64,quantized -- -c llama2-70b-q.bin -t 0.0 -s 11 -p "The only thing"                                                                                                                                 
The only thing that I can think of is that the          
achieved tok/s: 0.89155835
```

Honestly, not so bad for running on my GPU machine, significantly faster than llama.c. 

Here's a run of 13B quantized:

```bash
> cargo run --release --features 13B,group_128,quantized -- -c l13orca.act.bin -t 0.0 -s 25 -p "Hello to all the cool people out there who "
> cargo run --release --features 13B,group_128,quantized -- -c l13orca.act.bin -t 0.0 -s 25 -p "Hello to all the cool people out there who "
Hello to all the cool people out there who are reading this. I hope you are having a great day. I am here
achieved tok/s: 5.1588936
```

Here's a run of 7B quantized:

```bash
cargo run --release --features 7B,group_128,quantized -- -c l7.ack.bin -t 0.0 -s 25 -p "Hello to all the cool people out there who "
cargo run --release --features 7B,group_128,quantized -- -c l7.ack.bin -t 0.0 -s 25 -p "Hello to all the cool people out there who "
> Hello to all the cool people out there who are reading this. I am a newbie here and I am looking for some
achieved tok/s: 9.048136
```

### Python

To run in Python, you need to first compile from the main directory with the python flag.
Before you build and run the python model we need to set up the build file to match the model.
Please adjsut the `features` parameter e.g:

`features = ["pyo3/extension-module", "python", "7B", "group_128", "quantized"]`

```bash
cargo build --release --features 7B,group_128,python,quantized
pip install .
```

You can then run the following code.

```python
import llama2_rs

def test_llama2_13b_4_128act_can_generate():
    model = llama2_rs.LlamaModel("lorca13b.act132.bin", False)
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
```


### Todos

* [ ] Support fast GPU processing with Triton
* [ ] Support https://github.com/oobabooga/text-generation-webui
* [ ] Documentation
* [ ] Blog Post about the methods for fast gptq
* [ ] Remove dependency on AutoGPTQ for preloading
* [ ] Support for safetensors directly. 


### Configuration

In order to make the model as fast as possible, you need to compile a new version to adapt to other Llama versions. Currently in `.cargo/config`. The model will fail if these disagree with the binary model that is being loaded. To turn quantization off set it to quant="no".

### See Also

Originally, a Rust port of Karpathy's [llama2.c](https://github.com/karpathy/llama2.c) but now has a bunch more features to make it scale to 70B.

Also check out:

* [llama2.rs](https://github.com/gaxler/llama2.rs) from @gaxler 
* [llama2.rs](https://github.com/leo-du/llama2.rs) from @leo-du
* [candle](https://github.com/LaurentMazare/candle) and candle llama from @LaurentMazare

### How does it work?

Started as a port of the original code, with extra type information to make it easier to extend. 

There are some dependencies: 
* `memmap2`for memory mapping
* `rayon` for parallel computation.
* `clap` for command-line args.
* `pyO3` for python calling
* SIMD enabled support with `portable_simd`

### Authors

Llama2.rs is written by @srush and @rachtsingh.
