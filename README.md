# llama2.rs

This is a Rust implementation of Llama2 inference on CPU.

The goal is to be as fast as possible. 🤗

It has the following features:

* Support for 4-bit GPT-Q Quantization
* Batched prefill of prompt tokens
* SIMD support for fast CPU inference
* Memory mapping, loads 70B instantly.
* Static size checks throughout
* Support for Grouped Query Attention (needed for big Llamas)

Can run up on *1 tok/s* 70B Llama2 and *9 tok/s* 7B Llama2. (on my intel i9 desktop) 

<img src="https://github.com/srush/llama2.rs/assets/35882/dac9a285-b141-409f-bb46-c81a28516cd1" width=300px>

To build, you'll need the nightly toolchain, which is used by default:

```bash
> rustup toolchain install nightly # to get nightly
> ulimit -s 10000000 # Increase your stack memory limit. 
```

If you get a build error you may need to change `.cargo/config` to match your chipset.

You can load models from the Hugging Face hub. For example this creates a version of a [70B quantized](https://huggingface.co/TheBloke/llama-2-70b-Guanaco-QLoRA-GPTQ)) model with 4 bit quant and 64 sized groups:

```
> pip install torch
> pip install transformers auto-gptq
> python export.py l70b.act64.bin TheBloke/llama-2-70b-Guanaco-QLoRA-GPTQ gptq-4bit-64g-actorder_True
```

The library needs to be *recompiled* to match the model. You do that by editing the `.cargo/config`. In this case it would be. 

```
 "--cfg", 'model_size="70B"', 
 "--cfg", 'quant="Q_4"', 
 "--cfg", 'group_size="64"'
```

To run:

```
> cargo build --release
> target/release/llama2_rs -c llama2-70b-q.bin -t 0.0 -s 11 "The only thing"                                                                                                                                 
The only thing that I can think of is that the          
achieved tok/s: 0.89155835
```

Honestly, not so bad for running on my GPU machine, significantly faster than llama.c. 

Here's a run of 13B quantized:

```bash
target/release/llama2_rs -c l13orca.act.bin -t 0.0 -s 25 -p "Hello to all the cool people out there who "
> Hello to all the cool people out there who are reading this. I hope you are having a great day. I am here
achieved tok/s: 5.1588936
```

Here's a run of 7B quantized:

```bash
target/release/llama2_rs -c l7.ack.bin -t 0.0 -s 25 -p "Hello to all the cool people out there who "
> Hello to all the cool people out there who are reading this. I am a newbie here and I am looking for some
achieved tok/s: 9.048136
```

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
* SIMD enabled support with `portable_simd`

### Why? 

Mostly this was an exercise in learning some Rust. Was curious how you port over things like memory mapping, parallel processing, and some of the mathematical tricks. 

This is my first Rust project, so if you are an expert I would love a code review!
