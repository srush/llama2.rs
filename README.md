# llama2.rs

This is a one-file Rust implementation of Llama2. Originally, a Rust port of Karpathy's [llama2.c](https://github.com/karpathy/llama2.c) but now has a bunch more features to make it scale to 70B:

* Support for 4-bit GPT-Q Quantization
* Support for Grouped Query Attention (needed for big Llamas)
* Memory mapping, loads 70B instantly.
* Static size checks, no pointers

There is also a faster version with SIMD enabled in the SIMD branch, but it requires Rust +nightly to run.

<img src="https://github.com/srush/llama2.rs/assets/35882/dac9a285-b141-409f-bb46-c81a28516cd1" width=300px>

To build:

```
> cargo build --release
```

To get model (loads [70B quantized](https://huggingface.co/TheBloke/llama-2-70b-Guanaco-QLoRA-GPTQ)):

```
pip install torch transformers auto-gptq
python export.py llama2-70b-q.bin
```

To run:

```
> target/release/llama2_rs llama2-70b-q.bin 0.0 11 "The only thing"
Configuration: Config { dim: 8192, hidden_dim: 28672, n_layers: 80, n_heads: 64, n_kv_heads: 64, vocab_size: 32000, seq_len: 2048, shared_weig
ht: false }                                                                                                                                   
<s> 
The only thing that I can think of is that the          
achieved tok/s: 0.060693607
```

Honestly, not so bad for running on my GPU machine, significantly faster than llama.c. 

Here's a run of 7B non-quantized:

```
> target/release/llama2_rs llama2_7b.bin 0.0 11 "The only thing"
The only thing that is certain in life is change.
achieved tok/s: 1.0298662
```

### Configuration

In order to make the model as fast as possible, you need to compile a new version to adapt to other Llama versions. Currently this is done by a group of constant headers at the top of the file. The model will fail if these disagree with the binary model that is being loaded. 

```rust
// Configuration for Llama 70B. Others in config.txt                                                                                          
const DIM: usize = 8192;                                                                                                                      
const HIDDEN_DIM: usize = 28672;                                                                                                              
const ATTN_GROUPS: usize = 8;                                                                                                                 
const N_LAYERS: usize = 80;                                                                                                                   
const N_HEADS: usize = 64;                                                                                                                    
const SEQ_LEN: usize = 2048;                                                                                                                  
const VOCAB_SIZE: usize = 32000;                                                                                                              
                                                                                                                                              
// Grouped Query Attention                                                                                                                    
const KV_DIM: usize = DIM / ATTN_GROUPS;                                                                                                      
const N_KV_HEADS: usize = N_HEADS / ATTN_GROUPS;                                                                                              
const HEAD_SIZE: usize = DIM / N_HEADS;                                                                                                       
                                                                                                                                              
// Turn on GPT-Q Quantization.                                                                                                                
type TWeights = QTransformerWeights;                                                                                                          
const BITS: usize = 4;                                                                                                                        
const GROUPSIZE: usize = 128; 
```

### See Also

* [llama2.rs](https://github.com/gaxler/llama2.rs) from @gaxler 
* [llama2.rs](https://github.com/leo-du/llama2.rs) from @leo-du
* [candle](https://github.com/LaurentMazare/candle) and candle llama from @LaurentMazare

### How does it work?

Started as a port of the original code, with extra type information to make it easier to extend. 

There are two dependencies: 
* `memmap2`for memory mapping
* `rayon` for parallel computation.

* SIMD enabled support with +nightly.


### Why? 

Mostly this was an exercise in learning some Rust. Was curious how you port over things like memory mapping, parallel processing, and some of the mathematical tricks. 

This is my first Rust project, so if you are an expert I would love a code review!
