# llama2.rs

This is a working Rust port of the https://github.com/srush/llama2.c

To build:

```bash
cargo build
wget https://huggingface.co/karpathy/tinyllamas/resolve/main/stories15M.bin
```

To Run 

```bash
$ target/release/llama2_rs stories15M.bin  0.0 20 
<s> Once upon a time, a little girl was very happy. She was a little little girl who loved
achieved tok/s: 121.79487
```

It seems like it is pretty fast. Waiting to get access to llama2 so I can benchmark it for real. 

### Dependencies

This relies on `memmap2`for (unsafe) memory mapping and `rayon` as a replacement of openmp. 

### Why? 

Mostly this was an exercise in learning some Rust. Was curious how you port over things like memory mapping, parallel processing, and some of the mathematical tricks. Generally it works pretty well. Feel like the Rust code is more readable, safer, and doesn't really sacrifice much. 

This is my first Rust project, so if you are an expert I would love a code review!
