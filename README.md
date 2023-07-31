# llama2.rs

This is a one-file Rust implementation of Llama2.
It's rust port of Karpathy's [llama2.c](https://github.com/karpathy/llama2.c)

To build:

```
> cargo build --release
```

To run (follow instructions to get [llama2_7b.bin](https://github.com/karpathy/llama2.c).)

```
> target/release/llama2_rs ../llama2.c/llama2_7b.bin 0.0 11 "The only thing"
The only thing that is certain in life is change.
achieved tok/s: 0.92618316

```

It actually seems like it is pretty fast! On my computer this is the speed and output of running the original llama2.c

```
> ./run llama2_7b.bin 0.0 11 "The only thing"
The only thing that is certain in life is change.
achieved tok/s: 0.139889
```

### How does it work?

This is basically a port of the original code, with extra type information to make it easier to extend. 

There are two dependencies: 
* `memmap2`for memory mapping
* `rayon` for parallel computation. 

### Why? 

Mostly this was an exercise in learning some Rust. Was curious how you port over things like memory mapping, parallel processing, and some of the mathematical tricks. 

This is my first Rust project, so if you are an expert I would love a code review!
