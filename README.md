# llama2.rs

This is a one-file Rust implementation of Llama2 that works pretty well. 
It's Rust port of Karpathy's [llama2.c](https://github.com/karpathy/llama2.c)

<img src="https://github.com/srush/llama2.rs/assets/35882/dac9a285-b141-409f-bb46-c81a28516cd1" width=300px>


To build:

```
> cargo build --release
```

To run (follow instructions to get [llama2_7b.bin](https://github.com/karpathy/llama2.c).)

```
> target/release/llama2_rs llama2_7b.bin 0.0 11 "The only thing"
The only thing that is certain in life is change.
achieved tok/s: 1.0298662
```

It actually seems like it is pretty fast! On my computer this is the speed and output of running the original llama2.c

```
> ./run llama2_7b.bin 0.0 11 "The only thing"
The only thing that is certain in life is change.
achieved tok/s: 0.139889
```
### See Also

* [llama2.rs](https://github.com/gaxler/llama2.rs) from @gaxler 
* [llama2.rs](https://github.com/leo-du/llama2.rs) from @leo-du
* [candle](https://github.com/LaurentMazare/candle) and candle llama from @LaurentMazare

### How does it work?

This is basically a port of the original code, with extra type information to make it easier to extend. 

There are two dependencies: 
* `memmap2`for memory mapping
* `rayon` for parallel computation.

Todo: 
* [ ] - Generic over floating point size
* [ ] - Faster matrix multiplications
* [ ] - More safety, remove some of the C hacks. 

### Why? 

Mostly this was an exercise in learning some Rust. Was curious how you port over things like memory mapping, parallel processing, and some of the mathematical tricks. 

This is my first Rust project, so if you are an expert I would love a code review!
