#![feature(portable_simd)]

//! This package
//! We begin by defining the core constants for model sizes.
//! Rust note: These are built into the code to allo for full optimization
//! You can select which is used with --cfg model_size=70B.

mod constants;
mod gptq;
mod models;
mod tokenizer;
mod util;

use constants::*;
use memmap2::MmapOptions;
use models::{softmax, transformer, RunState, TWeights};
use std::fs::File;
use std::io;
use std::io::{Seek, SeekFrom, Write};
use std::mem;
use tokenizer::{Token, Tokenizer, RET, START};
use util::{argmax, time_in_ms, Random};

fn print_token(tokenizer: &Tokenizer, token: Token, next: Token) {
    let token_str = if token == 1 && tokenizer.vocab[next].starts_with(' ') {
        &tokenizer.vocab[next][1..]
    } else {
        &tokenizer.vocab[next]
    };
    print!("{}", token_str);
    io::stdout().flush().expect("flush failed");
}

/// For the prompt, fills the caches by running transformer.
/// While not strictly necessary, this can lead to faster
/// Performance by batching computation. Should lead to
/// Identical results.
fn prefill<const A: usize>(
    pos: &mut usize,
    state: &mut RunState,
    prompt_tokens: &Vec<usize>,
    tokenizer: &Tokenizer,
    weights: &TWeights,
) {
    while *pos + A < prompt_tokens.len() {
        let mut tokens = [0; A];
        let mut positions = [0; A];
        let mut fake_logits = [[0.0; VOCAB_SIZE]; 1];
        for i in 0..A {
            let next = if *pos == 0 {
                START
            } else {
                prompt_tokens[*pos - 1]
            };
            positions[i] = *pos;
            tokens[i] = next;
            *pos += 1;
            let token = next;
            print_token(tokenizer, token, next)
        }
        transformer(&mut fake_logits, &tokens, &positions, state, &weights);
    }
}

use clap::Parser;

/// Simple program to greet a person
#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Checkpoint name to use.
    #[arg(short, long)]
    checkpoint: String,

    /// Temperature for generation
    #[arg(short, long, default_value_t = 0.0)]
    temperature: f32,

    /// Number of steps to generate
    #[arg(short, long, default_value_t = 20)]
    steps: usize,

    /// Number of times to greet
    #[arg(short, long)]
    prompt: String,

    /// Number of times to greet
    #[arg(short, long, default_value_t = false)]
    debug: bool,
}

fn main() {
    let args = Args::parse();
    // 'checkpoint' is necessary arg
    let mut random = Random::new();

    // Read in the model.bin file
    let (weights, _mmap): (&'static TWeights, _) = {
        let mut file = File::open(&args.checkpoint).unwrap();
        let conf = Config::load(&mut file);
        if args.debug {
            println!("Configuration: {conf:?}");
        }
        let start = file.seek(SeekFrom::Current(0)).unwrap();
        let mmap = unsafe { MmapOptions::new().offset(start).map(&file).unwrap() };
        assert_eq!(mmap.len(), mem::size_of::<TWeights>());
        (unsafe { &*(mmap.as_ptr() as *const TWeights) }, mmap)
    };

    // Read in the tokenizer.bin file
    let tokenizer = {
        let mut file = File::open("tokenizer.bin").unwrap();
        Tokenizer::load(&mut file)
    };

    // create and init the application RunState
    let mut state: Box<RunState> = RunState::new();

    // process the prompt, if any
    let prompt_tokens = if !args.prompt.is_empty() {
        let prompt = format!(" {}", args.prompt.trim());
        tokenizer.bpe_encode(&prompt)
    } else {
        Vec::new()
    };

    // start the main loop
    let start = time_in_ms(); // used to time our code, only initialized after first iteration
    let mut next; // will store the next token in the sequence
    let mut pos = 0; // position in the sequence

    // Do a little backoff to handle different sizes.This costs us compilation time,
    // But allows us to compile versions of with the longest lnength prefill possible.
    prefill::<64>(&mut pos, &mut state, &prompt_tokens, &tokenizer, &weights);
    prefill::<32>(&mut pos, &mut state, &prompt_tokens, &tokenizer, &weights);
    prefill::<16>(&mut pos, &mut state, &prompt_tokens, &tokenizer, &weights);
    prefill::<8>(&mut pos, &mut state, &prompt_tokens, &tokenizer, &weights);
    prefill::<1>(&mut pos, &mut state, &prompt_tokens, &tokenizer, &weights);

    let mut token: Token = if pos == 0 {
        START
    } else {
        prompt_tokens[pos - 1]
    }; // init with token 1 (=BOS), as done in Llama-2 sentencepiece tokenizer
    print_token(&tokenizer, token, token);
    let mut outputs = Vec::new();
    let mut raw_logits = [[0.0; VOCAB_SIZE]; 1];
    while pos < args.steps {
        // forward the transformer to get logits for the next token
        let tokens = [token];
        let positions = [pos];

        transformer(&mut raw_logits, &tokens, &positions, &mut state, &weights);
        let logits = &mut raw_logits[0];
        if pos < prompt_tokens.len() {
            // if we are still processing the input prompt, force the next prompt token
            next = prompt_tokens[pos];
            // println!("{}", logits[next]);
        } else {
            // sample the next token
            if args.temperature == 0.0 {
                // greedy argmax sampling: take the token with the highest probability
                next = argmax(logits);
                // println!("{}", logits[next]);
            } else {
                // apply the temperature to the logits
                for q in 0..VOCAB_SIZE {
                    logits[q] /= args.temperature;
                }
                // apply softmax to the logits to get the probabilities for next token
                softmax(&mut logits[..VOCAB_SIZE]);
                // we sample from this distribution to get the next token
                next = random.sample(logits, VOCAB_SIZE);
            }
        };
        // following BOS token (1), sentencepiece decoder strips any leading whitespace (see PR #89)
        //println!("{} {}", next, state.logits[next]);
        print_token(&tokenizer, token, next);

        // advance forward
        token = next;
        outputs.push(token);
        let l = outputs.len();

        // Heuristic stopping criteria.
        if l > 6
            && outputs[l - 1] == RET
            && outputs[l - 2] == RET
            && outputs[l - 3] == RET
            && outputs[l - 4] == RET
        {
            break;
        }
        pos += 1;
    }

    // report achieved tok/s
    let end = time_in_ms();
    println!(
        "\nachieved tok/s: {}",
        (pos) as f32 / (end - start) as f32 * 1000.0
    );
}
