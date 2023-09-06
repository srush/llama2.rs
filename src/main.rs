#![feature(portable_simd)]

//! This package
//! We begin by defining the core constants for model sizes.
//! Rust note: These are built into the code to allo for full optimization
//! You can select which is used with --cfg model_size=70B.

use llama2_rs::inference::generate;
use llama2_rs::tokenizer::Tokenizer;
use llama2_rs::util::Random;
use llama2_rs::LlamaModel;

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
    let model = LlamaModel::from_file(&args.checkpoint, args.debug);

    // Read in the tokenizer.bin file
    let tokenizer = Tokenizer::new("tokenizer.bin");

    let (gen_time, ret) = generate(
        &model,
        &tokenizer,
        &args.prompt,
        args.steps,
        &mut random,
        args.temperature,
        true,
    );

    // report achieved tok/s
    println!(
        "\nachieved tok/s: {}",
        ((ret.len() - 1) as f32 / gen_time as f32) * 1000.0
    );
}
