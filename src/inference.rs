#[cfg(feature = "python")]
use pyo3::pyfunction;

use crate::constants::VOCAB_SIZE;
use crate::models::TWeights;
use crate::models::{softmax, transformer, RunState};
use crate::tokenizer::{Token, Tokenizer, START};
use crate::util::{argmax, time_in_ms, Random};
use crate::util::{get_token, print_token};
use crate::LlamaModel;

/// For the prompt, fills the caches by running transformer.
/// While not strictly necessary, this can lead to faster
/// Performance by batching computation. Should lead to
/// Identical results.
pub fn prefill<const A: usize>(
    pos: &mut usize,
    state: &mut RunState,
    prompt_tokens: &Vec<usize>,
    tokenizer: &Tokenizer,
    weights: &TWeights,
    print_tokens: bool,
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
            if print_tokens {
                print_token(get_token(tokenizer, token, next));
            }
        }
        transformer(&mut fake_logits, &tokens, &positions, state, weights);
    }
}

// NOTE: this pyo3 signature only applies to Python code (i.e. in Rust code you must specify print_tokens).
#[cfg_attr(feature = "python", pyfunction)]
#[cfg_attr(feature = "python", pyo3(signature = (model, tokenizer, prompt, steps, random, temperature=0.0, print_tokens=false)))]
pub fn generate(
    model: &LlamaModel,
    tokenizer: &Tokenizer,
    prompt: &str,
    steps: usize,
    random: &mut Random,
    temperature: f32,
    print_tokens: bool,
) -> (i64, Vec<String>) {
    // create and init the application RunState
    let mut state: Box<RunState> = RunState::new();

    // process the prompt, if any
    let prompt_tokens = if !prompt.is_empty() {
        let prompt = format!(" {}", prompt.trim());
        tokenizer.bpe_encode(&prompt)
    } else {
        Vec::new()
    };

    let mut ret = Vec::new(); // will store generated tokens

    // start the main loop
    let start: i64 = time_in_ms(); // used to time our code, only initialized after first iteration
    let mut next; // will store the next token in the sequence
    let mut pos = 0; // position in the sequence

    // Do a little backoff to handle different sizes.This costs us compilation time,
    // But allows us to compile versions of with the longest lnength prefill possible.
    if model.prefill() {
        prefill::<64>(
            &mut pos,
            &mut state,
            &prompt_tokens,
            tokenizer,
            model.weights(),
            print_tokens,
        );
        prefill::<32>(
            &mut pos,
            &mut state,
            &prompt_tokens,
            tokenizer,
            model.weights(),
            print_tokens,
        );
        prefill::<16>(
            &mut pos,
            &mut state,
            &prompt_tokens,
            tokenizer,
            model.weights(),
            print_tokens,
        );
        prefill::<8>(
            &mut pos,
            &mut state,
            &prompt_tokens,
            tokenizer,
            model.weights(),
            print_tokens,
        );
        prefill::<4>(
            &mut pos,
            &mut state,
            &prompt_tokens,
            tokenizer,
            model.weights(),
            print_tokens,
        );
        prefill::<2>(
            &mut pos,
            &mut state,
            &prompt_tokens,
            tokenizer,
            model.weights(),
            print_tokens,
        );
    }
    let mut token: Token = if pos == 0 {
        START
    } else {
        prompt_tokens[pos - 1]
    }; // init with token 1 (=BOS), as done in Llama-2 sentencepiece tokenizer

    if print_tokens {
        let token_str = get_token(tokenizer, token, token);
        print_token(token_str);
    }

    let mut outputs = Vec::new();
    let mut raw_logits = [[0.0; VOCAB_SIZE]; 1];
    while pos < steps {
        // forward the transformer to get logits for the next token
        let tokens = [token];
        let positions = [pos];

        transformer(
            &mut raw_logits,
            &tokens,
            &positions,
            &mut state,
            model.weights(),
        );
        let logits = &mut raw_logits[0];
        if pos < prompt_tokens.len() {
            // if we are still processing the input prompt, force the next prompt token
            next = prompt_tokens[pos];
            // println!("{}", logits[next]);
        } else {
            // sample the next token
            if temperature == 0.0 {
                // greedy argmax sampling: take the token with the highest probability
                next = argmax(logits);
                // println!("{}", logits[next]);
            } else {
                // apply the temperature to the logits
                for q in 0..VOCAB_SIZE {
                    logits[q] /= temperature;
                }
                // apply softmax to the logits to get the probabilities for next token
                softmax(&mut logits[..VOCAB_SIZE]);
                // we sample from this distribution to get the next token
                next = random.sample(logits, VOCAB_SIZE);
            }
        };

        // following BOS token (1), sentencepiece decoder strips any leading whitespace (see PR #89)
        //println!("{} {}", next, state.logits[next]);
        let token_str = get_token(tokenizer, token, next);
        ret.push(token_str.to_string());
        if print_tokens {
            print_token(token_str);
        }

        // advance forward
        token = next;
        outputs.push(token);
        let _l = outputs.len();

        pos += 1;
    }

    ((time_in_ms() - start), ret)
}
