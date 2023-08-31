use std::fs::File;

#[cfg(feature = "python")]
use pyo3::{pyclass, pymethods};

use crate::constants::VOCAB_SIZE;
use crate::util::{read_float, read_str, read_usize, str_lookup};
pub type Token = usize;

pub const START: Token = 1;
pub const RET: Token = 13;

#[derive(Debug)]
#[cfg_attr(feature = "python", pyclass)]
pub struct Tokenizer {
    pub vocab: Vec<String>,
    vocab_scores: Vec<f32>,
    max_token_length: usize,
}

impl Tokenizer {
    pub fn load(file: &mut File) -> Tokenizer {
        // read in the tokenizer.bin file
        let mut tokenizer = Tokenizer {
            vocab: Vec::with_capacity(VOCAB_SIZE),
            vocab_scores: Vec::with_capacity(VOCAB_SIZE),
            max_token_length: 0u32 as usize,
        };
        tokenizer.max_token_length = read_usize(file);
        for _i in 0..VOCAB_SIZE {
            tokenizer.vocab_scores.push(read_float(file));
            let len = read_usize(file);
            tokenizer.vocab.push(read_str(file, len));
            //println!("{} {:?} {:?}", i, tokenizer.vocab[i], tokenizer.vocab_scores[i]);
        }
        tokenizer
    }
}

impl Tokenizer {
    pub fn new(filename: &str) -> Tokenizer {
        let mut file = File::open(filename).expect(format!("Failed to open {}", filename).as_str());
        Tokenizer::load(&mut file)
    }

    pub fn bpe_encode(&self, text: &str) -> Vec<Token> {
        let mut tokens: Vec<Token> = Vec::new();
        let mut digits: Vec<bool> = Vec::new();
        let mut spaces: Vec<bool> = Vec::new();
        let mut str_buffer = String::new();

        // first encode every individual byte in the input string
        for c in text.chars() {
            str_buffer.clear();
            str_buffer.push(c);
            let id = str_lookup(&str_buffer, self.vocab.as_slice()).expect("not good");
            tokens.push(id);
            digits.push(c.is_digit(10));
            spaces.push(c == ' ');
        }

        // merge the best consecutive pair each iteration, according the scores in vocab_scores
        loop {
            let mut best_score = -1e10;
            let mut best_id = 0;
            let mut best_idx = None;

            for i in 0..(tokens.len() - 1) {
                // check if we can merge the pair (tokens[i], tokens[i+1])
                str_buffer.clear();
                // if digits[i] || (digits[i+1] && !spaces[i]) {
                //     // Split digits (from sentence piece)
                //     continue;
                // }
                let item1 = &self.vocab[tokens[i]];
                let item2 = &self.vocab[tokens[i + 1]];
                str_buffer.push_str(item1);
                str_buffer.push_str(item2);
                let id = str_lookup(&str_buffer, self.vocab.as_slice());
                match id {
                    Some(id) => {
                        if self.vocab_scores[id] > best_score {
                            // this merge pair exists in vocab! record its score and position
                            best_score = self.vocab_scores[id];
                            best_id = id;
                            best_idx = Some(i);
                        }
                    }
                    None => {}
                }
            }
            match best_idx {
                None => return tokens, // we couldn't find any more pairs to merge, so we're done
                Some(best_idx) => {
                    // merge the consecutive pair (best_idx, best_idx+1) into new token best_id
                    tokens[best_idx] = best_id;

                    // delete token at position best_idx+1, shift the entire sequence back 1
                    for i in (best_idx + 1)..(tokens.len() - 1) {
                        tokens[i] = tokens[i + 1];
                        digits[i] = digits[i + 1];
                        spaces[i] = spaces[i + 1];
                    }
                    tokens.pop(); // token length decreased
                }
            }
        }
    }
}

#[cfg(feature = "python")]
#[pymethods]
impl Tokenizer {
    #[new]
    pub fn new_py(filename: &str) -> Tokenizer {
        Tokenizer::new(filename)
    }

    #[pyo3(name = "bpe_encode")]
    pub fn bpe_encode_py(&self, text: &str) -> Vec<Token> {
        self.bpe_encode(text)
    }
}