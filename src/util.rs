use std::fs::File;
use std::io::{Read, self, Write};
use std::time::{SystemTime, UNIX_EPOCH};

#[cfg(feature = "python")]
use pyo3::{pyclass, pymethods};

use crate::tokenizer::{Tokenizer, Token};

pub fn read_float(file: &mut File) -> f32 {
    let mut buf = [0u8; 4];
    file.read_exact(&mut buf).unwrap();
    f32::from_le_bytes(buf)
}

pub fn read_str(file: &mut File, len: usize) -> String {
    let mut buf: Vec<u8> = vec![0u8; len];
    file.read_exact(&mut buf).unwrap();
    std::str::from_utf8(&buf).unwrap().to_owned()
}

// Some helpers for reading from binary files.
pub fn read_usize(file: &mut File) -> usize {
    let mut buf = [0u8; 4];
    file.read_exact(&mut buf).unwrap();
    i32::from_le_bytes(buf) as usize
}

pub fn str_lookup(str: &str, vocab: &[String]) -> Option<usize> {
    //! find the first perfect match for str in vocab, return its index or -1 if not found
    let x = vocab
        .into_iter()
        .skip(260)
        .position(|x| x == str)
        .map(|x| x + 260);
    x.or(vocab.into_iter().position(|x| x == str))
}

pub fn argmax(v: &[f32]) -> usize {
    //! return argmax of v in elements 0..n
    v.iter()
        .enumerate()
        .reduce(|a, b| if a.1 > b.1 { a } else { b })
        .unwrap()
        .0
}

pub fn time_in_ms() -> i64 {
    // return time in milliseconds, for benchmarking the model speed
    let time = SystemTime::now().duration_since(UNIX_EPOCH).unwrap();
    time.as_secs() as i64 * 1000 + time.subsec_millis() as i64
}

// Probably should just use Rust random. This came from llama2.c
#[cfg_attr(feature = "python", pyclass)]
pub struct Random {
    seed: u64,
}

impl Random {
    pub fn new() -> Random {
        // seed rng with time. if you want deterministic behavior use temperature 0.0
        Random {
            seed: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_secs() as u64,
        }
    }

    fn random_u32(&mut self) -> u32 {
        // xorshift rng: https://en.wikipedia.org/wiki/Xorshift#xorshift.2A
        self.seed ^= self.seed >> 12;
        self.seed ^= self.seed << 25;
        self.seed ^= self.seed >> 27;
        (self.seed.wrapping_mul(0x2545_f491_4f6c_dd1d) >> 32) as u32
    }

    fn random(&mut self) -> f32 {
        // random float32 in [0,1)
        (self.random_u32() >> 8) as f32 / 16777216.0
    }
}

#[cfg(feature = "python")]
#[pymethods]
impl Random {
    #[new]
    pub fn new_py() -> Random {
        Random::new()
    }
}

impl Default for Random {
    fn default() -> Self {
        Self::new()
    }
}

// this isn't exposed to Python since we don't have a conversion to &[f32]
impl Random {
    pub fn sample(&mut self, probabilities: &[f32], n: usize) -> usize {
        // sample index from probabilities, they must sum to 1
        let r = self.random();
        let mut cdf = 0.0;
        for i in 0..n {
            cdf += probabilities[i];
            if r < cdf {
                return i;
            }
        }
        n - 1 // in case of rounding errors
    }
}

pub fn get_token(tokenizer: &Tokenizer, token: Token, next: Token) -> &str {
    let token_str = if token == 1 && tokenizer.vocab[next].starts_with(' ') {
        &tokenizer.vocab[next][1..]
    } else {
        &tokenizer.vocab[next]
    };
    token_str
}

pub fn print_token(token_str: &str) {
    print!("{}", token_str);
    io::stdout().flush().expect("flush failed");
}