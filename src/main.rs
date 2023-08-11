#![feature(portable_simd)]

// This is a conversion of llama2.c to rust.
// It is basically line-by-line following chatgpt :)
use memmap2::MmapOptions;
use rayon::prelude::*;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom, Write};
use std::mem;
use std::time::{SystemTime, UNIX_EPOCH};
use std::{env, io};

// Configuration for Llama 70B. Others in config.txt
// set these configuration options using .cargo/config
#[cfg(model_size = "70B")]
mod constants {
    pub const DIM: usize = 8192;
    pub const HIDDEN_DIM: usize = 28672;
    pub const ATTN_GROUPS: usize = 8;
    pub const N_LAYERS: usize = 80;
    pub const N_HEADS: usize = 64;
    pub const SEQ_LEN: usize = 2048;
    pub const VOCAB_SIZE: usize = 32000;
}

// Llama 13B
#[cfg(model_size = "13B")]
pub mod constants {
    pub const DIM: usize = 5120;
    pub const HIDDEN_DIM: usize = 13824;
    pub const ATTN_GROUPS: usize = 1;
    pub const N_LAYERS: usize = 40;
    pub const N_HEADS: usize = 40;
    pub const SEQ_LEN: usize = 2048;
    pub const VOCAB_SIZE: usize = 32000;
}

// Llama 7B
#[cfg(model_size = "7B")]
mod constants {
    pub const DIM: usize = 4096;
    pub const HIDDEN_DIM: usize = 11008;
    pub const N_LAYERS: usize = 32;
    pub const ATTN_GROUPS: usize = 1;
    pub const N_HEADS: usize = 32;
    pub const SEQ_LEN: usize = 2048;
    pub const VOCAB_SIZE: usize = 32000;
}

// Grouped Query Attention
const KV_DIM: usize = DIM / ATTN_GROUPS;
const N_KV_HEADS: usize = N_HEADS / ATTN_GROUPS;
const HEAD_SIZE: usize = DIM / N_HEADS;
use constants::{ATTN_GROUPS, DIM, HIDDEN_DIM, N_HEADS, N_LAYERS, SEQ_LEN, VOCAB_SIZE};

#[repr(C)]
pub struct Linear<const IN: usize, const OUT: usize> {
    w: [[f32; IN]; OUT],
}

impl<const IN: usize, const OUT: usize> Linear<IN, OUT> {
    pub fn matvec(self: &Self, xout: &mut [f32; OUT], x: &[f32; IN]) {
        // W (d,n) @ x (n,) -> xout (d,)
        // by far the most amount of time is spent inside this little function
        xout.par_iter_mut().enumerate().for_each(|(i, v)| {
            *v = self.w[i]
                .iter()
                .zip(x.iter())
                .fold(0.0, |acc, (&_w, &_x)| acc + _w * _x);
        });
    }
}

#[cfg(quant = "no")]
mod model {
    /// Code for standard non-quantized matrix vector models.

    #[repr(C)]
    #[allow(dead_code)]
    struct TransformerWeights {
        // token embedding table
        pub token_embedding_table: [[f32; DIM]; VOCAB_SIZE], // (vocab_size, dim)
        // weights for rmsnorms
        pub rms_att_weight: [[f32; DIM]; N_LAYERS], // (layer, dim) rmsnorm weights
        // weights for matmuls
        pub wq: [Linear<DIM, DIM>; N_LAYERS], // (layer, dim, dim)
        pub wk: [Linear<DIM, KV_DIM>; N_LAYERS], // (layer, dim, dim)
        pub wv: [Linear<DIM, KV_DIM>; N_LAYERS], // (layer, dim, dim)
        pub wo: [Linear<DIM, DIM>; N_LAYERS], // (layer, dim, dim)

        pub rms_ffn_weight: [[f32; DIM]; N_LAYERS], // (layer, dim)
        // weights for ffn
        pub w1: [Linear<DIM, HIDDEN_DIM>; N_LAYERS], // (layer, hidden_dim, dim)
        pub w2: [Linear<HIDDEN_DIM, DIM>; N_LAYERS], // (layer, dim, hidden_dim)
        pub w3: [Linear<DIM, HIDDEN_DIM>; N_LAYERS], // (layer, hidden_dim, dim)
        // final rmsnorm
        pub rms_final_weight: [f32; DIM], // (dim,)
        // freq_cis for RoPE relatively positional embeddings
        pub freq_cis_real: [[f32; DIM / N_HEADS / 2]; SEQ_LEN], // (seq_len, dim/2)
        pub freq_cis_imag: [[f32; DIM / N_HEADS / 2]; SEQ_LEN], // (seq_len, dim/2)
        // (optional) classifier weights for the logits, on the last layer
        pub wcls: Linear<DIM, VOCAB_SIZE>, // (dim,)
    }

    // Turn off GPT-Q Quantization.
    pub type TWeights = TransformerWeights;
}

// Quant 4 bits
#[cfg(quant = "Q_4_128")]
mod model {
    /// Code for quantized SIMD implementation.
    use rayon::prelude::*;
    use std::simd::{f32x8, i32x8, SimdFloat, SimdInt};

    ///
    const fn int_div_up(x: usize, y: usize) -> usize {
        x / y + if x % y == 0 { 0 } else { 1 }
    }

    use super::{DIM, HIDDEN_DIM, KV_DIM, N_HEADS, N_LAYERS, SEQ_LEN, VOCAB_SIZE};
    const BITS: usize = 4;
    const GROUPSIZE: usize = 128;

    #[repr(C)]
    pub struct QLinear<
        const IN: usize,
        const OUT: usize,
        const GROUPS: usize,
        const ING: usize,
        const OUTG: usize,
    > {
        qweight: [[i32; ING]; OUT],
        qzeros: [[i32; GROUPS]; OUTG],
        scales: [[f32; GROUPS]; OUT],
    }

    impl<
            const IN: usize,
            const OUT: usize,
            const GROUPS: usize,
            const ING: usize,
            const OUTG: usize,
        > QLinear<IN, OUT, GROUPS, ING, OUTG>
    {
        pub fn matvec(self: &Self, xout: &mut [f32; OUT], x: &[f32; IN]) {
            assert_eq!(ING, IN / 32 * BITS);
            assert_eq!(OUTG, OUT / 32 * BITS);
            assert_eq!(GROUPS, int_div_up(IN, GROUPSIZE));

            let mask = (1 << BITS) - 1;
            let elems_per_i32 = 32 / BITS;
            let ipg: usize = GROUPSIZE / 32 * BITS;
            let mask_4bits = i32x8::splat(mask);
            let shift_right = i32x8::from_array([0, 4, 8, 12, 16, 20, 24, 28]);

            xout.par_iter_mut()
                .enumerate()
                .for_each(|(oi, o): (usize, &mut f32)| {
                    *o = 0.0;
                    // Do K at a time
                    let zero = f32x8::splat(0.0);
                    let qzeros = &self.qzeros[oi / elems_per_i32];
                    let out_elem = oi % elems_per_i32;
                    let qweight = self.qweight[oi].chunks_exact(ipg);

                    let collect = self.scales[oi]
                        .into_iter()
                        .zip(qweight)
                        .enumerate()
                        .map(|(group, (scale, qweight))| {
                            let qz = ((qzeros[group] >> (BITS * out_elem)) & mask) + 1;
                            let scale_simd = f32x8::splat(scale);
                            let zero_simd = i32x8::splat(qz);
                            let in_pos = group * GROUPSIZE;
                            let xs = x[in_pos..in_pos + GROUPSIZE].chunks_exact(8);
                            qweight
                                .iter()
                                .zip(xs)
                                .map(|(v, x)| {
                                    //Extract v into 8 chunks
                                    let x = f32x8::from_slice(x);
                                    let num_simd = i32x8::splat(*v);
                                    let qw: i32x8 = (num_simd >> shift_right) & mask_4bits;
                                    let combine: f32x8 = (qw - zero_simd).cast::<f32>();
                                    let weight: f32x8 = scale_simd * combine;
                                    weight * x
                                })
                                .fold(zero, |x, y| x + y)
                        })
                        .fold(zero, |x, y| x + y);
                    *o = collect.reduce_sum();
                });
        }
    }

    const DIM_GROUPS: usize = int_div_up(DIM, GROUPSIZE);
    const HDIM_GROUPS: usize = int_div_up(HIDDEN_DIM, GROUPSIZE);
    const DIM_G: usize = DIM / 32 * BITS;
    const KV_DIM_G: usize = KV_DIM / 32 * BITS;
    const HDIM_G: usize = HIDDEN_DIM / 32 * BITS;

    type Att = [QLinear<DIM, DIM, DIM_GROUPS, DIM_G, DIM_G>; N_LAYERS];
    type AttKV = [QLinear<DIM, KV_DIM, DIM_GROUPS, DIM_G, KV_DIM_G>; N_LAYERS];

    #[repr(C)]
    pub struct QTransformerWeights {
        // token embedding table
        pub token_embedding_table: [[f32; DIM]; VOCAB_SIZE], // (vocab_size, dim)
        // weights for rmsnorms
        pub rms_att_weight: [[f32; DIM]; N_LAYERS], // (layer, dim) rmsnorm weights
        // weights for matmuls
        pub wq: Att,   // (layer, dim, dim)
        pub wk: AttKV, // (layer, dim, dim)
        pub wv: AttKV, // (layer, dim, dim)
        pub wo: Att,   // (layer, dim, dim)

        pub rms_ffn_weight: [[f32; DIM]; N_LAYERS], // (layer, dim)
        // weights for ffn
        pub w1: [QLinear<DIM, HIDDEN_DIM, DIM_GROUPS, DIM_G, HDIM_G>; N_LAYERS], // (layer, hidden_dim, dim)
        pub w2: [QLinear<HIDDEN_DIM, DIM, HDIM_GROUPS, HDIM_G, DIM_G>; N_LAYERS], // (layer, dim, hidden_dim)
        pub w3: [QLinear<DIM, HIDDEN_DIM, DIM_GROUPS, DIM_G, HDIM_G>; N_LAYERS], // (layer, hidden_dim, dim)
        // final rmsnorm
        pub rms_final_weight: [f32; DIM], // (dim,)
        // freq_cis for RoPE relatively positional embeddings
        pub freq_cis_real: [[f32; DIM / N_HEADS / 2]; SEQ_LEN], // (seq_len, dim/2)
        pub freq_cis_imag: [[f32; DIM / N_HEADS / 2]; SEQ_LEN], // (seq_len, dim/2)
        // (optional) classifier weights for the logits, on the last layer
        pub wcls: super::Linear<DIM, VOCAB_SIZE>, // (dim,)
    }

    // Turn on GPT-Q Quantization.
    pub type TWeights = QTransformerWeights;
}

use model::TWeights;

// Some helpers for reading from binary files.
fn read_usize(file: &mut File) -> usize {
    let mut buf = [0u8; 4];
    file.read_exact(&mut buf).unwrap();
    i32::from_le_bytes(buf) as usize
}

fn read_float(file: &mut File) -> f32 {
    let mut buf = [0u8; 4];
    file.read_exact(&mut buf).unwrap();
    f32::from_le_bytes(buf)
}

fn read_str(file: &mut File, len: usize) -> String {
    let mut buf: Vec<u8> = vec![0u8; len];
    file.read_exact(&mut buf).unwrap();
    std::str::from_utf8(&buf).unwrap().to_owned()
}

fn str_lookup(str: &str, vocab: &[String]) -> Option<usize> {
    // find the first perfect match for str in vocab, return its index or -1 if not found
    vocab.into_iter().position(|x| x == str)
}

fn argmax(v: &[f32]) -> usize {
    // return argmax of v in elements 0..n
    v.iter()
        .enumerate()
        .reduce(|a, b| if a.1 > b.1 { a } else { b })
        .unwrap()
        .0
}

#[derive(Debug)]
#[allow(dead_code)]
struct Config {
    dim: usize,        // transformer dimension
    hidden_dim: usize, // for ffn layers
    n_layers: usize,   // number of layers
    n_heads: usize,    // number of query heads
    n_kv_heads: usize, // number of key/value heads (can be < query heads because of multiquery)
    vocab_size: usize, // vocabulary size, usually 256 (byte-level)
    seq_len: usize,    // max sequence length
    shared_weight: bool,
}

// This config is mostly ignored.
// We use it to check compile time constants.
impl Config {
    fn check_static(self: &Self) {
        assert_eq!(self.dim, DIM);
        assert_eq!(self.hidden_dim, HIDDEN_DIM);
        assert_eq!(self.n_layers, N_LAYERS);
        assert_eq!(self.n_heads, N_HEADS);
        assert_eq!(self.seq_len, SEQ_LEN);
        assert_eq!(self.vocab_size, VOCAB_SIZE);
    }

    fn load(file: &mut File) -> Self {
        let mut conf = Config {
            dim: read_usize(file),
            hidden_dim: read_usize(file),
            n_layers: read_usize(file),
            n_heads: read_usize(file),
            n_kv_heads: read_usize(file),
            vocab_size: 0,
            seq_len: 0,
            shared_weight: false,
        };
        let vocab_size = read_usize(file) as i32;
        conf.vocab_size = vocab_size.abs() as usize;
        conf.shared_weight = vocab_size > 0;
        conf.seq_len = read_usize(file);
        println!("Configuration: {conf:?}");
        conf.check_static();
        conf
    }
}

struct RunState {
    // current wave of activations
    x: [f32; DIM],             // activation at current time stamp (dim,)
    xb: [f32; DIM],            // same, but inside a residual branch (dim,)
    xb2: [f32; DIM],           // an additional buffer just for convenience (dim,)
    hb: [f32; HIDDEN_DIM],     // buffer for hidden dimension in the ffn (hidden_dim,)
    hb2: [f32; HIDDEN_DIM],    // buffer for hidden dimension in the ffn (hidden_dim,)
    q: [f32; DIM],             // query (dim,)
    k: [f32; KV_DIM],          // key (dim,)
    v: [f32; KV_DIM],          // value (dim,)
    att: Vec<[f32; SEQ_LEN]>,  // buffer for scores/attention values (n_heads, seq_len)
    logits: [f32; VOCAB_SIZE], // output logits
    // kv cache
    key_cache: Vec<Vec<[[f32; HEAD_SIZE]; N_KV_HEADS]>>, // (layer, seq_len, dim)
    value_cache: Vec<Vec<[[f32; HEAD_SIZE]; N_KV_HEADS]>>, // (layer, seq_len, dim)
}

impl RunState {
    fn new() -> Box<Self> {
        Box::new(RunState {
            x: [0.0; DIM],
            xb: [0.0; DIM],
            xb2: [0.0; DIM],
            hb: [0.0; HIDDEN_DIM],
            hb2: [0.0; HIDDEN_DIM],
            q: [0.0; DIM],
            k: [0.0; KV_DIM],
            v: [0.0; KV_DIM],
            att: vec![[0.0; SEQ_LEN]; N_HEADS],
            logits: [0.0; VOCAB_SIZE],
            key_cache: vec![vec![[[0.0; HEAD_SIZE]; N_KV_HEADS]; SEQ_LEN]; N_LAYERS],
            value_cache: vec![vec![[[0.0; HEAD_SIZE]; N_KV_HEADS]; SEQ_LEN]; N_LAYERS],
        })
    }
}

type Token = usize;

#[derive(Debug)]
struct Tokenizer {
    vocab: Vec<String>,
    vocab_scores: Vec<f32>,
    max_token_length: usize,
}

impl Tokenizer {
    fn load(file: &mut File, config: &Config) -> Tokenizer {
        // read in the tokenizer.bin file
        let mut tokenizer = Tokenizer {
            vocab: Vec::with_capacity(config.vocab_size),
            vocab_scores: Vec::with_capacity(config.vocab_size),
            max_token_length: 0u32 as usize,
        };
        tokenizer.max_token_length = read_usize(file);
        for _ in 0..config.vocab_size {
            tokenizer.vocab_scores.push(read_float(file));
            let len = read_usize(file);
            tokenizer.vocab.push(read_str(file, len));
        }
        tokenizer
    }

    fn bpe_encode(self: &Self, text: &str) -> Vec<Token> {
        let mut tokens: Vec<Token> = Vec::new();
        let mut str_buffer = String::new();

        // first encode every individual byte in the input string
        for c in text.chars() {
            str_buffer.clear();
            str_buffer.push(c);
            let id = str_lookup(&str_buffer, self.vocab.as_slice()).expect("not good");
            tokens.push(id);
        }

        // merge the best consecutive pair each iteration, according the scores in vocab_scores
        loop {
            let mut best_score = -1e10;
            let mut best_id = 0;
            let mut best_idx = None;

            for i in 0..(tokens.len() - 1) {
                // check if we can merge the pair (tokens[i], tokens[i+1])
                str_buffer.clear();
                str_buffer.push_str(&self.vocab[tokens[i]]);
                str_buffer.push_str(&self.vocab[tokens[i + 1] as usize]);
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
                    }
                    tokens.pop(); // token length decreased
                }
            }
        }
    }
}

// ----------------------------------------------------------------------------
// neural net blocks

fn accum(a: &mut [f32], b: &[f32]) {
    for i in 0..a.len() {
        a[i] += b[i];
    }
}

fn rmsnorm(o: &mut [f32; DIM], xo: Option<&[f32]>, weight: &[f32; DIM]) {
    // calculate sum of squares
    let mut ss: f32 = 0.0;
    for i in 0..DIM {
        let x = xo.unwrap_or(o);
        ss += x[i] * x[i];
    }
    ss /= DIM as f32;
    ss += 1e-5;
    ss = 1.0 / ss.sqrt();
    // normalize and scale
    for j in 0..DIM {
        // Solve some borrow nonsense.
        o[j] = weight[j] * ss * (xo.unwrap_or(o)[j])
    }
}

fn softmax(x: &mut [f32]) {
    // find max value (for numerical stability)
    let max_val = x.iter().fold(x[0], |acc, &x_i| acc.max(x_i));
    // exp and sum
    let mut sum = 0.0;
    for x_i in x.iter_mut() {
        *x_i = (*x_i - max_val).exp();
        sum += *x_i;
    }
    // normalize
    for x_i in x.iter_mut() {
        *x_i /= sum;
    }
}

fn dot(q: &[f32], k: &[f32]) -> f32 {
    assert_eq!(q.len(), k.len());
    q.iter()
        .zip(k.iter())
        .map(|(&q_i, &k_i)| q_i * k_i)
        .sum::<f32>()
}

fn transformer(token: usize, pos: usize, s: &mut RunState, w: &TWeights) {
    // a few convenience variables
    let x = &mut s.x;

    // copy the token embedding into x
    x.copy_from_slice(&w.token_embedding_table[token]);

    // pluck out the "pos" row of freq_cis_real and freq_cis_imag
    let freq_cis_real_row = &w.freq_cis_real[pos];
    let freq_cis_imag_row = &w.freq_cis_imag[pos];
    // forward all the layers
    for l in 0..N_LAYERS {
        // attention rmsnorm
        rmsnorm(&mut s.xb, Some(x), &w.rms_att_weight[l]);

        // qkv matvecs for this position
        w.wq[l].matvec(&mut s.q, &s.xb);
        w.wk[l].matvec(&mut s.k, &s.xb);
        w.wv[l].matvec(&mut s.v, &s.xb);

        // apply RoPE rotation to the q and k vectors for each head
        for h in 0..N_HEADS {
            // get the q and k vectors for this head
            let q = &mut s.q[h * HEAD_SIZE..];

            // rotate q and k by the freq_cis_real and freq_cis_imag
            for i in (0..HEAD_SIZE).step_by(2) {
                let q0 = q[i];
                let q1 = q[i + 1];
                let fcr = freq_cis_real_row[i / 2];
                let fci = freq_cis_imag_row[i / 2];
                q[i + 0] = q0 * fcr - q1 * fci;
                q[i + 1] = q0 * fci + q1 * fcr;
                if h < N_KV_HEADS {
                    let k = &mut s.k[h * HEAD_SIZE..];
                    let k0 = k[i];
                    let k1 = k[i + 1];
                    k[i + 0] = k0 * fcr - k1 * fci;
                    k[i + 1] = k0 * fci + k1 * fcr;
                }
            }
        }

        // save key,value at this time step (pos) to our kv cache
        let key_cache_row = &mut s.key_cache[l][pos];
        let value_cache_row = &mut s.value_cache[l][pos];

        for h in 0..N_KV_HEADS {
            for d in 0..HEAD_SIZE {
                key_cache_row[h][d] = s.k[h * HEAD_SIZE + d];
                value_cache_row[h][d] = s.v[h * HEAD_SIZE + d];
            }
        }

        // multihead attention. iterate over all heads
        // We do this a bit differently in rust.
        // Chunk things up so that each head is a separate slice.
        let xbs: Vec<&mut [f32]> = s.xb.chunks_mut(HEAD_SIZE).collect();
        let qs: Vec<&[f32]> = s.q.chunks(HEAD_SIZE).collect();
        assert_eq!(xbs.len(), s.att.len());
        s.att
            .par_iter_mut()
            .zip(xbs)
            .enumerate()
            .for_each(|(h, (att, xb))| {
                // get the query vector for this head
                let q = &qs[h];
                // attention scores for this head
                //let mut att = &mut s.att[h * p.seq_len..];
                // iterate over all timesteps, including the current one
                for t in 0..=pos {
                    // get the key vector for this head and at this timestep
                    let k = &s.key_cache[l][t][h / ATTN_GROUPS];
                    // calculate the attention score as the dot product of q and k
                    let score = dot(q, k) / (HEAD_SIZE as f32).sqrt();
                    // save the score to the attention buffer
                    att[t] = score;
                }

                // softmax the scores to get attention weights, from 0..pos inclusively
                softmax(&mut att[..=pos]);

                // weighted sum of the values, store back into xb
                //let xb = &mut s.xb[h * head_size..];
                for i in 0..HEAD_SIZE {
                    xb[i] = 0.0;
                }
                for t in 0..=pos {
                    // get the value vector for this head and at this timestep
                    let v = &s.value_cache[l][t][h / ATTN_GROUPS];
                    // get the attention weight for this timestep
                    let a = att[t];
                    // accumulate the weighted value into xb
                    for i in 0..HEAD_SIZE {
                        xb[i] += a * v[i];
                    }
                }
            });

        // final matvec to get the output of the attention
        w.wo[l].matvec(&mut s.xb2, &s.xb);

        // residual connection back into x
        accum(x, &s.xb2);

        // ffn rmsnorm
        rmsnorm(&mut s.xb, Some(x), &w.rms_ffn_weight[l]);

        // Now for FFN in PyTorch we have: self.w2(F.silu(self.w1(x)) * self.w3(x))
        // first calculate self.w1(x) and self.w3(x)
        w.w1[l].matvec(&mut s.hb, &s.xb);
        w.w3[l].matvec(&mut s.hb2, &s.xb);

        // F.silu; silu(x)=x*σ(x),where σ(x) is the logistic sigmoid
        for i in 0..HIDDEN_DIM {
            s.hb[i] *= 1.0 / (1.0 + (-s.hb[i]).exp());
        }

        // elementwise multiply with w3(x)
        for i in 0..HIDDEN_DIM {
            s.hb[i] *= s.hb2[i];
        }

        // final matvec to get the output of the ffn
        w.w2[l].matvec(&mut s.xb, &s.hb);

        // residual connection
        accum(x, &s.xb);
    }

    // final rmsnorm
    rmsnorm(x, None, &w.rms_final_weight);

    // classifier into logits
    w.wcls.matvec(&mut s.logits, &s.x);
}

fn time_in_ms() -> i64 {
    // return time in milliseconds, for benchmarking the model speed
    let time = SystemTime::now().duration_since(UNIX_EPOCH).unwrap();
    time.as_secs() as i64 * 1000 + time.subsec_millis() as i64
}

struct Random {
    seed: u64,
}
impl Random {
    fn new() -> Random {
        // seed rng with time. if you want deterministic behavior use temperature 0.0
        Random {
            seed: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_secs() as u64,
        }
    }

    fn random_u32(self: &mut Self) -> u32 {
        // xorshift rng: https://en.wikipedia.org/wiki/Xorshift#xorshift.2A
        self.seed ^= self.seed >> 12;
        self.seed ^= self.seed << 25;
        self.seed ^= self.seed >> 27;
        (self.seed.wrapping_mul(0x2545_f491_4f6c_dd1d) >> 32) as u32
    }

    fn random(self: &mut Self) -> f32 {
        // random float32 in [0,1)
        (self.random_u32() >> 8) as f32 / 16777216.0
    }

    fn sample(self: &mut Self, probabilities: &[f32], n: usize) -> usize {
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

fn main() {
    // poor man's Rust argparse
    // 'checkpoint' is necessary arg
    let args: Vec<String> = env::args().skip(1).collect();
    if args.len() < 1 {
        println!("Usage: <checkpoint_file> [temperature] [steps] [prompt]");
        return;
    }
    let checkpoint = args[0].clone();
    let temperature = if args.len() >= 2 {
        args[1].parse().expect("temperature must be float")
    } else {
        0.9
    };

    let mut steps = if args.len() >= 3 {
        args[2].parse().expect("steps must be int")
    } else {
        256
    };
    let prompt = args.get(3).unwrap_or(&String::from("")).to_owned();

    let mut random = Random::new();
    // read in the model.bin file
    let mut file = File::open(&checkpoint).unwrap();

    // Config
    let config = Config::load(&mut file);
    io::stdout().flush().expect("flush failed");
    let start = file.seek(SeekFrom::Current(0)).unwrap();
    let mmap = unsafe { MmapOptions::new().offset(start).map(&file).unwrap() };
    assert_eq!(mmap.len(), mem::size_of::<TWeights>());
    // let mut content = Vec::new();
    // file.read_to_end(&mut content);
    let weights: &'static TWeights = unsafe { &*(mmap.as_ptr() as *const TWeights) };

    // right now we cannot run for more than config.seq_len steps
    if steps <= 0 || steps > config.seq_len {
        steps = config.seq_len;
    }

    let tokenizer = {
        let mut file = File::open("tokenizer.bin").unwrap();
        Tokenizer::load(&mut file, &config)
    };

    // create and init the application RunState
    let mut state: Box<RunState> = RunState::new();
    // process the prompt, if any
    let prompt_tokens = if !prompt.is_empty() {
        tokenizer.bpe_encode(&prompt)
    } else {
        Vec::new()
    };

    // start the main loop
    let mut start = 0; // used to time our code, only initialized after first iteration
    let mut next; // will store the next token in the sequence
    let mut token: Token = 1; // init with token 1 (=BOS), as done in Llama-2 sentencepiece tokenizer
    let mut pos = 0; // position in the sequence
    println!("<s>"); // explicit print the initial BOS token for stylistic symmetry reasons
    while pos < steps {
        // forward the transformer to get logits for the next token
        transformer(token, pos, &mut state, &weights);
        if pos < prompt_tokens.len() {
            // if we are still processing the input prompt, force the next prompt token
            next = prompt_tokens[pos];
        } else {
            // sample the next token
            if temperature == 0.0 {
                // greedy argmax sampling: take the token with the highest probability
                next = argmax(&state.logits);
            } else {
                // apply the temperature to the logits
                for q in 0..config.vocab_size {
                    state.logits[q] /= temperature;
                }
                // apply softmax to the logits to get the probabilities for next token
                softmax(&mut state.logits[..VOCAB_SIZE]);
                // we sample from this distribution to get the next token
                next = random.sample(&state.logits, config.vocab_size);
            }
        };
        // following BOS token (1), sentencepiece decoder strips any leading whitespace (see PR #89)
        //println!("{} {}", next, state.logits[next]);

        let token_str = if token == 1 && tokenizer.vocab[next].starts_with(' ') {
            &tokenizer.vocab[next][1..]
        } else {
            &tokenizer.vocab[next]
        };
        print!("{}", token_str);
        io::stdout().flush().expect("flush failed");

        // advance forward
        token = next;
        pos += 1;
        // init our timer here because the first iteration is slow due to memmap
        if start == 0 {
            start = time_in_ms();
        }
    }

    // report achieved tok/s
    let end = time_in_ms();
    println!(
        "\nachieved tok/s: {}",
        (steps - 1) as f32 / (end - start) as f32 * 1000.0
    );
}
