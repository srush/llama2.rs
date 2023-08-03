use std::str::from_utf8_unchecked;
// This is a conversion of llama2.c to rust.
// It is basically line-by-line following chatgpt :)
use memmap2::MmapOptions;
use rayon::prelude::*;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom, Write};
use std::mem;
use std::time::{SystemTime, UNIX_EPOCH};
use std::{env, io};

// Profile stories 15M
const DIM: usize = 288;
const HIDDEN_DIM: usize = 768;
const N_LAYERS: usize = 6;
const N_HEADS: usize = 6;
const N_KV_HEADS: usize = 6;
const SEQ_LEN: usize = 256;
const VOCAB_SIZE: usize = 32000;
const SHARED_SIZE: usize = 0;
const HEAD_SIZE: usize = DIM / N_HEADS;

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
    unsafe { from_utf8_unchecked(&buf).to_owned() }
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

impl Config {
    fn check_static(self: &Self) {
        assert_eq!(self.dim, DIM);
        assert_eq!(self.hidden_dim, HIDDEN_DIM);
        assert_eq!(self.n_layers, N_LAYERS);
        assert_eq!(self.n_heads, N_HEADS);
        assert_eq!(self.n_kv_heads, N_KV_HEADS);
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
        conf.check_static();
        conf
    }
}

struct RunState {
    // current wave of activations
    x: [f32; DIM],                  // activation at current time stamp (dim,)
    xb: [f32; DIM],                 // same, but inside a residual branch (dim,)
    xb2: [f32; DIM],                // an additional buffer just for convenience (dim,)
    hb: [f32; HIDDEN_DIM],          // buffer for hidden dimension in the ffn (hidden_dim,)
    hb2: [f32; HIDDEN_DIM],         // buffer for hidden dimension in the ffn (hidden_dim,)
    q: [f32; DIM],                  // query (dim,)
    k: [f32; DIM],                  // key (dim,)
    v: [f32; DIM],                  // value (dim,)
    att: [[f32; SEQ_LEN]; N_HEADS], // buffer for scores/attention values (n_heads, seq_len)
    logits: [f32; VOCAB_SIZE],      // output logits
    // kv cache
    key_cache: [[[[f32; HEAD_SIZE]; N_HEADS]; SEQ_LEN]; N_LAYERS], // (layer, seq_len, dim)
    value_cache: [[[[f32; HEAD_SIZE]; N_HEADS]; SEQ_LEN]; N_LAYERS], // (layer, seq_len, dim)
}

impl RunState {
<<<<<<< HEAD
    fn new() -> Self {
=======
    fn new(p: &Config) -> Self {
        // I'm too lazy
>>>>>>> 6c881aa (typed)
        RunState {
            x: [0.0; DIM],
            xb: [0.0; DIM],
            xb2: [0.0; DIM],
            hb: [0.0; HIDDEN_DIM],
            hb2: [0.0; HIDDEN_DIM],
            q: [0.0; DIM],
            k: [0.0; DIM],
            v: [0.0; DIM],
            att: [[0.0; SEQ_LEN]; N_HEADS],
            logits: [0.0; VOCAB_SIZE],
            key_cache: [[[[0.0; HEAD_SIZE]; N_HEADS]; SEQ_LEN]; N_LAYERS],
            value_cache: [[[[0.0; HEAD_SIZE]; N_HEADS]; SEQ_LEN]; N_LAYERS],
        }
    }
}

#[repr(C)]
struct TransformerWeights {
    // token embedding table
    token_embedding_table: [[f32; DIM]; VOCAB_SIZE], // (vocab_size, dim)
    // weights for rmsnorms
    rms_att_weight: [[f32; DIM]; N_LAYERS], // (layer, dim) rmsnorm weights
    // weights for matmuls
    wq: [[[f32; DIM]; DIM]; N_LAYERS], // (layer, dim, dim)
    wk: [[[f32; DIM]; DIM]; N_LAYERS], // (layer, dim, dim)
    wv: [[[f32; DIM]; DIM]; N_LAYERS], // (layer, dim, dim)
    wo: [[[f32; DIM]; DIM]; N_LAYERS], // (layer, dim, dim)

    rms_ffn_weight: [[f32; DIM]; N_LAYERS], // (layer, dim)
    // weights for ffn
    w1: [[[f32; DIM]; HIDDEN_DIM]; N_LAYERS], // (layer, hidden_dim, dim)
    w2: [[[f32; HIDDEN_DIM]; DIM]; N_LAYERS], // (layer, dim, hidden_dim)
    w3: [[[f32; DIM]; HIDDEN_DIM]; N_LAYERS], // (layer, hidden_dim, dim)
    // final rmsnorm
    rms_final_weight: [f32; DIM], // (dim,)
    // freq_cis for RoPE relatively positional embeddings
    freq_cis_real: [[f32; DIM / N_HEADS / 2]; SEQ_LEN], // (seq_len, dim/2)
    freq_cis_imag: [[f32; DIM / N_HEADS / 2]; SEQ_LEN], // (seq_len, dim/2)
    // (optional) classifier weights for the logits, on the last layer
    wcls: [[f32; DIM]; SHARED_SIZE], // (dim,)
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

fn rmsnorm(o: &mut [f32], xo: Option<&[f32]>, weight: &[f32; DIM]) {
    // calculate sum of squares
    let mut ss: f32 = 0.0;
    for i in 0..DIM {
        let x = xo.unwrap_or(o);
        ss += x[i] * x[i];
    }
    ss /= o.len() as f32;
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

fn matmul<const N: usize, const D: usize>(xout: &mut [f32; D], x: &[f32; N], w: &[[f32; N]; D]) {
    // W (d,n) @ x (n,) -> xout (d,)
    // by far the most amount of time is spent inside this little function
    xout.par_iter_mut().enumerate().for_each(|(i, v)| {
        *v = w[i]
            .iter()
            .zip(x.iter())
            .fold(0f32, |acc, (&_w, &_x)| acc + _w * _x);
    });
}

fn dot(q: &[f32], k: &[f32]) -> f32 {
    assert_eq!(q.len(), k.len());
    q.iter()
        .zip(k.iter())
        .map(|(&q_i, &k_i)| q_i * k_i)
        .sum::<f32>()
}

fn transformer(
    token: usize,
    pos: usize,
    s: &mut RunState,
    w: &TransformerWeights,
    last_layer: &[[f32; DIM]; VOCAB_SIZE],
) {
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

        // qkv matmuls for this position
        matmul(&mut s.q, &s.xb, &w.wq[l]);
        matmul(&mut s.k, &s.xb, &w.wk[l]);
        matmul(&mut s.v, &s.xb, &w.wv[l]);

        // apply RoPE rotation to the q and k vectors for each head
        for h in 0..N_HEADS {
            // get the q and k vectors for this head
            let q = &mut s.q[h * HEAD_SIZE..];
            let k = &mut s.k[h * HEAD_SIZE..];
            // rotate q and k by the freq_cis_real and freq_cis_imag
            for i in (0..HEAD_SIZE).step_by(2) {
                let q0 = q[i];
                let q1 = q[i + 1];
                let k0 = k[i];
                let k1 = k[i + 1];
                let fcr = freq_cis_real_row[i / 2];
                let fci = freq_cis_imag_row[i / 2];
                q[i + 0] = q0 * fcr - q1 * fci;
                q[i + 1] = q0 * fci + q1 * fcr;
                k[i + 0] = k0 * fcr - k1 * fci;
                k[i + 1] = k0 * fci + k1 * fcr;
            }
        }

        // save key,value at this time step (pos) to our kv cache
        let key_cache_row = &mut s.key_cache[l][pos];
        let value_cache_row = &mut s.value_cache[l][pos];
        for h in 0..N_HEADS {
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
                    let k = &s.key_cache[l][t][h];
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
                    let v = &s.value_cache[l][t][h];
                    // get the attention weight for this timestep
                    let a = att[t];
                    // accumulate the weighted value into xb
                    for i in 0..HEAD_SIZE {
                        xb[i] += a * v[i];
                    }
                }
            });

        // final matmul to get the output of the attention
        matmul(&mut s.xb2, &s.xb, &w.wo[l]);

        // residual connection back into x
        accum(x, &s.xb2);

        // ffn rmsnorm
        rmsnorm(&mut s.xb, Some(x), &w.rms_ffn_weight[l]);

        // Now for FFN in PyTorch we have: self.w2(F.silu(self.w1(x)) * self.w3(x))
        // first calculate self.w1(x) and self.w3(x)
        matmul(&mut s.hb, &s.xb, &w.w1[l]);
        matmul(&mut s.hb2, &s.xb, &w.w3[l]);

        // F.silu; silu(x)=x*σ(x),where σ(x) is the logistic sigmoid
        for i in 0..HIDDEN_DIM {
            s.hb[i] *= 1.0 / (1.0 + (-s.hb[i]).exp());
        }

        // elementwise multiply with w3(x)
        for i in 0..HIDDEN_DIM {
            s.hb[i] *= s.hb2[i];
        }

        // final matmul to get the output of the ffn
        matmul(&mut s.xb, &s.hb, &w.w2[l]);

        // residual connection
        accum(x, &s.xb);
    }

    // final rmsnorm
    rmsnorm(x, None, &w.rms_final_weight);

    // classifier into logits
    matmul(&mut s.logits, &s.x, &last_layer);
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

    fn random_f32(self: &mut Self) -> f32 {
        // random float32 in [0,1)
        (self.random_u32() >> 8) as f32 / 16777216.0
    }

    fn sample(self: &mut Self, probabilities: &[f32], n: usize) -> usize {
        // sample index from probabilities, they must sum to 1
        let r = self.random_f32();
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
    println!("Configuration: {config:?}");

    let start = file.seek(SeekFrom::Current(0)).unwrap();
    let mmap = unsafe { MmapOptions::new().offset(start).map(&file).unwrap() };
    assert_eq!(mmap.len(), mem::size_of::<TransformerWeights>());
    let weights: Box<TransformerWeights> =
        unsafe { Box::from_raw(mmap.as_ptr() as *mut TransformerWeights) };

<<<<<<< HEAD
    let last_layer;
=======
    let mut last_layer;
>>>>>>> 6c881aa (typed)
    let mut wcls: Box<[[f32; DIM]; VOCAB_SIZE]> = vec![[0.0; DIM]; VOCAB_SIZE]
        .into_boxed_slice()
        .try_into()
        .unwrap();

    for i in 0..VOCAB_SIZE {
        for j in 0..DIM {
            if SHARED_SIZE == 0 {
                wcls[i][j] = weights.token_embedding_table[i][j];
            } else {
                wcls[i][j] = weights.wcls[i][j];
            }
        }
    }
    last_layer = &wcls;

    //let weights = TransformerWeights::init(&config, data);
    //println!("weights {:?} {:?}", weights.token_embedding_table[0], weights.w2[0]);

    // right now we cannot run for more than config.seq_len steps
    if steps <= 0 || steps > config.seq_len {
        steps = config.seq_len;
    }

    let tokenizer = {
        let mut file = File::open("tokenizer.bin").unwrap();
        Tokenizer::load(&mut file, &config)
    };

    // create and init the application RunState
    let mut state = RunState::new();

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
        transformer(token, pos, &mut state, &weights, &last_layer);
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

    // Don't free weights, they're mmapped.
    mem::forget(weights);
}
