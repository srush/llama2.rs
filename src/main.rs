use std::str::from_utf8_unchecked;
// This is a conversion of llama2.c to rust. 
// It is basically line-by-line following chatgpt :)
use std::time::{SystemTime, UNIX_EPOCH};
use std::{env, io};
use std::fs::File;
use std::io::{Read, Write, Seek, SeekFrom};
use rayon::prelude::*;
use memmap2::MmapOptions;

// Some helpers for reading from binary files.
fn read_usize(file : &mut File) -> usize {
    let mut buf = [0u8; 4];
    file.read_exact(&mut buf).unwrap();
    i32::from_le_bytes(buf) as usize
}

fn read_float(file : &mut File) -> f32 {
    let mut buf = [0u8; 4];
    file.read_exact(&mut buf).unwrap();
    f32::from_le_bytes(buf)
}

fn read_str(file : &mut File, len: usize) -> String {
    let mut buf: Vec<u8> = vec![0u8; len];
    file.read_exact(&mut buf).unwrap();
    unsafe {from_utf8_unchecked(&buf).to_owned() }
}

fn str_lookup(str: &str, vocab: &[String]) -> Option<usize> {
    // find the first perfect match for str in vocab, return its index or -1 if not found
    vocab.into_iter().position(|x| x == str)
}


// C-like point arithmetic
struct Ptr<'a> {
    x: &'a[f32],
    total: usize
}

impl<'a> Ptr<'a> {
    fn align(self: &mut Self, size: usize) -> &'a[f32] {
        self.total = self.total + size;
        let ret = &self.x[..size];
        self.x = &self.x[size..];
        ret
    }
} 

fn argmax(v: &[f32]) -> usize {
    // return argmax of v in elements 0..n
    v.iter().enumerate().reduce(|a, b| 
        if a.1 > b.1 { a } else { b }).unwrap().0
}


#[derive(Debug)]
#[allow(dead_code)]
struct Config {
    dim:  usize, // transformer dimension
    hidden_dim: usize, // for ffn layers
    n_layers: usize, // number of layers
    n_heads: usize, // number of query heads
    n_kv_heads: usize, // number of key/value heads (can be < query heads because of multiquery)
    vocab_size: usize, // vocabulary size, usually 256 (byte-level)
    seq_len: usize, // max sequence length
}

impl Config {
    fn load(file :& mut File) -> Self {
        Config {
            dim : read_usize(file),
            hidden_dim: read_usize(file), 
            n_layers: read_usize(file), 
            n_heads: read_usize(file), 
            n_kv_heads: read_usize(file),
            vocab_size: read_usize(file),
            seq_len: read_usize(file),
        }
    }
}

struct RunState {
    // current wave of activations
    x:  Vec<f32>, // activation at current time stamp (dim,)
    xb: Vec<f32>, // same, but inside a residual branch (dim,)
    xb2: Vec<f32>, // an additional buffer just for convenience (dim,)
    hb: Vec<f32>, // buffer for hidden dimension in the ffn (hidden_dim,)
    hb2: Vec<f32>, // buffer for hidden dimension in the ffn (hidden_dim,)
    q: Vec<f32>, // query (dim,)
    k: Vec<f32>, // key (dim,)
    v: Vec<f32>, // value (dim,)
    att: Vec<Vec<f32>>, // buffer for scores/attention values (n_heads, seq_len)
    logits: Vec<f32>, // output logits
    // kv cache
    key_cache: Vec<f32>,   // (layer, seq_len, dim)
    value_cache: Vec<f32>, // (layer, seq_len, dim)
}

impl RunState {
    fn new(p: &Config) -> Self {
        RunState {
            x: vec![0.0; p.dim],
            xb: vec![0.0; p.dim],
            xb2: vec![0.0; p.dim],
            hb: vec![0.0; p.hidden_dim],
            hb2: vec![0.0; p.hidden_dim],
            q: vec![0.0; p.dim],
            k: vec![0.0; p.dim],
            v: vec![0.0; p.dim],
            att: vec![vec![0.0;  p.seq_len]; p.n_heads],
            logits: vec![0.0; p.vocab_size],
            key_cache: vec![0.0; p.n_layers * p.seq_len * p.dim],
            value_cache: vec![0.0; p.n_layers * p.seq_len * p.dim],
        }
    }
}

struct TransformerWeights<'a> {
    // token embedding table
    token_embedding_table: &'a[f32], // (vocab_size, dim)
    // weights for rmsnorms
    rms_att_weight: &'a[f32]    , // (layer, dim) rmsnorm weights
    rms_ffn_weight: &'a[f32], // (layer, dim)
    // weights for matmuls
    wq: &'a[f32], // (layer, dim, dim)
    wk: &'a[f32], // (layer, dim, dim)
    wv: &'a[f32], // (layer, dim, dim)
    wo: &'a[f32], // (layer, dim, dim)
    // weights for ffn
    w1: &'a[f32], // (layer, hidden_dim, dim)
    w2: &'a[f32], // (layer, dim, hidden_dim)
    w3: &'a[f32], // (layer, hidden_dim, dim)
    // final rmsnorm
    rms_final_weight: &'a[f32], // (dim,)
    // freq_cis for RoPE relatively positional embeddings
    freq_cis_real: &'a[f32], // (seq_len, dim/2)
    freq_cis_imag: &'a[f32], // (seq_len, dim/2)
    // (optional) classifier weights for the logits, on the last layer
    wcls: &'a[f32],
}

impl<'a> TransformerWeights<'a> {
    fn init(p: &Config, f: &'a[f32], shared_weights: bool) -> Self {
        let mut ptr = Ptr{x:f, total:0};
        let head_size = p.dim / p.n_heads;
        let mut ret = TransformerWeights {
            token_embedding_table: ptr.align(p.vocab_size * p.dim),
            rms_att_weight: ptr.align( p.n_layers * p.dim),
            wq: ptr.align( p.n_layers * p.dim * p.dim),
            wk: ptr.align( p.n_layers * p.dim * p.dim),
            wv: ptr.align( p.n_layers * p.dim * p.dim),
            wo: ptr.align( p.n_layers * p.dim * p.dim),
            rms_ffn_weight: ptr.align( p.n_layers * p.dim),
            w1: ptr.align( p.n_layers * p.hidden_dim * p.dim),
            w2: ptr.align( p.n_layers * p.dim * p.hidden_dim),
            w3: ptr.align( p.n_layers * p.hidden_dim * p.dim),
            rms_final_weight: ptr.align( p.dim),
            freq_cis_real: ptr.align( p.seq_len as usize * head_size / 2),
            freq_cis_imag: ptr.align( p.seq_len as usize * head_size / 2),
            wcls: &[] 
        };
        ret.wcls = if !shared_weights { 
            &ptr.align( p.dim * p.vocab_size as usize) 
        } else {
            ret.token_embedding_table
        };
        ret
    }
}


type Token = usize;

#[derive(Debug)]
struct Tokenizer {
    vocab: Vec<String>,
    vocab_scores: Vec<f32>,
    max_token_length: usize
}

impl Tokenizer {
    fn load(file: &mut File, config: &Config) -> Tokenizer {
        // read in the tokenizer.bin file
        let mut tokenizer = Tokenizer {
            vocab: Vec::with_capacity(config.vocab_size),
            vocab_scores: Vec::with_capacity(config.vocab_size),
            max_token_length: 0u32 as usize
        };
        tokenizer.max_token_length = read_usize(file);
        for _ in 0..config.vocab_size {
            tokenizer.vocab_scores.push(read_float(file));
            let len = read_usize(file);
            tokenizer.vocab.push(read_str(file, len));
        };
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
                None => { return tokens } // we couldn't find any more pairs to merge, so we're done
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

fn rmsnorm(o: &mut [f32], xo: Option<&[f32]>, weight: &[f32], size: usize) {
    // calculate sum of squares
    let mut ss: f32 = 0.0;
    for i in 0 .. size {
        let x = xo.unwrap_or(o);
        ss += x[i] * x[i];
    }
    ss /= o.len() as f32;
    ss += 1e-5;
    ss = 1.0 / ss.sqrt();
    // normalize and scale
    for j in 0..size {
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

fn matmul(xout: &mut [f32], x: &[f32], w: &[f32], n: usize, d: usize) {
    // W (d,n) @ x (n,) -> xout (d,)
    // by far the most amount of time is spent inside this little function
    assert_eq!(d, xout.len());
    xout.par_iter_mut().enumerate().for_each(|(i, v)| {
        let mut val = 0.0;
        for j in 0..n {
            val += w[i * n + j] * x[j];
        }
        *v = val;
    })
}

fn transformer(token: usize, pos: usize, p: &Config, s: &mut RunState, w: &TransformerWeights) {
    // a few convenience variables
    let x = &mut s.x;
    let dim = p.dim;
    let hidden_dim = p.hidden_dim;
    let head_size = dim / p.n_heads;

    // copy the token embedding into x
    let content_row = &w.token_embedding_table[(token * dim)..][..dim];
    x.copy_from_slice(content_row);

    // pluck out the "pos" row of freq_cis_real and freq_cis_imag
    let freq_cis_real_row = &w.freq_cis_real[(pos * head_size / 2)..][..head_size / 2];
    let freq_cis_imag_row = &w.freq_cis_imag[(pos * head_size / 2)..][..head_size / 2];

    // forward all the layers
    for l in 0..p.n_layers {
        // attention rmsnorm
        rmsnorm(&mut s.xb, Some(x), &w.rms_att_weight[l * dim..], dim);

        // qkv matmuls for this position
        matmul(&mut s.q, &s.xb, &w.wq[l * dim * dim..], dim, dim);
        matmul(&mut s.k, &s.xb, &w.wk[l * dim * dim..], dim, dim);
        matmul(&mut s.v, &s.xb, &w.wv[l * dim * dim..], dim, dim);

        // apply RoPE rotation to the q and k vectors for each head
        for h in 0..p.n_heads {
            // get the q and k vectors for this head
            let q = &mut s.q[h * head_size..];
            let k = &mut s.k[h * head_size..];
            // rotate q and k by the freq_cis_real and freq_cis_imag
            for i in (0..head_size).step_by(2) {
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
        let loff = l * p.seq_len * dim; // kv cache layer offset for convenience
        let key_cache_row = &mut s.key_cache[loff + pos * dim..][..dim];
        let value_cache_row = &mut s.value_cache[loff + pos * dim..][..dim];
        key_cache_row.copy_from_slice(&s.k);
        value_cache_row.copy_from_slice(&s.v);

        // multihead attention. iterate over all heads
        // This part requires ensuring that there is safety in parallel mutation.
        let xbs: Vec<&mut [f32]> = s.xb.chunks_mut(head_size).collect();
        s.att.par_iter_mut().zip(xbs).enumerate().for_each(|(h, (att, xb))| {
            // get the query vector for this head
            let q = &s.q[h * head_size..];
            // attention scores for this head
            //let mut att = &mut s.att[h * p.seq_len..];
            // iterate over all timesteps, including the current one
            for t in 0..=pos {
                // get the key vector for this head and at this timestep
                let k = &s.key_cache[loff + t * dim + h * head_size..];
                // calculate the attention score as the dot product of q and k
                let score = q.iter().zip(k.iter()).map(|(&q_i, &k_i)| q_i * k_i).sum::<f32>() / (head_size as f32).sqrt();
                // save the score to the attention buffer
                att[t] = score;
            }

            // softmax the scores to get attention weights, from 0..pos inclusively
            softmax(&mut att[..=pos]);

            // weighted sum of the values, store back into xb
            //let xb = &mut s.xb[h * head_size..];
            for i in 0..head_size {
                xb[i] = 0.0;
            }
            for t in 0..=pos {
                // get the value vector for this head and at this timestep
                let v = &s.value_cache[loff + t * dim + h * head_size..];
                // get the attention weight for this timestep
                let a = att[t];
                // accumulate the weighted value into xb
                for i in 0..head_size {
                    xb[i] += a * v[i];
                }
            }
        });

        // final matmul to get the output of the attention
        matmul(&mut s.xb2, &s.xb, &w.wo[l * dim * dim..(l + 1) * dim * dim], dim, dim);

        // residual connection back into x
        accum(x, &s.xb2);

        // ffn rmsnorm
        rmsnorm(&mut s.xb, Some(x), &w.rms_ffn_weight[l * dim..], dim);

        // Now for FFN in PyTorch we have: self.w2(F.silu(self.w1(x)) * self.w3(x))
        // first calculate self.w1(x) and self.w3(x)
        matmul(&mut s.hb, &s.xb, &w.w1[l * dim * hidden_dim..], dim, hidden_dim);
        matmul(&mut s.hb2, &s.xb, &w.w3[l * dim * hidden_dim..], dim, hidden_dim);

        // F.silu; silu(x)=x*σ(x),where σ(x) is the logistic sigmoid
        for i in 0..hidden_dim {
            s.hb[i] *= 1.0 / (1.0 + (-s.hb[i]).exp());
        }

        // elementwise multiply with w3(x)
        for i in 0..hidden_dim {
            s.hb[i] *= s.hb2[i];
        }

        // final matmul to get the output of the ffn
        matmul(&mut s.xb, &s.hb, &w.w2[l * dim * hidden_dim..], hidden_dim, dim);

        // residual connection
        accum(x, &s.xb);
    }

    // final rmsnorm
    rmsnorm(x, None, &w.rms_final_weight, dim);

    // classifier into logits
    matmul(&mut s.logits, &s.x, &w.wcls, dim, p.vocab_size);
}






fn time_in_ms() -> i64 {
    // return time in milliseconds, for benchmarking the model speed
    let time = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap();
    time.as_secs() as i64 * 1000 + time.subsec_millis() as i64
}


struct Random {
    seed: u64
}
impl Random {
    fn new() -> Random {
        // seed rng with time. if you want deterministic behavior use temperature 0.0
        Random {
            seed : SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_secs() as u64
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
    let temperature = 
        if args.len() >=2 { args[1].parse().expect("temperature must be float") }
        else {0.9};

    let mut steps = 
        if args.len() >=3 { args[2].parse().expect("steps must be int") }
        else {256};
    let prompt = args.get(3).unwrap_or(&String::from("")).to_owned();


    let mut random = Random::new();
    // read in the model.bin file
    let mut file = File::open(&checkpoint).unwrap();

    // Config
    let config = Config::load(&mut file);
    // println!("Configuration: {config:?}");

    let start = file.seek(SeekFrom::Current(0)).unwrap();
    //let start = 0;
    let mmap = unsafe { MmapOptions::new().offset(start).map(&file).unwrap() };
    let data = unsafe { std::slice::from_raw_parts(mmap.as_ptr() as *const f32, 
                                                    mmap.len() / std::mem::size_of::<f32>())};
   
    let shared_weights = true;
    let weights = TransformerWeights::init(&config, data, shared_weights);
    //println!("weights {:?} {:?}", weights.token_embedding_table[0], weights.w2[0]);

    // right now we cannot run for more than config.seq_len steps
    if steps <= 0 || steps > config.seq_len {
        steps = config.seq_len;
    }
    
    let tokenizer = {
        let mut file = File::open("tokenizer.bin").unwrap();
        Tokenizer::load(&mut file, &config)
    };
    // println!("Tokenizer: {tokenizer:?}");

    // create and init the application RunState
    let mut state = RunState::new(&config);

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
        transformer(token, pos, &config, &mut state, &weights);
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
                softmax(&mut state.logits[..config.vocab_size]);
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
    println!("\nachieved tok/s: {}", (steps - 1) as f32 / (end - start) as f32 * 1000.0);
}