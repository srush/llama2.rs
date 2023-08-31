// Constant sizes of the model are stored in a different file.
use crate::constants::{
    ATTN_GROUPS, DIM, HEAD_SIZE, HIDDEN_DIM, KV_DIM, N_HEADS, N_KV_HEADS, N_LAYERS, SEQ_LEN,
    VOCAB_SIZE,
};

#[cfg(gpu = "yes")]
mod x {
    use crate::gptq_cuda::QTransformerWeightsCuda;
    pub type TWeights = QTransformerWeightsCuda;
}

#[cfg(all(quant = "Q_4",gpu= "no"))]
mod x {
    use crate::gptq::QTransformerWeights;
    pub type TWeights = QTransformerWeights;
}

#[cfg(quant = "no")]
mod x {
    // Turn off GPT-Q Quantization.
    use crate::model::TransformerWeights;
    pub type TWeights = TransformerWeights;
}

use rayon::prelude::*;
pub use x::TWeights;

pub struct RunState {
    // kv cache
    key_cache: Vec<Vec<[[f32; HEAD_SIZE]; N_KV_HEADS]>>, // (layer, seq_len, dim)
    value_cache: Vec<Vec<[[f32; HEAD_SIZE]; N_KV_HEADS]>>, // (layer, seq_len, dim)
}

impl RunState {
    pub fn new() -> Box<Self> {
        Box::new(RunState {
            key_cache: vec![vec![[[0.0; HEAD_SIZE]; N_KV_HEADS]; SEQ_LEN]; N_LAYERS],
            value_cache: vec![vec![[[0.0; HEAD_SIZE]; N_KV_HEADS]; SEQ_LEN]; N_LAYERS],
        })
    }
}

// ----------------------------------------------------------------------------
// neural net blocks

fn accum(a: &mut [f32], b: &[f32]) {
    for (a_i, b_i) in a.iter_mut().zip(b) {
        *a_i += b_i;
    }
}

fn rmsnorm(o: &mut [f32; DIM], xo: &[f32; DIM], weight: &[f32; DIM], epsilon: f32) {
    // calculate sum of squares
    let mut ss = xo.iter().fold(0.0, |acc, x| acc + x * x);

    // take mean
    ss /= DIM as f32;
    ss += epsilon;
    ss = 1.0 / ss.sqrt();
    // normalize and scale
    for (j, weight_j) in weight.iter().enumerate() {
        // Solve some borrow nonsense.
        o[j] = weight_j * ss * xo[j];
    }
}

pub fn softmax(x: &mut [f32]) {
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

/// F.silu; silu(x)=x*σ(x),where σ(x) is the logistic sigmoid
fn silu(s: &mut [f32], s2: &[f32]) {
    for (s_i, s2_i) in s.iter_mut().zip(s2) {
        *s_i = *s_i * (1.0 / (1.0 + (-*s_i).exp()));
        // elementwise multiply with w3(x)
        *s_i = *s_i * s2_i;
    }
}

fn rope(queries: &mut [f32; DIM], keys: &mut [f32; KV_DIM], pos: usize) {
    for h in 0..N_HEADS {
        // get the q and k vectors for this head
        let q = &mut queries[h * HEAD_SIZE..][..HEAD_SIZE];

        // rotate q and k by the freq_cis_real and freq_cis_imag
        for i in 0..HEAD_SIZE {
            let freq = 1.0 / f32::powf(10000.0, ((2 * i % HEAD_SIZE) as f32) / (HEAD_SIZE as f32));
            let val = (pos as f32) * freq;
            let fcr = f32::cos(val);
            let fci = f32::sin(val);

            if i < HEAD_SIZE / 2 {
                q[i] = q[i] * fcr - q[(HEAD_SIZE / 2) + i] * fci;
            } else {
                q[i] = q[i] * fcr + q[i - (HEAD_SIZE / 2)] * fci;
            }

            if h < N_KV_HEADS {
                let k = &mut keys[h * HEAD_SIZE..][..HEAD_SIZE];
                if i < HEAD_SIZE / 2 {
                    k[i] = k[i] * fcr - k[(HEAD_SIZE / 2) + i] * fci;
                } else {
                    k[i] = k[i] * fcr + k[i - (HEAD_SIZE / 2)] * fci;
                }
            }
        }
    }
    // Apply RoPE rotation to the q and k vectors for each head
}

fn multihead_attention(
    out: &mut [[f32; DIM]],
    queries: &[[f32; DIM]],
    value_cache: &[[[f32; HEAD_SIZE]; N_KV_HEADS]],
    key_cache: &[[[f32; HEAD_SIZE]; N_KV_HEADS]],
    pos: &[usize],
) {
    for (i, pos) in pos.into_iter().enumerate() {
        let out = &mut out[i];
        let queries = &queries[i];

        // multihead attention. iterate over all heads
        // We do this a bit differently in rust.
        // Chunk things up so that each head is a separate slice.
        let mut xbs: Vec<&mut [f32]> = out.chunks_mut(HEAD_SIZE).collect();
        let qs: Vec<&[f32]> = queries.chunks_exact(HEAD_SIZE).collect();

        xbs.par_iter_mut().enumerate().for_each(|(h, xb)| {
            // get the query vector for this head
            let q = qs[h];
            let mut att = [0.0; SEQ_LEN];
            // attention scores for this head
            // iterate over all timesteps, including the current one
            for (t, att_t) in (0..=*pos).zip(att.iter_mut()) {
                // get the key vector for this head and at this timestep
                let k = &key_cache[t][h / ATTN_GROUPS];
                // calculate the attention score as the dot product of q and k
                *att_t = dot(q, k) / (HEAD_SIZE as f32).sqrt();
            }

            // softmax the scores to get attention weights, from 0..pos inclusively
            softmax(&mut att[..=*pos]);

            xb.fill(0.0);
            // weighted sum of the values, store back into xb
            for t in 0..=*pos {
                // get the value vector for this head and at this timestep
                let v = &value_cache[t][h / ATTN_GROUPS];
                // accumulate the weighted value into xb
                for (xb_i, v_i) in xb.iter_mut().zip(v) {
                    *xb_i += att[t] * v_i;
                }
            }
        });
    }
}

pub fn transformer<const B: usize>(
    logits: &mut [[f32; VOCAB_SIZE]; 1],
    tokens: &[usize; B],
    pos: &[usize; B],
    s: &mut RunState,
    w: &TWeights,
) {
    // Run state
    let mut x = [[0.0; DIM]; B];

    // copy the token embedding into x
    for (x_i, token) in x.iter_mut().zip(tokens) {
        x_i.copy_from_slice(&w.token_embedding_table[*token]);
    }

    // pluck out the "pos" row of freq_cis_real and freq_cis_imag
    // let freq_cis_real_row: Vec<[f32; DIM / N_HEADS / 2]> =
    //     pos.iter().map(|i| w.freq_cis_real[*i]).collect();
    // let freq_cis_imag_row: Vec<[f32; DIM / N_HEADS / 2]> =
    //     pos.iter().map(|i| w.freq_cis_imag[*i]).collect();

    // FFN
    let mut xb2 = [[0.0; DIM]; B];
    let mut hb = [[0.0; HIDDEN_DIM]; B];
    let mut hb2 = [[0.0; HIDDEN_DIM]; B];
    // Attention
    let mut q = [[0.0; DIM]; B];
    let mut k = [[0.0; KV_DIM]; B];
    let mut v = [[0.0; KV_DIM]; B];
    let mut xb = [[0.0; DIM]; B];
    // forward all the layers
    for l in 0..N_LAYERS {
        // qkv matvecs for this position

        {
            // if l == 0 {
            //     println!("emb {:?}", x[0]);
            // }
            // attention rmsnorm
            for (xb, x) in xb.iter_mut().zip(x.iter()) {
                rmsnorm(xb, x, &w.rms_att_weight[l], w.rms_eps);
            }
            // if l == 0 {
            //     println!("norm {:?}", xb[0]);
            // }

            w.wq[l].matvec(&mut q, &xb);
            w.wk[l].matvec(&mut k, &xb);
            w.wv[l].matvec(&mut v, &xb);

            for i in 0..B {
                rope(&mut q[i], &mut k[i], pos[i]);
            }
            // save key,value at this time step (pos) to our kv cache
            // if l == 31 {
            //     println!("key 31 {:?}", s.key_cache[l][0][0]);
            // }

            for (i, pos) in pos.into_iter().enumerate() {
                for (row, k) in s.key_cache[l][*pos]
                    .iter_mut()
                    .zip(k[i].chunks_exact(HEAD_SIZE))
                {
                    row.copy_from_slice(&k);
                }
                for (row, v) in s.value_cache[l][*pos]
                    .iter_mut()
                    .zip(v[i].chunks_exact(HEAD_SIZE))
                {
                    row.copy_from_slice(&v);
                }
            }
            multihead_attention(
                &mut xb,
                &q,
                &s.value_cache[l].as_slice(),
                &s.key_cache[l].as_slice(),
                pos,
            );
        }
        {
            // final matvec to get the output of the attention
            w.wo[l].matvec(&mut xb2, &xb);

            // residual connection back into x
            for i in 0..B {
                accum(&mut x[i], &xb2[i]);
                // ffn rmsnorm
                rmsnorm(&mut xb[i], &x[i], &w.rms_ffn_weight[l], w.rms_eps);
            }

            // Now for FFN in PyTorch we have: self.w2(F.silu(self.w1(x)) * self.w3(x))
            // first calculate self.w1(x) and self.w3(x)
            w.w1[l].matvec(&mut hb, &xb);
            w.w3[l].matvec(&mut hb2, &xb);

            for (hb, hb2) in hb.iter_mut().zip(hb2.iter()) {
                silu(hb, hb2);
            }

            // final matvec to get the output of the ffn
            w.w2[l].matvec(&mut xb, &hb);
        }

        // residual connection
        for (x, xb) in x.iter_mut().zip(xb.iter()) {
            accum(x, xb);
        }
    }
    let mut fin = [[0.0; DIM]; B];
    if B == 1 {
        // final rmsnorm
        for (x, fin) in x.iter().zip(fin.iter_mut()) {
            rmsnorm(fin, x, &w.rms_final_weight, w.rms_eps);
        }

        // classifier into logits
        w.wcls.matvec(logits, &fin[..1].try_into().expect("size"));
        //println!("logits {:?}", logits);
    };
}
