use crate::constants::{DIM, HIDDEN_DIM, KV_DIM, N_HEADS, N_LAYERS, SEQ_LEN, VOCAB_SIZE};
#[cfg(feature = "python")]
use pyo3::pyclass;
use rayon::prelude::*;

/// A dense linear layer.
///
/// Rust Notes:
/// 1) We use IN and OUT as generic constrants for safety
/// 2) We need repr(c) becuase we are memory mapping from a C file.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct Linear<const IN: usize, const OUT: usize> {
    w: [[f32; IN]; OUT], // Storage is as a dense matrix.
}

impl<const IN: usize, const OUT: usize> Linear<IN, OUT> {
    /// W (d,n) @ x (n,) -> xout (d,)
    // by far the most amount of time is spent inside this little function
    /// Rust note: par_iter_mut is from the RAYON library. It run in parallel.
    /// Rust note: x is passed by reference, xout as mutiple reference.    
    pub fn matvec<const B: usize>(self: &Self, xout: &mut [[f32; OUT]; B], x: &[[f32; IN]; B]) {
        for (xout, x) in xout.iter_mut().zip(x) {
            xout.par_iter_mut().enumerate().for_each(|(i, v)| {
                *v = self.w[i]
                    .iter()
                    .zip(x.iter())
                    .fold(0.0, |acc, (&_w, &_x)| acc + _w * _x);
            });
        }
    }
}

/// This is the main standard Transformer model
/// This is generally slower, but included for sanity and debugging.
#[repr(C)]
#[allow(dead_code)]
pub struct TransformerWeights {
    pub rms_eps: f32,

    // token embedding table
    pub token_embedding_table: [[f32; DIM]; VOCAB_SIZE],

    // weights for rmsnorms
    pub rms_att_weight: [[f32; DIM]; N_LAYERS],

    // weights for matmuls
    pub wq: [Linear<{ DIM }, { DIM }>; N_LAYERS],
    pub wk: [Linear<{ DIM }, { KV_DIM }>; N_LAYERS],
    pub wv: [Linear<{ DIM }, { KV_DIM }>; N_LAYERS],
    pub wo: [Linear<{ DIM }, { DIM }>; N_LAYERS],

    pub rms_ffn_weight: [[f32; DIM]; N_LAYERS],
    // weights for ffn
    pub w1: [Linear<{ DIM }, { HIDDEN_DIM }>; N_LAYERS],
    pub w2: [Linear<{ HIDDEN_DIM }, { DIM }>; N_LAYERS],
    pub w3: [Linear<{ DIM }, { HIDDEN_DIM }>; N_LAYERS],
    // final rmsnorm
    pub rms_final_weight: [f32; DIM], // (dim,)

    // Deprecated. freq_cis for RoPE relatively positional embeddings
    pub _freq_cis_real: [[f32; DIM / N_HEADS / 2]; SEQ_LEN],
    pub _freq_cis_imag: [[f32; DIM / N_HEADS / 2]; SEQ_LEN],

    // Classifier weights for the logits, on the last layer
    pub wcls: Linear<{ DIM }, { VOCAB_SIZE }>, // (dim,)
}
