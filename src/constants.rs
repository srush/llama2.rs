use crate::util::read_usize;
use std::fs::File;

// currently only supports 4 bit quantization
pub const BITS: usize = 4;

#[cfg(model_size = "70B")]
mod constants {
    /// Llama 70B configuration
    pub const DIM: usize = 8192; // Base dimension
    pub const HIDDEN_DIM: usize = 28672; // Hidden dimension
    pub const ATTN_GROUPS: usize = 8; // Grouped query attention
    pub const N_LAYERS: usize = 80; // Number of layers
    pub const N_HEADS: usize = 64; // Number of heads
    pub const SEQ_LEN: usize = 2048; // Max sequence length
    pub const VOCAB_SIZE: usize = 32000; // Vocab size
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
pub mod constants {
    pub const DIM: usize = 4096;
    pub const HIDDEN_DIM: usize = 11008;
    pub const N_LAYERS: usize = 32;
    pub const ATTN_GROUPS: usize = 1;
    pub const N_HEADS: usize = 32;
    pub const SEQ_LEN: usize = 2048;
    pub const VOCAB_SIZE: usize = 32000;
}

#[cfg(group_size = "128")]
pub mod group {
    pub const GROUPSIZE: usize = 128;
}

#[cfg(group_size = "64")]
pub mod group {
    pub const GROUPSIZE: usize = 64;
}

#[cfg(group_size = "32")]
pub mod group {
    pub const GROUPSIZE: usize = 32;
}

pub use constants::*;
pub use group::*;

// Head size is constant but standardized.
pub const HEAD_SIZE: usize = DIM / N_HEADS;

// KV may be smaller due to Grouped Query Attention in 30B and 70B Llama2 models.
pub const KV_DIM: usize = DIM / ATTN_GROUPS;
pub const N_KV_HEADS: usize = N_HEADS / ATTN_GROUPS;

// This is kept in for debugging.
#[derive(Debug)]
#[allow(dead_code)]
pub struct Config {
    dim: usize,        // transformer dimension
    hidden_dim: usize, // for ffn layers
    n_layers: usize,   // number of layers
    n_heads: usize,    // number of query heads
    n_kv_heads: usize, // number of key/value heads (can be < query heads because of multiquery)
    vocab_size: usize,
    seq_len: usize,
    shared_weight: bool,
}

// This config is mostly ignored.
// We use it to check compile time constants.
impl Config {
    fn check_static(self: &Self) {
        assert_eq!(self.dim as usize, DIM);
        assert_eq!(self.hidden_dim as usize, HIDDEN_DIM);
        assert_eq!(self.n_layers as usize, N_LAYERS);
        assert_eq!(self.n_heads as usize, N_HEADS);
        assert_eq!(self.vocab_size as usize, VOCAB_SIZE);
    }

    pub fn load(file: &mut File) -> Self {
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
