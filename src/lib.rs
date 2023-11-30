#![feature(portable_simd)]

use std::{fs::File, mem};

use constants::Config;
use memmap2::{Mmap, MmapOptions};

#[cfg(feature = "python")]
use pyo3::prelude::*;

pub mod constants;
#[cfg(quant = "Q_4")]
pub mod gptq;
#[cfg(gpu = "yes")]
pub mod gptq_cuda;
pub mod inference;
pub mod model;
pub mod models;
pub mod tokenizer;
pub mod util;

#[cfg(gpu = "yes")]
mod llama_model {
    use crate::gptq::QTransformerWeights;
    use crate::gptq_cuda::{convert, QTransformerWeightsCuda};
    use cust::prelude::*;
    use std::io::Seek;

    #[allow(dead_code)]
    #[cfg_attr(feature = "python", pyclass)]
    pub struct LlamaModel {
        pub config: super::Config,
        pub weights: QTransformerWeightsCuda,
        _ctx: Context,
    }
    impl LlamaModel {
        pub fn weights(&self) -> &QTransformerWeightsCuda {
            &self.weights
        }

        pub fn prefill(&self) -> bool {
            false
        }

        pub fn from_file(checkpoint: &str, debug: bool) -> LlamaModel {
            let _ctx = cust::quick_init().expect("start");
            let mut file = super::File::open(checkpoint).unwrap();
            let config = super::Config::load(&mut file);
            if debug {
                println!("Configuration: {config:?}");
            }
            let start = file.stream_position().unwrap();
            let mmap: super::Mmap =
                unsafe { super::MmapOptions::new().offset(start).map(&file).unwrap() };
            assert_eq!(mmap.len(), super::mem::size_of::<QTransformerWeights>());
            let res = unsafe { &*(mmap.as_ptr() as *const QTransformerWeights) };

            LlamaModel {
                config,
                weights: convert(res).expect("conversion"),
                _ctx: _ctx,
            }
        }
    }
}

#[cfg(gpu = "no")]
mod llama_model {
    use crate::models::TWeights;
    use pyo3::pyclass;
    #[allow(dead_code)]
    #[cfg_attr(feature = "python", pyclass)]
    pub struct LlamaModel {
        mmap: super::Mmap,
        pub config: super::Config,
        pub weights: &'static TWeights,
    }
    use std::io::Seek;
    impl LlamaModel {
        pub fn weights(&self) -> &TWeights {
            self.weights
        }

        pub fn prefill(&self) -> bool {
            true
        }

        pub fn from_file(checkpoint: &str, debug: bool) -> LlamaModel {
            let mut file = super::File::open(checkpoint).unwrap();
            let config = super::Config::load(&mut file);
            if debug {
                println!("Configuration: {config:?}");
                println!("No CUDA");
            }
            let start = file.stream_position().unwrap();
            let mmap = unsafe { super::MmapOptions::new().offset(start).map(&file).unwrap() };
            assert_eq!(mmap.len(), super::mem::size_of::<TWeights>());
            let weights = unsafe { &*(mmap.as_ptr() as *const TWeights) };
            LlamaModel {
                mmap: mmap,
                config: config,
                weights: weights,
            }
        }
    }
}

pub use llama_model::LlamaModel;

// workaround needed because of https://github.com/PyO3/pyo3/issues/780
#[cfg(feature = "python")]
#[pymethods]
impl LlamaModel {
    // NOTE: this pyo3 signature only applies to Python code (i.e. in Rust code you must specify debug).
    #[new]
    #[pyo3(signature = (checkpoint, debug=false))]
    pub fn from_file_py(checkpoint: &str, debug: bool) -> LlamaModel {
        LlamaModel::from_file(checkpoint, debug)
    }
}

#[cfg(feature = "python")]
#[pymodule]
fn llama2_rs_pylib(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<tokenizer::Tokenizer>()?;
    m.add_class::<util::Random>()?;
    m.add_class::<constants::Config>()?;
    m.add_class::<LlamaModel>()?;
    m.add_wrapped(wrap_pyfunction!(inference::generate))?;
    Ok(())
}
