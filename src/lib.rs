#![feature(portable_simd)]

use std::{fs::File, io::Seek, mem};

use constants::Config;
use memmap2::{Mmap, MmapOptions};
use models::{QTransformerWeights, QTransformerWeights2, TWeights, convert};

#[cfg(feature = "python")]
use pyo3::prelude::*;

pub mod constants;
pub mod gptq;
pub mod models;
pub mod tokenizer;
pub mod util;
pub mod inference;
pub mod gptq_cuda;

#[cfg(gpu="yes")]
mod Model {
    use crate::models::QTransformerWeights2;
    use std::{fs::File, io::Seek, mem};

    #[allow(dead_code)]
    #[cfg_attr(feature = "python", pyclass)]
    pub struct LlamaModel {
        pub config: super::Config,
        pub weights: QTransformerWeights2
    }
    impl LlamaModel {
        pub fn weights(&self) -> &QTransformerWeights2 {
            &self.weights
        }

        pub fn prefill(&self) -> bool {
            false
        }

        pub fn from_file(checkpoint: &str, debug: bool) -> LlamaModel {
            let mut file = super::File::open(checkpoint).unwrap();
            let config = super::Config::load(&mut file);
            if debug {
                println!("Configuration: {config:?}");
            }
            let start = file.stream_position().unwrap();
            let mmap = unsafe { super::MmapOptions::new().offset(start).map(&file).unwrap() };
            assert_eq!(mmap.len(), super::mem::size_of::<super::QTransformerWeights>());
            let res = unsafe { &*(mmap.as_ptr() as *const super::QTransformerWeights) };
         
            LlamaModel { config, 
                weights: super::convert(res).expect("conversion")}
        }
    }
}

#[cfg(gpu="no")]
mod Model {
    #[allow(dead_code)]
    #[cfg_attr(feature = "python", pyclass)]
    pub struct LlamaModel {
        mmap: Mmap,
        pub config: Config,
        pub weights: &'static TWeights,
    }
    impl LlamaModel {
        pub fn weights(&self) -> &TWeights {
            self.weights
        }

        pub fn prefill(&self) -> bool {
            true
        }

        pub fn from_file(checkpoint: &str, debug: bool) -> LlamaModel {
            let mut file = File::open(checkpoint).unwrap();
            let config = Config::load(&mut file);
            if debug {
                println!("Configuration: {config:?}");
            }
            let start = file.stream_position().unwrap();
            let mmap = unsafe { MmapOptions::new().offset(start).map(&file).unwrap() };
            assert_eq!(mmap.len(), mem::size_of::<TWeights>());
            let weights = unsafe { &*(mmap.as_ptr() as *const TWeights) };
            weights
        }

    }
}

pub use Model::LlamaModel;

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
fn llama2_rs_pylib (_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<tokenizer::Tokenizer>()?;
    m.add_class::<util::Random>()?;
    m.add_class::<constants::Config>()?;
    m.add_class::<LlamaModel>()?;
    m.add_wrapped(wrap_pyfunction!(inference::generate))?;
    Ok(())
}