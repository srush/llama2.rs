#![feature(portable_simd)]

use std::{fs::File, io::{SeekFrom, Seek}, mem};

use constants::Config;
use memmap2::{Mmap, MmapOptions};
use models::TWeights;
use pyo3::prelude::*;

pub mod constants;
pub mod gptq;
pub mod models;
pub mod tokenizer;
pub mod util;
pub mod inference;

#[allow(dead_code)]
#[pyclass]
pub struct LlamaModel {
    mmap: Mmap,
    #[pyo3(get)]
    pub config: Config,
    pub weights: &'static TWeights,
}

#[pymethods]
impl LlamaModel {
    // NOTE: this pyo3 signature only applies to Python code (i.e. in Rust code you must specify debug).
    #[new]
    #[pyo3(signature = (checkpoint, debug=false))]
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
        LlamaModel { mmap, config, weights }
    }
}

#[pymodule]
fn llama2_rs_pylib (_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<tokenizer::Tokenizer>()?;
    m.add_class::<util::Random>()?;
    m.add_class::<constants::Config>()?;
    m.add_class::<LlamaModel>()?;
    m.add_wrapped(wrap_pyfunction!(inference::generate))?;
    Ok(())
}