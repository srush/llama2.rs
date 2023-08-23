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

#[allow(dead_code)]
#[pyclass]
pub struct LoadedModel {
    mmap: Mmap,
    #[pyo3(get)]
    pub config: Config,
    pub weights: &'static TWeights,
}

#[pymethods]
impl LoadedModel {
    #[new]
    pub fn from_file(checkpoint: &str, debug: bool) -> LoadedModel {
        let mut file = File::open(checkpoint).unwrap();
        let config = Config::load(&mut file);
        if debug {
            println!("Configuration: {config:?}");
        }
        let start = file.seek(SeekFrom::Current(0)).unwrap();
        let mmap = unsafe { MmapOptions::new().offset(start).map(&file).unwrap() };
        assert_eq!(mmap.len(), mem::size_of::<TWeights>());
        let weights = unsafe { &*(mmap.as_ptr() as *const TWeights) };
        LoadedModel { mmap, config, weights }
    }
}

#[pymodule]
fn llama2_rs_pylib (_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<tokenizer::Tokenizer>()?;
    m.add_class::<util::Random>()?;
    m.add_class::<constants::Config>()?;
    m.add_class::<LoadedModel>()?;
    Ok(())
}