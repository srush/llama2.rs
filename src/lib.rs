#![feature(portable_simd)]

use pyo3::prelude::*;

pub mod constants;
pub mod gptq;
pub mod models;
pub mod tokenizer;
pub mod util;



#[pymodule]
fn llama2_rs_pylib (_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<tokenizer::Tokenizer>()?;
    Ok(())
}