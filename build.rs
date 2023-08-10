fn main() {
    // note: Cargo features are just binary flags, so this decides a more clear cfg key/value pair
    if cfg!(feature="7B") {
        println!("cargo:rustc-cfg=model_size=\"7B\"");
    } else if cfg!(feature="13B") {
        println!("cargo:rustc-cfg=model_size=\"13B\"");
    } else  if cfg!(feature="70B") {
        println!("cargo:rustc-cfg=model_size=\"70B\"");
    } else {
        panic!("Must set one of 7B, 13B or 70B features to pick a model size");
    }

    // set quantization level
    if cfg!(feature="quantized") {
        println!("cargo:rustc-cfg=quant=\"Q_4_128\"");
    }
    else {
        println!("cargo:rustc-cfg=quant=\"no\"");
    }

    // set target-feature based on AVX512/AVX2 desired support
    if cfg!(feature="avx512") {
        if !std::is_x86_feature_detected!("avx512f") {
            panic!("AVX512 is not supported on this machine");
        }
        println!("cargo:rustc-env=RUSTFLAGS=-C target-feature=+avx,+avx2,+fma,+sse3,+avx512f target-cpu=native");
    }
    else if std::is_x86_feature_detected!("avx2") {
        println!("cargo:rustc-env=RUSTFLAGS=-C target-feature=+avx,+avx2,+fma,+sse3 target-cpu=native");
    }
}
