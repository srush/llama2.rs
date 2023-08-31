use std::env;

fn main() {
    // note: Cargo features are just binary flags, so this decides a more clear cfg key/value pair
    let model_size = if cfg!(feature="7B") {
        "7B"
    } else if cfg!(feature="13B") {
        "13B"
    } else  if cfg!(feature="70B") {
        "70B"
    } else {
        panic!("Must set one of 7B, 13B or 70B features to pick a model size");
    };
    println!("cargo:rustc-cfg=model_size=\"{}\"", model_size);

    // get group size
    let group_size = if cfg!(feature="group_32") {
        32
    } else if cfg!(feature="group_64") {
        64
    } else if cfg!(feature="group_128") {
        128
    } else {
        // ok if there's no quantization
        0
    };

    // set quantization level
    if cfg!(feature="quantized") {
        if group_size == 0 {
            panic!("Must set one of group_32, group_64 or group_128 features to use quantization");
        }
        println!("cargo:rustc-cfg=group_size=\"{}\"", group_size);
        println!("cargo:rustc-cfg=quant=\"Q_4\"");
    }
    else {
        println!("cargo:rustc-cfg=quant=\"no\"");
    }

    if cfg!(feature="gpu") {
        println!("cargo:rustc-cfg=gpu=\"yes\"");
    } else {
        println!("cargo:rustc-cfg=gpu=\"no\"");
    }

    let mut rustflags = [
        "-C target-cpu=native",
        "-C link-args=-Wl,-zstack-size=419430400"
    ].map(String::from).to_vec();

    match env::var("CARGO_CFG_TARGET_ARCH").unwrap().as_str() {
        "x86_64" => {
            rustflags.push("-C target-feature=+avx,+avx2,+fma,+sse3".to_string());
        },
        // on Mac M* devices, SIMD support is via NEON, not AVX (and is enabled via target-cpu=native by default)
        // leave this here since we'll need explicit Accelerate linking for BLAS
        "aarch64" | _ => {},
    }
    println!("cargo:rustc-env=RUSTFLAGS={}", rustflags.join(" "));
}
