#![feature(portable_simd)]
// Quant 4 bits
use cust::prelude::*;
use std::error::Error;

/// How many numbers to generate and add together.


static PTX: &str = include_str!("gptq.ptx");


fn main() {
    run().expect("doesn't work");
}
const IN :usize = 4096;
const OUT:usize = 4096;
const BITS:usize = 4;
    
const GROUPSIZE: usize = 128;
const GROUPS: usize = IN / GROUPSIZE;
const ING: usize = IN / 8;
const OUTG: usize = OUT / 8;

use rayon::prelude::*;
/// Code for quantized SIMD implementation.
use std::simd::{f32x8, i32x8, SimdFloat, SimdInt};

const B: usize = 1;
fn matvec() {
    let ten = f32x8::splat(10.0);
    let qweight: [[i32; ING]; OUT] = [[1985229328; ING] ; OUT];     
    let scales: [[f32; GROUPS]; OUT] = [[10.0; GROUPS]; OUT];
    let qzeros:  [[i32; GROUPS]; OUTG] = [[1985229328; GROUPS]; OUTG];
    
    let x_temp: [[f32x8; B]; IN/8] = [[ten; B];IN/8];
    let mut xout_temp: [[f32; B]; OUT] = [[0.0; B];OUT];
    let n_elements = OUT * IN;

    assert_eq!(ING, IN / 32 * BITS);
    assert_eq!(OUTG, OUT / 32 * BITS);


    // Constants for 4 bit.
            let z = f32x8::splat(0.0);
        let mask = (1 << BITS) - 1;
        let elems_per_i32 = 32 / BITS;
        let ipg: usize = GROUPSIZE / 32 * BITS;
        let mask_4bits = i32x8::splat(mask);
        let shift_right = i32x8::from_array([0, 4, 8, 12, 16, 20, 24, 28]);

        // Enumerate over each output position.
        xout_temp.par_iter_mut().enumerate().for_each(|(oi, o)| {
            let qzeros = &qzeros[oi / elems_per_i32];
            let out_elem = oi % elems_per_i32;
            let qweight = qweight[oi].chunks_exact(ipg);
            let mut sum = [z; B];

            // Everything in the batch starts at zero.
            o.fill(0.0);

            // Iterate over weight groups (32/64/128).
            for (((scale, qweight), x_temp), qzs) in scales[oi]
                .iter()
                .zip(qweight)
                .zip(x_temp.chunks_exact(GROUPSIZE / 8))
                .zip(qzeros)
            {
                let qz = ((qzs >> (BITS * out_elem)) & mask) + 1;
                let scale_simd = f32x8::splat(*scale);
                let zero_simd = i32x8::splat(qz);
                // Iterate over chunks of 8 weights. 
                for (&v, x) in qweight.iter().zip(x_temp) {
                    //Extract v into 8 chunks
                    let num_simd = i32x8::splat(v);
                    let qw: i32x8 = (num_simd >> shift_right) & mask_4bits;
                    let combine: f32x8 = (qw - zero_simd).cast::<f32>();
                    let weight: f32x8 = scale_simd * combine;
                    
                    // For each in the batch mult weight by input.
                    for (&x, s) in x.iter().zip(sum.iter_mut()) {
                        *s += weight * x;
                    }
                }
            }
            // Save the sum for last since reduce_sum is expensive.
            for (xout, s) in o.iter_mut().zip(sum) {
                *xout += s.reduce_sum();
            }
        });
        // Untranspose the output.
        println!("{:?}", &xout_temp[..10]);
    // for i in 0..xout.len() {
    //         for j in 0..OUT {
    //             xout[i][j] = xout_temp[j][i];
    //         }
    //     }
    
}




fn run()  -> Result<(), (Box<dyn Error>)> {
    matvec();
    let qweight: [[i32; ING]; OUT] = [[1985229328; ING] ; OUT];     
    let scales: [[f32; GROUPS]; OUT] = [[10.0; GROUPS]; OUT];
    let qzeros:  [[i32; GROUPS]; OUTG] = [[1985229328; GROUPS]; OUTG];
    let x_temp: [f32; IN] = [10.0; IN];
    let xout_temp: [f32; 1000000] = [0.0; 1000000];
    let mut xout: [[f32; OUT]; B] = [[0.0; OUT];B];
    let n_elements = OUT * IN;
    // grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),);
    // gptq_kernel[grid](q_weights, q_scale, q_zeros, x, q_out, BLOCK_SIZE=128);
    // q_out
            println!("init");
            // initialize CUDA, this will pick the first available device and will
            // make a CUDA context from it.
            // We don't need the context for anything but it must be kept alive.
            let _ctx = cust::quick_init()?;
            
           
            // Make the CUDA module, modules just house the GPU code for the kernels we created.
            // they can be made from PTX code, cubins, or fatbins.
            let module = Module::from_ptx(PTX, &[])?;
            
            println!("start");
            // make a CUDA stream to issue calls to. You can think of this as an OS thread but for dispatching
            // GPU calls.
            let stream = Stream::new(StreamFlags::NON_BLOCKING, None)?;
            
            // allocate the GPU memory needed to house our numbers and copy them over.
            // let mut qweight2 = vec![0; ING * OUT];
            // let mut i = 0;
            // for q in qweight {
            //     for v in q {
            //         qweight2[i] = v;
            //         i += 1;
            //     }
            // }
            let qweight_gpu = qweight.as_slice().as_dbuf()?;

            // let mut scale = vec![0.0; GROUPS * OUT];
            // let mut i = 0;
            // for q in scales {
            //     for v in q {
            //         scale[i] = v;
            //         i += 1;
            //     }
            // }
            // let mut zeros = vec![0; GROUPS * OUTG];
            // let mut i = 0;
            // for q in qzeros {
            //     for v in q {
            //         zeros[i] = v;
            //         i += 1;
            //     }
            // }

            // let qweight_gpu = self.qweight.as_dbuf()?;
            let qscale_gpu = scales.as_slice().as_dbuf()?;
            let qzeros_gpu = qzeros.as_slice().as_dbuf()?;
            let xtemp_gpu = x_temp.as_slice().as_dbuf()?;
            
            // allocate our output buffer. You could also use DeviceBuffer::uninitialized() to avoid the
            // cost of the copy, but you need to be careful not to read from the buffer.
            let out_buf = xout_temp.as_dbuf()?;
            
           
            // retrieve the add kernel from the module so we can calculate the right launch config.
            let func = module.get_function("gptq_kernel_0d1d2d3d4d5d").expect("couldn't find kernel");

           
            // use the CUDA occupancy API to find an optimal launch configuration for the grid and block size.
            // This will try to maximize how much of the GPU is used by finding the best launch configuration for the
            // current CUDA device/architecture.
            // let (_, block_size) = func.suggested_launch_configuration(0, 0.into())?;
            // let block_size: u32 = 1024 as u32;
            // let (_, block_size) = func.suggested_launch_configuration(0, 0.into())?;
    let block_size = 128 as u32;
    let eblock_size = 1024 as u32;

    let grid_size = ( ((IN * OUT) as u32) + eblock_size - 1) / eblock_size;


            // println!(
            //      "using {} blocks and {} threads per block",
            //      grid_size, block_size
            //  );
            
            // // Actually launch the GPU kernel. This will queue up the launch on the stream, it will
            // // not block the thread until the kernel is finished.
            // println!("Call");
            unsafe {
                launch!(
                    // slices are passed as two parameters, the pointer and the length.
                    func<<<grid_size, block_size, 512, stream>>>(
                        qweight_gpu.as_device_ptr(),
                        qscale_gpu.as_device_ptr(),
                        qzeros_gpu.as_device_ptr(),
                        xtemp_gpu.as_device_ptr(),
                        out_buf.as_device_ptr(),
                        IN as i32,
                        block_size
                    )
                )?;
            }
            println!("sync");
            stream.synchronize()?;
    let xout_temp = out_buf.as_slice().as_host_vec()?;
    for i in 0..xout.len() {
        for j in 0..OUT {
            xout[i][j] = 0.0;
            for k in 0..IN / eblock_size as usize {
                xout[i][j] += xout_temp[j * (IN / eblock_size as usize) + k];
            }
        }
    }
    
    println!("{:?}", &xout);

    Ok(())
}

