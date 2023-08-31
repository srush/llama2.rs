#![feature(portable_simd)]
// Quant 4 bits
use cust::prelude::*;
use std::error::Error;

/// How many numbers to generate and add together.


static PTX: &str = include_str!("gptq.ptx");
use rayon::prelude::*;
/// Code for quantized SIMD implementation.
use std::simd::{f32x8, i32x8, SimdFloat, SimdInt};
use rand::Rng;

fn main() {
    run().expect("doesn't work");
}
const IN :usize = 4096;
const OUT:usize = 2048;
const BITS:usize = 4;
    
const GROUPSIZE: usize = 128;
const GROUPS: usize = IN / GROUPSIZE;
const ING: usize = IN / 8;
const OUTG: usize = OUT / 8;

const B: usize = 1;

struct Data {
    qweight: [[i32; ING]; OUT],
    scales: [[f32; GROUPS]; OUT],
    qzeros:  [[i32; GROUPS]; OUTG],
    x_temp: [f32; IN]
}
fn basic() -> Data {
//     q_weights = torch.tensor([[1985229328] * (IN // 8)] * OUT).int().cuda()
// q_scale = torch.tensor([[10.] * (IN // GROUPSIZE)] * OUT).float().cuda()
// q_zeros = torch.tensor([[1985229328] * (IN // GROUPSIZE)] * (OUT // 8)).int().cuda()
// x = torch.tensor([[10.] * IN]).cuda()
    let mut d = Data {
        qweight: [[1985229328; ING]; OUT],
        scales: [[10.0; GROUPS]; OUT],
        //qzeros:  [[1985229328; GROUPS]; OUTG],
        qzeros:  [[0; GROUPS]; OUTG],
        x_temp: [10.0; IN]
    };
    d
}

fn rand() -> Data {
    let mut d = Data {
        qweight: [[0; ING]; OUT],
        scales: [[-1.0; GROUPS]; OUT],
        qzeros:  [[1985229328; GROUPS]; OUTG],
        //qzeros:  [[0; GROUPS]; OUTG],
        x_temp: [1.0; IN]
    };
    let mut rng = rand::thread_rng();

     for x in d.x_temp.iter_mut() {
         *x = rng.gen();
     }
    for o in 0..OUT {
        for g in 0..GROUPS {
             d.scales[o][g] = rng.gen();
        }
        for i in 0..ING {
            d.qweight[o][i] = rng.gen();
        }
    }
    for o in 0..OUTG {
        for g in 0..GROUPS {
            d.qzeros[o][g] = rng.gen();
        }
    }
    // for i in 0..GROUPS {
    //     d.qzeros[0][i] = 0;
    // }
    //d.qzeros[0][1] = 1;
    d
}

fn matvec(d: &Data) {
    let ten = f32x8::splat(0.0);
    let qweight = d.qweight;
    let scales = d.scales;
    let qzeros = d.qzeros;

    let mut x_temp: [[f32x8; B]; IN/8] = [[ten; B];IN/8];
    for i in 0..IN/8 {
        x_temp[i][0] = f32x8::from_slice(&d.x_temp[i * 8..][..8]);
    }


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
        println!("{:?}", &xout_temp[..20]);
    // for i in 0..xout.len() {
    //         for j in 0..OUT {
    //             xout[i][j] = xout_temp[j][i];
    //         }
    //     }
    
}




fn run()  -> Result<(), (Box<dyn Error>)> {
    let d = rand();
    matvec(&d);
    let qweight: [[i32; ING]; OUT] = d.qweight;
    let scales: [[f32; GROUPS]; OUT] = d.scales;
    let qzeros:  [[i32; GROUPS]; OUTG] = d.qzeros;
    let x_temp: [f32; IN] = d.x_temp;
    let xout_temp: [f32; 1000000] = [0.0; 1000000];
    let mut xout: [[f32; OUT]; B] = [[0.0; OUT];B];
    let n_elements = OUT * IN;
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
    let block_size = 128 as usize;
    let eblock1_size: usize = 2048 as usize;
    let eblock2_size: usize =  16 as usize;
    let grid_size1 = 1; //(((IN ) as usize) + eblock1_size - 1) / eblock1_size;
    let grid_size2 = 1; // (((OUT) as usize) + eblock2_size - 1) / eblock2_size;
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
                    func<<<(grid_size1 as u32, grid_size2 as u32), block_size as u32, 9216, stream>>>(
                        qweight_gpu.as_device_ptr(),
                        qscale_gpu.as_device_ptr(),
                        qzeros_gpu.as_device_ptr(),
                        xtemp_gpu.as_device_ptr(),
                        out_buf.as_device_ptr(),
                        IN as i32
                    )
                )?;
            }
            println!("sync");
            stream.synchronize()?;
    let xout_temp = out_buf.as_slice().as_host_vec()?;
    //println!("{:?}", &xout_temp[..IN / 256 * 16]);
    // for i in 0..(OUT / eblock2_size) {
    //     for k in 0..eblock2_size {
    //         xout[0][eblock2_size * i + k] = 0.0;
    //         for j in 0..IN / eblock1_size as usize {
    //             xout[0][eblock2_size * i + k] += xout_temp[i * eblock2_size * (IN/eblock1_size) + j * eblock2_size + k];
    //         }
    //         //println!("{} {}", 32 * i + k,  xout[0][32 * i + k] );
    //     }
    // }
    
    println!("{:?}", &xout_temp[..20]);

    Ok(())
}

