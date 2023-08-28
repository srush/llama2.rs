// Quant 4 bits
use crate::constants::{BITS, GROUPSIZE};
use cust::prelude::*;
use std::error::Error;
use crate::util::{argmax, time_in_ms, Random};
/// How many numbers to generate and add together.

static PTX: &str = include_str!("gptq.ptx");
#[repr(C)]
#[derive(Debug)]
pub struct QLinear<
    const IN: usize,
    const OUT: usize,
    const GROUPS: usize,
    const ING: usize,
    const OUTG: usize,
> {
    // The quantized weights
    qweight: [[i32; ING]; OUT],
    // The zero term per group.
    qzeros: [[i32; GROUPS]; OUTG],
    // The scale term per group
    scales: [[f32; GROUPS]; OUT],
    // Remapping for ACT_ORDER=True
    pub g_index: [i32; IN],
}


impl<
        const IN: usize,
        const OUT: usize,
        const GROUPS: usize,
        const ING: usize,
        const OUTG: usize,
    > QLinear<IN, OUT, GROUPS, ING, OUTG>
{
    pub fn convert(self: &Self) -> Result<QLinear2<IN, OUT, GROUPS, ING, OUTG>, Box<dyn Error>> {
        let t: [f32; 500000] = [0.0; 500000];
        let out = QLinear2 {
            qweight: self.qweight.as_dbuf()?,
            scales: self.scales.as_dbuf()?,
            qzeros: self.qzeros.as_dbuf()?,
            g_index: self.g_index.clone(),
            xout_temp: t.as_dbuf()?,
            module : Module::from_ptx(PTX, &[])?
        };
        Ok(out)
    }
}
#[derive(Debug)]
pub struct QLinear2<
    const IN: usize,
    const OUT: usize,
    const GROUPS: usize,
    const ING: usize,
    const OUTG: usize,
> {
    // The quantized weights
    qweight: DeviceBuffer<[i32; ING]>,
    // The zero term per group.
    qzeros: DeviceBuffer<[i32; GROUPS]>,
    // The scale term per group
    scales: DeviceBuffer<[f32; GROUPS]>,
    // Remapping for ACT_ORDER=True
    pub g_index: [i32; IN],
    xout_temp :  DeviceBuffer<f32>,
    module : Module
}

impl<
    const IN: usize,
    const OUT: usize,
    const GROUPS: usize,
    const ING: usize,
    const OUTG: usize,
> QLinear2<IN, OUT, GROUPS, ING, OUTG>
{

    pub fn matvec<const B: usize>(self: &Self, xout: &mut [[f32; OUT]; B], x: &[[f32; IN]; B]) -> Result<(), (Box<dyn Error>)> {
        assert_eq!(ING, IN / 32 * BITS);
        assert_eq!(OUTG, OUT / 32 * BITS);
        assert_eq!(B, 1);
        let s = time_in_ms();
        let q_time = time_in_ms();
        // Transpose the output.
        //let xout_temp: [f32; 500000] = [0.0; 500000];
        
        // This should be IN / 8 but rust can't handle this statically.
        let mut x_temp: [[f32; IN]; B] = [[0.0; IN]; B];

        // Remap the input for ACT_ORDER=True
        for i in 0..B {
            for j in 0..IN {
                x_temp[i][j] = x[i][self.g_index[j] as usize];
            }
        }
        
        // Transpose the input.
        // let mut x_temp = [[z; B]; IN];
        // for i in 0..IN / 8 {
        //     for j in 0..B {
        //         x_temp[i][j] = f32x8::from_slice(&x_temp1[j][i * 8..][..8]);
        //     }
        // }
        // Enumerate over each output position.
 
            // initialize CUDA, this will pick the first available device and will
            // make a CUDA context from it.
            // We don't need the context for anything but it must be kept alive.

            
           
            // Make the CUDA module, modules just house the GPU code for the kernels we created.
            // they can be made from PTX code, cubins, or fatbins.

            
//            println!("start");
            // make a CUDA stream to issue calls to. You can think of this as an OS thread but for dispatching
            
        //let qweight_gpu = self.qweight.as_dbuf()?;
        //let qscale_gpu = self.scales.as_dbuf()?;
        //let qzeros_gpu = self.qzeros.as_dbuf()?;
        let xtemp_gpu = x_temp[0].as_dbuf()?;            
        //let out_buf = xout_temp.as_dbuf()?;
        //let module = Module::from_ptx(PTX, &[])?;
          let s = time_in_ms();
            // retrieve the add kernel from the module so we can calculate the right launch config.
        let func = self.module.get_function("gptq_kernel_0d1d2d3d4d5d").expect("couldn't find kernel");
        let block_size = 128;
        let eblock_size = 256;
            let grid_size = ( ((IN * OUT) as u32)  + eblock_size - 1) / eblock_size;

            // GPU calls.
            let stream = Stream::new(StreamFlags::NON_BLOCKING, None)?;
            unsafe {
                let a = self.qweight.as_device_ptr();
                let b = self.scales.as_device_ptr();
                let c = self.qzeros.as_device_ptr();
                let d = xtemp_gpu.as_device_ptr();
                let e = self.xout_temp.as_device_ptr();
                let s = time_in_ms();
                launch!(
                    // slices are passed as two parameters, the pointer and the length.
                    func<<<grid_size, block_size, 512, stream>>>(
                        a,b,c,d,e, IN
                    )
                )?;
            }

        stream.synchronize()?;
        // let s = time_in_ms();
        // // copy back the data from the GPU.
            let xout_temp = self.xout_temp.as_slice().as_host_vec()?;
            // println!("{:?}", &xout_temp[..10]);
            for i in 0..xout.len() {
                for j in 0..OUT {
                    xout[i][j] = 0.0;
                    for k in 0..IN / eblock_size as usize {
                        xout[i][j] += xout_temp[j * (IN / eblock_size as usize) + k];
                    }
                }
            }
           // println!("{:?}", xout);
        Ok(())
    }
}
