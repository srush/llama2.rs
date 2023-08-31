// Quant 4 bits
use crate::constants::{BITS};
use cust::memory::DeviceCopy;
use cust::prelude::*;
use std::error::Error;
use crate::util::{time_in_ms};
use crate::gptq::{QLinear};
use half::f16;

#[derive(Default,Clone,Copy,Debug)]
struct F16 ( f16 );
unsafe impl DeviceCopy for F16 { }
static PTX: &str = include_str!("gptq.ptx");

impl<
        const IN: usize,
        const OUT: usize,
        const GROUPS: usize,
        const ING: usize,
        const OUTG: usize,
    > QLinear<IN, OUT, GROUPS, ING, OUTG>
{
    pub fn convert(self: &Self) -> Result<QLinearCuda<IN, OUT, GROUPS, ING, OUTG>, Box<dyn Error>> {
        let t: [F16; OUT] = [F16(f16::from_f32(0.0)); OUT];
        let out = QLinearCuda {
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
pub struct QLinearCuda<
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
    xout_temp :  DeviceBuffer<F16>,
    module : Module
}

impl<
    const IN: usize,
    const OUT: usize,
    const GROUPS: usize,
    const ING: usize,
    const OUTG: usize,
> QLinearCuda<IN, OUT, GROUPS, ING, OUTG>
{

    pub fn matvec<const B: usize>(self: &Self, xout: &mut [[f32; OUT]; B], x: &[[f32; IN]; B]) {
        assert_eq!(ING, IN / 32 * BITS);
        assert_eq!(OUTG, OUT / 32 * BITS);
        assert_eq!(B, 1);

        let mut x_temp: [[F16; IN]; B] = [[F16(f16::from_f32(0.0)); IN]; B];
        // Remap the input for ACT_ORDER=True
        for i in 0..B {
            for j in 0..IN {
                x_temp[i][j] = F16(f16::from_f32(x[i][self.g_index[j] as usize]));
            }
        }

        let res = | | {
        let xtemp_gpu = x_temp[0].as_dbuf()?;            
        
        // retrieve the add kernel from the module so we can calculate the right launch config.
        let func = self.module.get_function("gptq_kernel_0d1d2d3d4d5d").expect("couldn't find kernel");
        let block_size = 128;
        let eblock1_size = 2048 as usize;
        let eblock2_size = 8 as usize;
        let grid_size1 = 1; 
        let grid_size2 = ( ((OUT) as usize)  + (eblock2_size) - 1) / eblock2_size;

        // GPU calls.
        let stream = Stream::new(StreamFlags::NON_BLOCKING, None)?;
        
        unsafe {
            let a = self.qweight.as_device_ptr();
            let b = self.scales.as_device_ptr();
            let c = self.qzeros.as_device_ptr();
            let d = xtemp_gpu.as_device_ptr();
            let e = self.xout_temp.as_device_ptr();
            launch!(
                // slices are passed as two parameters, the pointer and the length.
                func<<<(grid_size1 as u32, grid_size2 as u32), block_size, 9216 , stream>>>(
                    a,b,c,d,e, IN as i32
                )
            )?;
        }
        stream.synchronize()?;
        Ok::<(), Box<dyn Error>>(())
    };
    res().expect("gpu failure");
    let xout_temp = self.xout_temp.as_slice().as_host_vec().expect("copy failure");

        xout[0] = xout_temp.iter().map(|x| f16::to_f32(x.0)).collect::<Vec<_>>().try_into().unwrap(); 
    }
}
