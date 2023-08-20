// Quant 4 bits
use crate::constants::{BITS, GROUPSIZE};
use rayon::prelude::*;
/// Code for quantized SIMD implementation.
use std::simd::{f32x8, i32x8, SimdFloat, SimdInt};

#[repr(C)]
pub struct QLinear<
    const IN: usize,
    const OUT: usize,
    const GROUPS: usize,
    const ING: usize,
    const OUTG: usize,
> {
    qweight: [[i32; ING]; OUT],
    qzeros: [[i32; GROUPS]; OUTG],
    scales: [[f32; GROUPS]; OUT],
    g_index: [i32; IN],
}

impl<
        const IN: usize,
        const OUT: usize,
        const GROUPS: usize,
        const ING: usize,
        const OUTG: usize,
    > QLinear<IN, OUT, GROUPS, ING, OUTG>
{
    pub fn matvec<const B: usize>(self: &Self, xout: &mut [[f32; OUT]; B], x: &[[f32; IN]; B]) {
        assert_eq!(ING, IN / 32 * BITS);
        assert_eq!(OUTG, OUT / 32 * BITS);

        //B = xout.len();
        // // Transpose the input and output.
        let mut xout_temp = [[0.0; B]; OUT];
        for i in 0..OUT {
            for j in 0..B {
                xout_temp[i][j] = xout[j][i];
            }
        }
        let z = f32x8::splat(0.0);
        // This should be IN / 8 but rust can't handle this statically.
        let mut x_temp1 = [[0.0; IN]; B];
        for i in 0..B {
            for j in 0..IN {
                x_temp1[i][j] = x[i][self.g_index[j] as usize];
            }
        }

        let mut x_temp = [[z; B]; IN];
        for i in 0..IN / 8 {
            for j in 0..B {
                //for k in 0 .. 8 {
                //                        x_temp[i * B + j][k] = x[j][i * 8 + k];
                x_temp[i][j] = f32x8::from_slice(&x_temp1[j][i * 8..][..8]);
                //}
            }
        }

        let mask = (1 << BITS) - 1;
        let elems_per_i32 = 32 / BITS;
        let ipg: usize = GROUPSIZE / 32 * BITS;
        let mask_4bits = i32x8::splat(mask);
        let shift_right = i32x8::from_array([0, 4, 8, 12, 16, 20, 24, 28]);

        // Check the output.
        xout_temp.par_iter_mut().enumerate().for_each(|(oi, o)| {
            // Do K at a time
            let qzeros = &self.qzeros[oi / elems_per_i32];
            let out_elem = oi % elems_per_i32;
            let qweight = self.qweight[oi].chunks_exact(ipg);
            let mut sum = [z; B];
            o.fill(0.0);
            for (((scale, qweight), x_temp), qzs) in self.scales[oi]
                .into_iter()
                .zip(qweight)
                .zip(x_temp.chunks_exact(GROUPSIZE / 8))
                .zip(qzeros)
            {
                let qz = ((qzs >> (BITS * out_elem)) & mask) + 1;
                let scale_simd = f32x8::splat(scale);
                let zero_simd = i32x8::splat(qz);

                for (&v, x) in qweight.into_iter().zip(x_temp) {
                    //Extract v into 8 chunks
                    let num_simd = i32x8::splat(v);
                    let qw: i32x8 = (num_simd >> shift_right) & mask_4bits;
                    let combine: f32x8 = (qw - zero_simd).cast::<f32>();
                    let weight: f32x8 = scale_simd * combine;

                    for (&x, s) in x.iter().zip(sum.iter_mut()) {
                        //let x = f32x8::from_slice(x);
                        //*xout += (weight * x).reduce_sum();
                        *s += weight * x;
                    }
                }
            }
            for (xout, s) in o.iter_mut().zip(sum) {
                *xout += s.reduce_sum();
            }
        });

        for i in 0..xout.len() {
            for j in 0..OUT {
                xout[i][j] = xout_temp[j][i];
            }
        }
    }
}
