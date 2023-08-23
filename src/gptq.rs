// Quant 4 bits
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
    // The quantized weights
    qweight: [[i32; ING]; OUT],
    // The zero term per group.
    qzeros: [[i32; GROUPS]; OUTG],
    // The scale term per group
    scales: [[f32; GROUPS]; OUT],
    // Remapping for ACT_ORDER=True
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


        // Transpose the output.
        let mut xout_temp = [[0.0; B]; OUT];
        for i in 0..OUT {
            for j in 0..B {
                xout_temp[i][j] = xout[j][i];
            }
        }
        let z = f32x8::splat(0.0);
        // This should be IN / 8 but rust can't handle this statically.
        let mut x_temp1 = [[0.0; IN]; B];

        // Remap the input for ACT_ORDER=True
        for i in 0..B {
            for j in 0..IN {
                x_temp1[i][j] = x[i][self.g_index[j] as usize];
            }
        }

        // Transpose the input.
        let mut x_temp = [[z; B]; IN];
        for i in 0..IN / 8 {
            for j in 0..B {
                x_temp[i][j] = f32x8::from_slice(&x_temp1[j][i * 8..][..8]);
            }
        }

        // Constants for 4 bit.
        let mask = (1 << BITS) - 1;
        let elems_per_i32 = 32 / BITS;
        let ipg: usize = GROUPSIZE / 32 * BITS;
        let mask_4bits = i32x8::splat(mask);
        let shift_right = i32x8::from_array([0, 4, 8, 12, 16, 20, 24, 28]);

        // Enumerate over each output position.
        xout_temp.par_iter_mut().enumerate().for_each(|(oi, o)| {
            let qzeros = &self.qzeros[oi / elems_per_i32];
            let out_elem = oi % elems_per_i32;
            let qweight = self.qweight[oi].chunks_exact(ipg);
            let mut sum = [z; B];

            // Everything in the batch starts at zero.
            o.fill(0.0);

            // Iterate over weight groups (32/64/128).
            for (((scale, qweight), x_temp), qzs) in self.scales[oi]
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
        for i in 0..xout.len() {
            for j in 0..OUT {
                xout[i][j] = xout_temp[j][i];
            }
        }
    }
}
