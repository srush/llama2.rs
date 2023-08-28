import torch

import triton
import triton.language as tl


@triton.jit
def gptq_kernel(
    qweight_ptr,  # *Pointer* to qweight int input matrix.
    qscale_ptr,  # *Pointer* to qsscale f16 input matrix.
    qzeros_ptr,  # *Pointer* to qzeros int input matrix.
    x_ptr, # *Pointer* to x input vector.
    y_ptr, # *Pointer* to y output vector.
    IN,
    BLOCK_SIZE: tl.constexpr,
    OUT_VALS: tl.constexpr):

    pid = tl.program_id(axis=0) # We use a 1D launch grid so axis is 0.
   
    GROUP_SIZE: tl.constexpr = 128
    BITS = 4
    mask = 2**4-1
    shift = tl.arange(0, 8) * BITS

    
    # Id of the in and out start elements
    in_elem = BLOCK_SIZE % IN
    out_elem = OUT_VALS * (BLOCK_SIZE // IN)
    
    
    # x is BLOCK_SIZE in in chunks
    x = tl.load(x_ptr + pid * in_elem + tl.arange(0, BLOCK_SIZE))[None, :]

    # scale is taken from the current group
    scale_block : tl.constexpr = (BLOCK_SIZE  + (GROUP_SIZE - 1))  // GROUP_SIZE
    group = pid * scale_block + tl.arange(0, scale_block)
    group = group[None, :] + (tl.arange(0, OUT_VALS) * (IN // GROUP_SIZE))[:, None]
    scale = tl.load(qscale_ptr + group)

    # zeros are taken repeat groupsize times
    zero_block : tl.constexpr = (BLOCK_SIZE  + (8 * GROUP_SIZE - 1)) // (8 * GROUP_SIZE)
    zero_groups = pid * zero_block + tl.arange(0, zero_block)
    zero_groups = zero_groups[None, :] + (tl.arange(0, OUT_VALS) * (IN // 8 // GROUP_SIZE))[:, None]
    zeros = tl.load(qzeros_ptr + zero_groups)

    # Shift for the zeros
    out_groups = zero_groups * 8 * GROUP_SIZE
    zero_shift = (out_groups % 8) * BITS
    zeros = ((zeros >> zero_shift) & mask) + 1

    # Compute
    val_block : tl.constexpr = (BLOCK_SIZE + (8 - 1))  // 8
    val_groups = pid * val_block + tl.arange(0, val_block)
    val_groups = val_groups[None, :] + (tl.arange(0, OUT_VALS) * (IN // 8))[:, None]
    splat = tl.load(qweight_ptr + val_groups)
    print(splat.shape)
    vals = (splat[:, :, None] >> shift[None, None, :]) & mask
    print(vals.shape)
    vals = tl.view(vals, (OUT_VALS, scale_block, GROUP_SIZE))

    # Merge
    print(vals.shape, zeros.shape)
    out = scale[:, :, None] * (vals[:, :, :] - zeros[:, :, None])

    out = tl.reshape(out, (OUT_VALS, BLOCK_SIZE))
    out = tl.sum(out * x, axis=1)
    out = tl.view(out, (OUT_VALS,))
    tl.store(y_ptr + OUT_VALS * pid + tl.arange(0, OUT_VALS), out)


IN = 4096
OUT = 4096
GROUPSIZE = 128

q_weights = torch.tensor([[1985229328] * (IN // 8)] * OUT).int().cuda()
q_scale = torch.tensor([[10.] * (IN // GROUPSIZE)] * OUT).float().cuda()
q_zeros = torch.tensor([[1985229328] * (IN // GROUPSIZE)] * (OUT // 8)).int().cuda()
x = torch.tensor([[10.] * IN]).cuda()
BLOCK_SIZE = 1024
OUT_VALS = 16
n_elements = OUT * IN
q_out = torch.zeros(OUT_VALS * (n_elements // BLOCK_SIZE) * 10,).float().cuda()
print(q_weights.shape)
print(q_scale.shape)
print(q_zeros.shape)


grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE'] * meta['OUT_VALS']),)
gptq_kernel[grid](q_weights, q_scale, q_zeros, x, q_out, IN, 
                  OUT_VALS=OUT_VALS,
                  BLOCK_SIZE=BLOCK_SIZE)
print(q_out)

print(dir(gptq_kernel.cache))
with open("gptq.ptx", "w") as a:
    print(list(gptq_kernel.cache[0].values())[0].asm['ptx'], file=a)
