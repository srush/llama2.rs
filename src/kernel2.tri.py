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
    BLOCK_SIZE_IN: tl.constexpr,
    BLOCK_SIZE_OUT: tl.constexpr):
    print("start")
    pid_in = tl.program_id(axis=0) 
    pid_out = tl.program_id(axis=1) 
   
    GROUP_SIZE: tl.constexpr = 128
    BITS: tl.constexpr = 4
    mask = 2**4-1
    shift = tl.arange(0, 8) * BITS
    
    # STEP 1: Load the input data from x
    print("step1")
    x = tl.load(x_ptr + pid_in * BLOCK_SIZE_IN + tl.arange(0, BLOCK_SIZE_IN)[None,:] + (tl.arange(0, BLOCK_SIZE_OUT)[:, None] * 0) )
    tl.view(x, (BLOCK_SIZE_OUT, BLOCK_SIZE_IN))

    # STEP 2: Load the scaling data 
    # scale (IN/ GROUPSIZE x OUT)
    print("step2")
    BLOCK_IN_GROUP : tl.constexpr = BLOCK_SIZE_IN // GROUP_SIZE
    stride_0 = IN // GROUP_SIZE
    scale_in_shape = pid_in * BLOCK_IN_GROUP + tl.arange(0, BLOCK_IN_GROUP) 
    scale_out_shape = pid_out * stride_0 * BLOCK_SIZE_OUT + (stride_0 * tl.arange(0, BLOCK_SIZE_OUT))
    scale_shape = scale_in_shape[None, :] + scale_out_shape[:, None]
    scale = tl.load(qscale_ptr + scale_shape)
    scale = tl.view(scale, (BLOCK_SIZE_OUT * BLOCK_IN_GROUP,))

    # STEP 3: Load the zeros
    # zeros (IN/ GROUPSIZE x OUT/8)
    print("step3")
    stride_0 = IN // GROUP_SIZE
    ZERO_BLOCK_OUT : tl.constexpr = BLOCK_SIZE_OUT // 8
    # IN
    zero_in_shape = pid_in * BLOCK_IN_GROUP +  tl.arange(0, BLOCK_IN_GROUP)
    #OUT
    zero_out_shape = pid_out * stride_0 * ZERO_BLOCK_OUT + stride_0 * tl.arange(0, ZERO_BLOCK_OUT)
    zero_shape = zero_in_shape[None, :] + zero_out_shape[:, None]
    zero_shape = tl.view(zero_shape, (ZERO_BLOCK_OUT, BLOCK_IN_GROUP))
    zeros = tl.load(qzeros_ptr + zero_shape)

    # # Zeros are stored in an int, by out position. 
    zeros = ((zeros[:,  None, :] >> shift[None, :, None]) & mask) + 1.0
    tl.view(zeros, (ZERO_BLOCK_OUT, 8,  BLOCK_IN_GROUP))
    # Same as scale now. 
    zeros = tl.reshape(zeros, (BLOCK_SIZE_OUT * BLOCK_IN_GROUP, 1))

    # Step 4: Unpack quantized values 
    # val (IN // 8 x OUT)
    print("step4")
    BLOCK_IN_8 : tl.constexpr = BLOCK_SIZE_IN // 8
    stride_0 = IN // 8
    val_shape_in = pid_in * BLOCK_IN_8 + tl.arange(0, BLOCK_IN_8)
    tl.view(val_shape_in, (BLOCK_IN_8,))
    val_shape_out = pid_out * stride_0 * BLOCK_SIZE_OUT + stride_0 * tl.arange(0, BLOCK_SIZE_OUT)
    tl.view(val_shape_out, (BLOCK_SIZE_OUT,))    
    val_shape = val_shape_in[None, :] + val_shape_out[:, None]
    # tl.view(val_shape, (BLOCK_SIZE_OUT, BLOCK_IN_8))    
    vals = tl.load(qweight_ptr + tl.view(val_shape, (BLOCK_SIZE_OUT * BLOCK_IN_8,)))
    
    # Shift out the values by in_pos
    vals = (vals[:, None] >> shift[None, :]) & mask
    tl.view(vals, (BLOCK_SIZE_OUT * BLOCK_IN_8, 8))
    vals = tl.reshape(vals, (BLOCK_SIZE_OUT * BLOCK_IN_GROUP, GROUP_SIZE))

    # Step 5: Do the offset, multiplication and reshape
    print("step5")
    out = scale[:, None] * (vals[:, :] - zeros[:, :])
    out = tl.sum(out * x, axis=1)
    tl.view(out, (BLOCK_SIZE_OUT,))

    # Step 6: Write out.     
    tl.store(y_ptr + pid_in * BLOCK_SIZE_OUT + pid_out * BLOCK_SIZE_OUT * (IN // BLOCK_SIZE_IN) + tl.arange(0, BLOCK_SIZE_OUT), 
             out)

IN = 4096
OUT = 4096
GROUPSIZE = 128

q_weights = torch.tensor([[1985229328] * (IN // 8)] * OUT).int().cuda()
q_scale = torch.tensor([[10.] * (IN // GROUPSIZE)] * OUT).float().cuda()
q_zeros = torch.tensor([[1985229328] * (IN // GROUPSIZE)] * (OUT // 8)).int().cuda()
x = torch.tensor([[10.] * IN]).cuda()
BLOCK_SIZE = 128
OUT_VALS = 8
n_elements = OUT * IN
q_out = torch.zeros(OUT // OUT_VALS, IN // BLOCK_SIZE, OUT_VALS).float().cuda()
print(q_weights.shape)
print(q_scale.shape)
print(q_zeros.shape)


grid = lambda meta:  (triton.cdiv(IN, meta['BLOCK_SIZE_IN']),
                    triton.cdiv(OUT, meta['BLOCK_SIZE_OUT']))
gptq_kernel[grid](q_weights, q_scale, q_zeros, x, q_out, IN, 
                  BLOCK_SIZE_IN=BLOCK_SIZE,
                BLOCK_SIZE_OUT=OUT_VALS)
print(q_out.shape)
r = q_out.sum(1).view(-1)
print(r[:20])
print(dir(gptq_kernel.cache))
with open("gptq.ptx", "w") as a:
    print(list(gptq_kernel.cache[0].values())[0].asm['ptx'], file=a)
