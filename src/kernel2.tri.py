import torch

import triton
import triton.language as tl

@triton.jit
def old_gptq_kernel(
    qweight_ptr,  # *Pointer* to qweight int input matrix.
    qscale_ptr,  # *Pointer* to qsscale f16 input matrix.
    qzeros_ptr,  # *Pointer* to qzeros int input matrix.
    x_ptr, # *Pointer* to x input vector.
    y_ptr, # *Pointer* to y output vector.
    IN,
    BLOCK_SIZE: tl.constexpr):

    pid = tl.program_id(axis=0) # We use a 1D launch grid so axis is 0.
    block_start = pid * BLOCK_SIZE
    block = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    group_size: tl.constexpr = 128
    BITS = 4
    mask = 2**4-1

    in_elem = block % IN
    out_elem = block // IN
    
    # x is just group size
    x = tl.load(x_ptr + in_elem)

    # scale is taken from the current group
    group = block // group_size
    scale = tl.load(qscale_ptr + group)

    # zeros are taken repeat groupsize times
    zeros = tl.load(qzeros_ptr + (in_elem * out_elem // 8 ) // group_size)
    zero_shift = (out_elem % 8) * BITS
    zeros = ((zeros >> zero_shift) & mask) + 1

    # Compute
    offset = (block % 8) * BITS
    splat = tl.load(qweight_ptr + (block // 8))
    vals = (splat >> offset) & mask
    out = scale * (vals - zeros) * x    
    tl.store(y_ptr + pid, tl.sum(out, axis=0))


# @triton.jit
# def test(a, b):



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
    pid_out_o = tl.program_id(axis=1) 
   
    GROUP_SIZE: tl.constexpr = 128
    BITS: tl.constexpr = 4
    mask = (1 << BITS) - 1
    shift = tl.arange(0, 8) * BITS
    BLOCK_IN_GROUP : tl.constexpr = BLOCK_SIZE_IN // GROUP_SIZE
    
    # Attempted to make an inner loop but failed.
    total1 = tl.zeros((BLOCK_SIZE_OUT,), tl.float32) 
    ZERO_BLOCK_OUT : tl.constexpr = BLOCK_SIZE_OUT // 8
    BLOCK_IN_8 : tl.constexpr = BLOCK_SIZE_IN // 8
    pid_out = pid_out_o
    total0 = tl.zeros((BLOCK_SIZE_OUT,), tl.float32)
    
    for pid_in in range(tl.cdiv(IN, BLOCK_SIZE_IN)):
        # STEP 1: Load the input data from x
        print("step1")
        x = tl.load(x_ptr + pid_in * BLOCK_SIZE_IN + tl.arange(0, BLOCK_SIZE_IN)[None, :], 
                    mask=(pid_in * BLOCK_SIZE_IN + tl.arange(0, BLOCK_SIZE_IN) < IN)[None, :],
                    other=0.0)

        # STEP 2: Load the scaling data 
        # scale (IN/ GROUPSIZE x OUT)
        print("step2")
        stride_0 = IN // GROUP_SIZE
        scale_in_shape = pid_in * BLOCK_IN_GROUP + tl.arange(0, BLOCK_IN_GROUP) 
        scale_out_shape = pid_out * stride_0 * BLOCK_SIZE_OUT + stride_0 * tl.arange(0, BLOCK_SIZE_OUT)
        scale_shape = scale_in_shape[None, :] + scale_out_shape[:, None]
        scale_shape = tl.view(scale_shape, (BLOCK_SIZE_OUT, 1, BLOCK_IN_GROUP))
        scale = tl.load(qscale_ptr + scale_shape)

        # STEP 3: Load the zeros
        # zeros (IN/ GROUPSIZE x OUT/8)

        print("step3")
        stride_0 = IN // GROUP_SIZE
        zero_in_shape = pid_in * BLOCK_IN_GROUP +  tl.arange(0, BLOCK_IN_GROUP)
        zero_out_shape = pid_out * stride_0 * ZERO_BLOCK_OUT + stride_0 * tl.arange(0, ZERO_BLOCK_OUT)
        #zero_out_shape = pid_out * stride_0 * ZERO_BLOCK_OUT + stride_0 * (tl.arange(0, BLOCK_SIZE_OUT) // 8)
        zero_shape = zero_in_shape[None, :] + zero_out_shape[:, None]  
        zero_shape = tl.view(zero_shape, (1, ZERO_BLOCK_OUT, BLOCK_IN_GROUP))
        #zero_shape = tl.view(zero_shape, (BLOCK_SIZE_OUT, 1, BLOCK_IN_GROUP))
        zeros = tl.load(qzeros_ptr + zero_shape)
        #tl.store(debug + BLOCK_IN_GROUP * tl.arange(0, BLOCK_SIZE_OUT)[:,  None] + tl.arange(0, BLOCK_IN_GROUP)[None, :], 
        #         zeros )
        #zeros = zeros[:, None, :]
        
        # Zeros are stored in an int, by out position. 
        zeros = ((zeros >> shift[:, None, None]) & mask).to(tl.int8) + 1
        #zero_shift = (tl.arange(0,BLOCK_SIZE_OUT)[:, None , None] % 8) * BITS
        #zeros = ((zeros >> zero_shift) & mask) + 1
        # Merge ZERO_BLOCK_OUT and with 8 Dim
        #zeros = tl.view(zeros, (BLOCK_SIZE_OUT, BLOCK_IN_GROUP))
        #tl.store(debug + BLOCK_IN_GROUP * tl.arange(0, BLOCK_SIZE_OUT)[:,  None] + tl.arange(0, BLOCK_IN_GROUP)[None, :], 
        #         zeros)
        zeros = tl.view(zeros, (BLOCK_SIZE_OUT, 1, BLOCK_IN_GROUP))

        # Step 4: Unpack quantized values 
        # val (IN // 8 x OUT)
        print("step4")
        stride_0 = IN // 8
        val_shape_in = pid_in * BLOCK_IN_8 + tl.arange(0, BLOCK_IN_8)
        val_shape_out = pid_out * stride_0 * BLOCK_SIZE_OUT + stride_0 * tl.arange(0, BLOCK_SIZE_OUT)
        val_shape = val_shape_in[None, :] + val_shape_out[:, None]
        val_shape = tl.view(val_shape, (BLOCK_SIZE_OUT, 1, BLOCK_IN_8))    
        vals = tl.load(qweight_ptr + val_shape)
        
        # Shift out the values by in_pos
        vals = ((vals >> shift[None, :, None]) & mask)
        # Merge BLOCK_IN_8 and with 8 Dim, pull out GROUP_SIZE dim
        vals = tl.view(vals, (BLOCK_SIZE_OUT, BLOCK_SIZE_IN))
        # Split pull out GROUP_SIZE dim
        vals = tl.view(vals, (BLOCK_SIZE_OUT, GROUP_SIZE, BLOCK_IN_GROUP))
        
        # # Step 5: Do the offset, multiplication and reshape
        print("step5")
        out = scale * (vals - zeros)
        # Merge GROUP_SIZE, BLOCK_IN_GROUP
        out = tl.view(out, (BLOCK_SIZE_OUT, BLOCK_SIZE_IN))
        out = out * x
        out = tl.sum(out, axis=1)
        out = tl.view(out, (BLOCK_SIZE_OUT,))
        total0 = total0 + out 
    # # Step 6: Write out.     
    # out OUT // BLOCK_SIZE_OUT x IN // BLOCK_SIZE_IN x BLOCK_SIZE_OUT
    tl.store(y_ptr + pid_out_o * BLOCK_SIZE_OUT +  tl.arange(0, BLOCK_SIZE_OUT), total0)
    
IN = 4096
OUT = 4096
GROUPSIZE = 128

q_weights = torch.tensor([[0] * (IN // 8)] * OUT).int().cuda()
q_scale = torch.tensor([[-1.] * (IN // GROUPSIZE)] * OUT).float().cuda()
q_zeros = torch.tensor([[1985229328] * (IN // GROUPSIZE)] * (OUT // 8)).int().cuda()
x = torch.tensor([[1.] * IN]).cuda()

# q_weights = torch.randint(0, 100, (OUT, (IN // 8))).int().cuda()
# q_scale = torch.rand( (OUT,  (IN // GROUPSIZE))).float().cuda()
# q_zeros = torch.randint(0, 100000, (OUT//8, (IN // GROUPSIZE))).int().cuda()
# x = torch.rand((IN)).cuda()
BLOCK_SIZE = 2048
OUT_VALS = 8
# BITS = 4
# mask = 2**4-1
# shift = torch.arange(0, 8).cuda() * BITS
# print(q_zeros.shape)
# for pid_in in range(IN // 256):
#     zero_in_shape =  pid_in * (256 //256) + torch.arange(0, IN//256)
#     zero_out_shape = (1) * torch.arange(0, 1)
#     zero_shape = zero_in_shape[None, :] + zero_out_shape[:, None]
#     qz = q_zeros.ravel()[zero_shape]
#     #qz = q_zeros[:1, pid:16]
#     print(qz.shape)
#     out = (qz[:, None, : ] >> shift[None, :, None]) & mask
#     print(out.shape)
#     out = out.view((8, -1))
#     print(out[0, :2])
#     print(out[1, :2])
#     print(out[2, :2])


n_elements = OUT * IN
q_out = torch.zeros(OUT).float().cuda().half()
print(q_weights.shape)
print(q_scale.shape)
print(q_zeros.shape)

debug = torch.zeros(BLOCK_SIZE * OUT_VALS).float().cuda()

grid = lambda meta: (1, 1)
# triton.cdiv(IN, meta['BLOCK_SIZE_IN']),
#                      triton.cdiv(OUT, meta['BLOCK_SIZE_OUT']))
debug = torch.zeros(OUT_VALS//8, BLOCK_SIZE // GROUPSIZE).int().cuda()
q_zeros[0][:] = 0
q_zeros[1][:] = 999999999
gptq_kernel[grid](q_weights, q_scale, q_zeros, x.half(), q_out, #debug,
                  IN, 
                  BLOCK_SIZE_IN=BLOCK_SIZE,
                  BLOCK_SIZE_OUT=OUT_VALS)
print(q_out[:20])
print(debug)
# print(q_out.shape)
# r = q_out.sum(1).view(-1)
# print(r)

BLOCK_SIZE=256
grid = lambda meta: (triton.cdiv(IN * OUT, meta['BLOCK_SIZE']),)
q_out = torch.zeros(OUT, IN // BLOCK_SIZE).float().cuda().half()
old_gptq_kernel[grid](q_weights, q_scale, q_zeros, x, q_out, IN, 
                   BLOCK_SIZE=BLOCK_SIZE)
print(q_out.sum(1).view(-1))


# print(r.shape)
# print(list(gptq_kernel.cache[0].values())[0].asm.keys())
#print(list(gptq_kernel.cache[0].values())[0].asm['ttgir'])
c = list(gptq_kernel.cache[0].values())[0]
print(dir(c))
print(c.shared)
with open("gptq.ptx", "w") as a:
    print(list(gptq_kernel.cache[0].values())[0].asm['ptx'], file=a)
