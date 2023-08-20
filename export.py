"""
This script exports the AutoGPT-Q Llama 2 weights in llama2rs.bin format.
"""
import os
import sys
import struct
from pathlib import Path
import json
import torch
from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
from transformers import AutoTokenizer


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)  # type: ignore
    freqs = torch.outer(t, freqs).float()  # type: ignore
    freqs_cos = torch.cos(freqs)  # real part
    freqs_sin = torch.sin(freqs)  # imaginary part
    return freqs_cos, freqs_sin


def export(model2, filepath='model.bin'):
    """export the model weights in fp32 into .bin file to be read from C"""
    f = open(filepath, 'wb')
    p = {}

    EXPAND = False
    model = model2.model.model
    p['dim'] = model.layers[0].mlp.up_proj.g_idx.shape[0]
    p['n_layers'] = len(model.layers)
    print(model2.model)
    def serialize(k):
        # w = None
        # if isinstance(k, torch.Tensor):
        #     w = k       
        # elif "GeneralQuantLinear" in str(k.__class__) and EXPAND:
        #     w = k.build()[0].T
        
        elif "GeneralQuantLinear" not in str(k.__class__):
            w = k.weight
        offset = torch.tensor([0, 4, 8, 12, 16, 20, 24, 28])
        if w is None:
            def rearrange(k):
                order = k.g_idx.cpu().argsort(stable=True)
                extract = (k.qweight.cpu()[:, None, :] >> offset[:, None]) & (2**4-1)
                extract = extract.view(k.g_idx.shape[0], -1)[order]
                store = extract << offset.repeat(1, extract.shape[0] // 8)[..., None]
                store = store.view(k.qweight.shape[0], 8, k.qweight.shape[1])
                final = torch.zeros(*k.qweight.shape, dtype=int)
                for i in range(8):
                    final = final | store[:, i]
                # print(final.shape, k.qweight.shape)
                # print(order)
                # print(final)
                # print(k.qweight)

                return final
            for w in [rearrange(k).type(torch.int32), k.qzeros.type(torch.int32), k.scales.type(torch.float32), k.g_idx.argsort(stable=True).type(torch.int32)]:
                print("Quant")
                print(w.shape)
                t = w.T.contiguous().view(-1).detach().cpu().numpy()
                f.write(memoryview(t))
        else:
            if hasattr(k, "weight"):
                w = k.weight
            else:
                w = k
            print("Regular")
            print(w.shape)
            t = w.contiguous().view(-1).detach().cpu().type(torch.float32).numpy()
            f.write(memoryview(t))

        # del state_dict[key]


    # first write out the header
    p['n_heads'] = model.layers[0].self_attn.num_heads
    hidden_dim = model.layers[0].mlp.up_proj.qweight.shape[1]
    
    p['vocab_size'] = 32000
    p['max_seq_len'] = 2048

    n_kv_heads = p.get('n_kv_heads') or p['n_heads']
    header = struct.pack(
        'iiiiiii',
        p['dim'], hidden_dim, p['n_layers'], p['n_heads'],
        n_kv_heads, -p['vocab_size'], p['max_seq_len']
    )
    # NOTE ABOVE: -ve vocab_size is indicating that the classifier weights are present
    # in the checkpoint and should be loaded.
    f.write(header)

    # next write out the embedding weights
    print("writing tok_embeddings...")
    serialize(model.embed_tokens)

    # now all the layers
    # attention weights
    for i in range(p['n_layers']): serialize(model.layers[i].input_layernorm)
    for i in range(p['n_layers']): serialize(model.layers[i].self_attn.q_proj)
    for i in range(p['n_layers']): serialize(model.layers[i].self_attn.k_proj)
    for i in range(p['n_layers']): serialize(model.layers[i].self_attn.v_proj)
    for i in range(p['n_layers']): serialize(model.layers[i].self_attn.o_proj)
    # ffn weights
    for i in range(p['n_layers']): serialize(model.layers[i].post_attention_layernorm)
    for i in range(p['n_layers']): serialize(model.layers[i].mlp.gate_proj)
    for i in range(p['n_layers']): serialize(model.layers[i].mlp.down_proj)
    for i in range(p['n_layers']): serialize(model.layers[i].mlp.up_proj)
    
    # final rmsnorm

    serialize(model.norm)
    # freqs_cis
    freqs_cos, freqs_sin = precompute_freqs_cis(p['dim'] // p['n_heads'], p['max_seq_len'] * 2)
    serialize(freqs_cos[:p['max_seq_len']])
    serialize(freqs_sin[:p['max_seq_len']])

    # finally write the output weights
    serialize(model2.model.lm_head)

    f.close()
    print(f"wrote {filepath}")

def load_and_export(model_name, revision, output_path):
    use_triton = False
    model = AutoGPTQForCausalLM.from_quantized(model_name,
            #model_basename=model_basename,
            use_safetensors=True,
            revision=revision,
            inject_fused_attention = False,
            inject_fused_mlp = False,
            trust_remote_code=True,
            device="cpu",
            use_triton=use_triton,                                   
            quantize_config=None,
    )
    export(model, output_path)

if __name__ == '__main__':
    if len(sys.argv) != 4:
        print('[output path]')
        exit()

    output_path = sys.argv[1]
    model_name = sys.argv[2]
    revision = sys.argv[3]
    load_and_export(model_name, revision, output_path) 
