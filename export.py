"""
This script exports the Llama 2 weights in llama2c.bin format.
"""
import os
import sys
import struct
from pathlib import Path
import json

import torch

# from model import precompute_freqs_cis
def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)  # type: ignore
    freqs = torch.outer(t, freqs).float()  # type: ignore
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis


def export(model2, filepath='model.bin'):
    """export the model weights in fp32 into .bin file to be read from C"""
    f = open(filepath, 'wb')
    p = {}
    EXPAND = False
    model = model2.model.model
    p['n_layers'] = len(model.layers)
    print(model)
    def serialize(k):
        # print(f"writing {key}...")
        w = None
        if isinstance(k, torch.Tensor):
            w = k
        elif "GeneralQuantLinear" in str(k.__class__) and EXPAND:
            w = k.build()[0].T
        
        elif "GeneralQuantLinear" not in str(k.__class__):
            w = k.weight

        if w is None:
            for w in [k.qweight.type(torch.int32), k.qzeros.type(torch.int32), k.scales.type(torch.float32)]:
                print(w.shape)
                t = w.T.contiguous().view(-1).detach().cpu().numpy()
                f.write(memoryview(t))
        else:
            print("regular")
            print(w.shape)
            t = w.contiguous().view(-1).detach().cpu().type(torch.float32).numpy()
            f.write(memoryview(t))

        # del state_dict[key]


    # first write out the header
    p['n_heads'] = model.layers[0].self_attn.num_heads
    hidden_dim = model.layers[0].mlp.up_proj.build()[0].shape[1]
    p['dim'] = model.layers[0].mlp.up_proj.build()[0].shape[0]

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
    freqs_cis = precompute_freqs_cis(p['dim'] // p['n_heads'], p['max_seq_len'] * 2)
    serialize(freqs_cis.real[:p['max_seq_len']])
    serialize(freqs_cis.imag[:p['max_seq_len']])

    # finally write the output weights
    serialize(model2.model.lm_head)

    f.close()
    print(f"wrote {filepath}")


# def concat_weights(models):
#     state_dict = {}
#     for name in list(models[0]):
#         tensors = [model[name] for model in models]
#         if len(tensors) == 1 or len(tensors[0].shape) == 1:
#             state_dict[name] = tensors[0]
#             continue
#         is_axis_1 = (
#             name.startswith('tok_embeddings.')
#             or name.endswith('.attention.wo.weight')
#             or name.endswith('.feed_forward.w2.weight')
#         )
#         axis = 1 if is_axis_1 else 0
#         state_dict[name] = torch.cat(tensors, dim=axis)
#         for model in models:
#             del model[name]
#     return state_dict


def load_and_export(model_path, output_path):
    from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
    model_name_or_path = "TheBloke/Llama-2-70B-chat-GPTQ"
    model_basename = "gptq_model-4bit-128g"
    # model_name_or_path = "TheBloke/Llama-2-70B-chat-GPTQ"
    # model_basename = "main"

    use_triton = False


    model = AutoGPTQForCausalLM.from_quantized(model_name_or_path,
            model_basename=model_basename,
            use_safetensors=True,
            revision="main",
            inject_fused_attention = False,
            inject_fused_mlp = False,
            trust_remote_code=True,
            device="cpu",
            use_triton=use_triton,
                                               
                                               quantize_config=None,
                    
    )

    # params_path = os.path.join(model_path, 'params.json')
    # with open(params_path) as f:
    #     params = json.load(f)
    #     print(params)

    # model_paths = sorted(list(Path(model_path).glob('consolidated.*.pth')))
    # models = [torch.load(p, map_location='cpu') for p in model_paths]
    # state_dict = concat_weights(models)
    # del models
    export(model, output_path)


if __name__ == '__main__':
    if len(sys.argv) == 1:
        print('[Llama model folder path] [output path]')
        exit()

    model_path = sys.argv[1]
    output_path = sys.argv[2]
    load_and_export(model_path, output_path)
