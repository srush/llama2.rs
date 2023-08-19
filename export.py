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
        w = None
        if isinstance(k, torch.Tensor):
            w = k
        elif "GeneralQuantLinear" in str(k.__class__) and EXPAND:
            w = k.build()[0].T
        
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
            print("Regular")
            print(w.shape)
            t = w.contiguous().view(-1).detach().cpu().type(torch.float32).numpy()
            f.write(memoryview(t))

        # del state_dict[key]


    # first write out the header
    p['n_heads'] = model.layers[0].self_attn.num_heads
    hidden_dim = model.layers[0].mlp.up_proj.qweight.shape[1]
    
    #hidden_dim = 11008
    #model.layers[0].mlp.up_proj.build()[0].shape[0]

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


model_name_or_path = "TheBloke/Platypus2-70B-Instruct-GPTQ"
model_basename = "gptq-4bit-32g-actorder_True"


def load_and_export(model_path, output_path):
    tokenizer = AutoTokenizer.from_pretrained("TheBloke/orca_mini_v3_13B-GPTQ")
    prompt = "### System:\nYou are an AI assistant that follows instruction extremely well. Help as much as you can.\n\n### User:\nTell me about Orcas.\n\n### Assistant:\n"
    x = tokenizer.backend_tokenizer.normalizer.normalize_str(prompt)
    #rev = {v: k for k, v in tokenizer.get_vocab().items()}
    #for i in range(32000):
    #    print(i, rev[i])
    #print(dir(tokenizer.backend_tokenizer.normalizer))
    #prompt = "1000"


    #for a, b in zip(prompt, x[1:]):
    #    print(a, b)
    #    print(a == b)
    
    #inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    #print(inputs)
    #exit()
    # model_name_or_path = "TheBloke/Llama-2-70B-chat-GPTQ"
    # model_basename = "main"

    use_triton = False
    model = AutoGPTQForCausalLM.from_quantized(model_name_or_path,
            #model_basename=model_basename,
            use_safetensors=True,
            revision=model_basename,
            inject_fused_attention = False,
            inject_fused_mlp = False,
            trust_remote_code=True,
            device="cpu",
            use_triton=use_triton,                                   
            quantize_config=None,
    )
    # tokens = tokenizer("the guy", return_tensors="pt").to(model.device)
    # print(tokens)
    # print(model)
    # print(model.lm_head)
    # print(model.lm_head.weight.shape)
    # with torch.no_grad():
    #     #out = model.forward(**tokens)
    #     model.float()
    #     model.model.model.embed_tokens.float()
    #     model.model.model.layers[0].input_layernorm.float()
    #     emb = model.model.model.embed_tokens(tokens["input_ids"])
    #     print(emb[0])
    #     print(model.model.model.layers[0].input_layernorm(emb)[0])
    #     #print(model.model.model.embed_tokens(out))
    #     #print(out.keys())
    #     out = model.forward(**tokens)
    #     print(out["past_key_values"][0][0][0, 0, 0])
    #     print(out["past_key_values"][31][0][0, 0, 0])
        
    #     print("logits", out["logits"][0][2])
    export(model, output_path)

if __name__ == '__main__':
    if len(sys.argv) == 1:
        print('[output path]')
        exit()

    output_path = sys.argv[1]
    load_and_export("", output_path)
