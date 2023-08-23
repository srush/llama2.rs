"""
This script exports the AutoGPT-Q Llama 2 weights in llama2rs.bin format.
"""
import pathlib
import click
import struct
import torch
from torch import nn
from auto_gptq import AutoGPTQForCausalLM
from auto_gptq.nn_modules import qlinear
from transformers.models.llama import modeling_llama

def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0) -> tuple[torch.tensor, torch.tensor]:
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)  # type: ignore
    freqs = torch.outer(t, freqs).float()  # type: ignore
    freqs_cos = torch.cos(freqs)  # real part
    freqs_sin = torch.sin(freqs)  # imaginary part
    return freqs_cos, freqs_sin

def export(model_wrapper: nn.Module, path: pathlib.Path):
    """export the model weights in fp32 into .bin file to be read from C"""
    f = open(path, 'wb')

    print(model_wrapper.model)
    model = model_wrapper.model.model

    Serializable = torch.Tensor | qlinear.GeneralQuantLinear | modeling_llama.LlamaRMSNorm | nn.modules.linear.Linear
    def serialize(k: Serializable):
        match k:
            case torch.Tensor() | modeling_llama.LlamaRMSNorm() | nn.modules.linear.Linear():
                if isinstance(k, torch.Tensor):
                    w = k
                else:
                    w = k.weight

                print("regular")
                print(w.shape)
                t = w.contiguous().view(-1).detach().cpu().type(torch.float32).numpy()
                f.write(memoryview(t))
            case qlinear.GeneralQuantLinear():
                # more complex case
                print("quantized")
                offset = torch.tensor([0, 4, 8, 12, 16, 20, 24, 28])
                def rearrange(k):
                    order = k.g_idx.cpu().argsort(stable=True)
                    extract = (k.qweight.cpu()[:, None, :] >> offset[:, None]) & (2**4-1)
                    extract = extract.view(k.g_idx.shape[0], -1)[order]
                    store = extract << offset.repeat(1, extract.shape[0] // 8)[..., None]
                    store = store.view(k.qweight.shape[0], 8, k.qweight.shape[1])
                    final = torch.zeros(*k.qweight.shape, dtype=int)
                    for i in range(8):
                        final = final | store[:, i]
                    return final
                for w in [rearrange(k).type(torch.int32), k.qzeros.type(torch.int32), k.scales.type(torch.float32), k.g_idx.argsort(stable=True).type(torch.int32)]:
                    print(w.shape)
                    t = w.T.contiguous().view(-1).detach().cpu().numpy()
                    f.write(memoryview(t))

    # first write out the header
    p = {}
    p['dim'] = model.layers[0].mlp.up_proj.g_idx.shape[0]
    p['n_layers'] = len(model.layers)
    p['n_heads'] = model.layers[0].self_attn.num_heads
    p['hidden_dim'] = model.layers[0].mlp.up_proj.qweight.shape[1]
    p['vocab_size'] = model.embed_tokens.num_embeddings
    p['max_seq_len'] = 2048

    n_kv_heads = p.get('n_kv_heads') or p['n_heads']
    header = struct.pack(
        'iiiiiii',
        p['dim'], p['hidden_dim'], p['n_layers'], p['n_heads'],
        n_kv_heads, -p['vocab_size'], p['max_seq_len']
    )
    # NOTE ABOVE: -ve vocab_size is indicating that the classifier weights are present
    # in the checkpoint and should be loaded.
    f.write(header)

    # next write out the embedding weights
    print("writing tok_embeddings...")
    f.write(memoryview(torch.tensor([model_wrapper.config.rms_norm_eps]).numpy()))
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
    serialize(model_wrapper.model.lm_head)

    f.close()
    print(f"wrote {path}")

@click.command()
@click.argument("output-path", type=click.Path(exists=False, path_type=pathlib.Path))
@click.argument("model-name", type=str)
@click.argument("revision", type=str)
def main(output_path: pathlib.Path, model_name: str, revision: str):
    print(f"Loading model {model_name} / {revision} ...")
    model = AutoGPTQForCausalLM.from_quantized(
        model_name,
        revision=revision,
        model_basename="model",
        use_safetensors=True,
        trust_remote_code=False,
        device="cpu",
        inject_fused_attention = False,
        inject_fused_mlp = False,
        use_triton=False,
        quantize_config=None,
    )
    print("Exporting...")
    export(model, output_path)

if __name__ == '__main__':
    main()