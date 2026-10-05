#!/usr/bin/env python3
"""Generate minimal loadable LLAMA GGUFs for ForgeCore testing.

Deterministic (seed 7/8). Default mode writes the decode-path fixture:
tiny dims vocab=32, embd=8, heads=2, kv=2, layers=1, ff=16, ctx=64 —
about 7 KB. `--tok` mode writes the tokenizer fixture instead: same
dims except vocab=269 (`<unk> <s> </s>`, ten `tokN` pieces, full
`<0xXX>` byte coverage so SPM byte fallback never throws) with
add_bos/add_eos both true. Files must be written OUTSIDE the repo
(e.g. $FORGE_LLAMA_DIR/models); see docs/NATIVE.md.

Requires: pip install gguf numpy (with PIP_TARGET outside the repo).

Usage:
    PYTHONPATH="$FORGE_LLAMA_DIR/py" python3 scripts/make-tiny-gguf.py \
        "$FORGE_LLAMA_DIR/models/tiny-llama.gguf"
    PYTHONPATH="$FORGE_LLAMA_DIR/py" python3 scripts/make-tiny-gguf.py --tok \
        "$FORGE_LLAMA_DIR/models/tiny-tok.gguf"
"""
import sys

import numpy as np

import gguf

VOCAB, EMBD, N_HEAD, N_KV, N_LAYER, N_FF, N_CTX = 32, 8, 2, 2, 1, 16, 64

rng = np.random.default_rng(7)


def rand(*shape):
    return (rng.standard_normal(shape) * 0.1).astype(np.float32)


def main(path):
    writer = gguf.GGUFWriter(path, arch="llama", use_temp_file=False)
    writer.add_vocab_size(VOCAB)
    writer.add_context_length(N_CTX)
    writer.add_embedding_length(EMBD)
    writer.add_block_count(N_LAYER)
    writer.add_feed_forward_length(N_FF)
    writer.add_head_count(N_HEAD)
    writer.add_head_count_kv(N_KV)
    writer.add_rope_freq_base(10000.0)
    writer.add_layer_norm_rms_eps(1e-5)

    writer.add_tokenizer_model("llama")
    tokens = [f"tok{i}" for i in range(VOCAB)]
    tokens[0], tokens[1], tokens[2] = "<unk>", "<s>", "</s>"
    writer.add_token_list(tokens)
    writer.add_token_scores([0.0] * VOCAB)
    writer.add_token_types([2, 3, 3] + [1] * (VOCAB - 3))
    writer.add_token_merges(["t o", "k e", "t o k"])
    writer.add_bos_token_id(1)
    writer.add_eos_token_id(2)
    writer.add_unk_token_id(0)

    # numpy shape (d1, d0) -> gguf ne [d0, d1]
    writer.add_tensor("token_embd.weight", rand(VOCAB, EMBD))
    for i in range(N_LAYER):
        prefix = f"blk.{i}."
        writer.add_tensor(prefix + "attn_norm.weight", rand(EMBD))
        writer.add_tensor(prefix + "attn_q.weight", rand(EMBD, EMBD))
        writer.add_tensor(prefix + "attn_k.weight", rand(EMBD, EMBD))
        writer.add_tensor(prefix + "attn_v.weight", rand(EMBD, EMBD))
        writer.add_tensor(prefix + "attn_output.weight", rand(EMBD, EMBD))
        writer.add_tensor(prefix + "ffn_norm.weight", rand(EMBD))
        writer.add_tensor(prefix + "ffn_gate.weight", rand(N_FF, EMBD))
        writer.add_tensor(prefix + "ffn_up.weight", rand(N_FF, EMBD))
        writer.add_tensor(prefix + "ffn_down.weight", rand(EMBD, N_FF))
    writer.add_tensor("output_norm.weight", rand(EMBD))
    writer.add_tensor("output.weight", rand(VOCAB, EMBD))

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    print(f"wrote {path}")


# Tokenizer fixture: <unk> <s> </s> (ids 0-2), tok0..tok9 (ids 3-12),
# <0x00>..<0xFF> byte pieces (ids 13-268). SPM byte fallback looks up
# "<0xXX>" (uppercase hex) then the raw byte; without byte pieces any
# non-empty encode throws std::out_of_range across the FFI boundary.
TOK_VOCAB = 3 + 10 + 256
TOK_WORDS = 10


def main_tok(path):
    local_rng = np.random.default_rng(8)

    def trand(*shape):
        return (local_rng.standard_normal(shape) * 0.1).astype(np.float32)

    writer = gguf.GGUFWriter(path, arch="llama", use_temp_file=False)
    writer.add_vocab_size(TOK_VOCAB)
    writer.add_context_length(N_CTX)
    writer.add_embedding_length(EMBD)
    writer.add_block_count(N_LAYER)
    writer.add_feed_forward_length(N_FF)
    writer.add_head_count(N_HEAD)
    writer.add_head_count_kv(N_KV)
    writer.add_rope_freq_base(10000.0)
    writer.add_layer_norm_rms_eps(1e-5)

    writer.add_tokenizer_model("llama")
    tokens = ["<unk>", "<s>", "</s>"]
    tokens += [f"tok{i}" for i in range(TOK_WORDS)]
    tokens += [f"<0x{b:02X}>" for b in range(256)]
    writer.add_token_list(tokens)
    writer.add_token_scores([0.0] * TOK_VOCAB)
    writer.add_token_types([2, 3, 3] + [1] * TOK_WORDS + [6] * 256)
    writer.add_token_merges(["t o", "k e", "t o k"])
    writer.add_bos_token_id(1)
    writer.add_eos_token_id(2)
    writer.add_unk_token_id(0)
    writer.add_add_bos_token(True)
    writer.add_add_eos_token(True)

    # numpy shape (d1, d0) -> gguf ne [d0, d1]
    writer.add_tensor("token_embd.weight", trand(TOK_VOCAB, EMBD))
    for i in range(N_LAYER):
        prefix = f"blk.{i}."
        writer.add_tensor(prefix + "attn_norm.weight", trand(EMBD))
        writer.add_tensor(prefix + "attn_q.weight", trand(EMBD, EMBD))
        writer.add_tensor(prefix + "attn_k.weight", trand(EMBD, EMBD))
        writer.add_tensor(prefix + "attn_v.weight", trand(EMBD, EMBD))
        writer.add_tensor(prefix + "attn_output.weight", trand(EMBD, EMBD))
        writer.add_tensor(prefix + "ffn_norm.weight", trand(EMBD))
        writer.add_tensor(prefix + "ffn_gate.weight", trand(N_FF, EMBD))
        writer.add_tensor(prefix + "ffn_up.weight", trand(N_FF, EMBD))
        writer.add_tensor(prefix + "ffn_down.weight", trand(EMBD, N_FF))
    writer.add_tensor("output_norm.weight", trand(EMBD))
    writer.add_tensor("output.weight", trand(TOK_VOCAB, EMBD))

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    print(f"wrote {path}")


# DSV4 fixture: minimal loadable DeepSeek-V4 GGUF exercising the
# llama_kv_cache_dsv4 memory class (arch-gated seq < n_seq_max
# invariant). Two layers (ratios 4 and 128: every compressor cache
# owns a layer — a layer-less cache with forced indexer rotation
# divides by zero natively), 2 experts / 1 shared,
# hyper-connection mult 4 (native asserts hc == 4). Deterministic
# (seed 9).
DSV4_VOCAB, DSV4_EMBD, DSV4_HEAD, DSV4_QLORA = 32, 16, 2, 4
DSV4_N_EXP, DSV4_N_FF_EXP, DSV4_N_SHEXP = 2, 8, 1
DSV4_O_GROUPS, DSV4_O_LORA, DSV4_HC = 2, 4, 4  # hc_mult 4: native asserts hc == 4
DSV4_HEAD_K = DSV4_EMBD // DSV4_HEAD  # 8 (loader fallback n_embd/n_head)


def main_dsv4(path):
    local_rng = np.random.default_rng(9)

    def drand(*shape):
        return (local_rng.standard_normal(shape) * 0.1).astype(np.float32)

    arch = "deepseek4"
    writer = gguf.GGUFWriter(path, arch=arch, use_temp_file=False)
    writer.add_vocab_size(DSV4_VOCAB)
    writer.add_context_length(N_CTX)
    writer.add_embedding_length(DSV4_EMBD)
    writer.add_block_count(2)
    writer.add_feed_forward_length(N_FF)
    writer.add_head_count(DSV4_HEAD)
    writer.add_head_count_kv(1)
    writer.add_rope_freq_base(10000.0)
    writer.add_layer_norm_rms_eps(1e-5)
    # DeepSeek-V4 arch keys (all required by the loader).
    writer.add_uint32(f"{arch}.attention.q_lora_rank", DSV4_QLORA)
    writer.add_uint32(f"{arch}.attention.sliding_window", 8)
    writer.add_uint32(f"{arch}.expert_count", DSV4_N_EXP)
    writer.add_uint32(f"{arch}.expert_used_count", 1)
    writer.add_uint32(f"{arch}.expert_feed_forward_length", DSV4_N_FF_EXP)
    writer.add_uint32(f"{arch}.expert_shared_count", DSV4_N_SHEXP)
    writer.add_float32(f"{arch}.expert_weights_scale", 1.0)
    writer.add_bool(f"{arch}.expert_weights_norm", True)
    writer.add_float32(f"{arch}.swiglu_clamp_exp", 7.0)
    writer.add_uint32(f"{arch}.attention.indexer.head_count", 1)
    # 64: differs from head_k (8) so only the indexer cache takes
    # the forced Hadamard path, and 64 rows always divide its nrot 64.
    writer.add_uint32(f"{arch}.attention.indexer.key_length", 64)
    writer.add_uint32(f"{arch}.attention.indexer.top_k", 1)
    writer.add_uint32(f"{arch}.attention.output_group_count", DSV4_O_GROUPS)
    writer.add_uint32(f"{arch}.attention.output_lora_rank", DSV4_O_LORA)
    writer.add_float32(f"{arch}.attention.compress_rope_freq_base", 10000.0)
    writer.add_uint32(f"{arch}.hyper_connection.count", DSV4_HC)
    writer.add_uint32(f"{arch}.hyper_connection.sinkhorn_iterations", 1)
    writer.add_float32(f"{arch}.hyper_connection.epsilon", 1e-5)
    writer.add_uint32(f"{arch}.hash_layer_count", 0)
    writer.add_key_value(
        f"{arch}.attention.compress_ratios", [4, 128],
        gguf.GGUFValueType.ARRAY,
        gguf.GGUFValueType.UINT32)
    writer.add_uint32(f"{arch}.expert_gating_func", 4)  # sqrtsoftplus

    writer.add_tokenizer_model("llama")
    tokens = [f"tok{i}" for i in range(DSV4_VOCAB)]
    tokens[0], tokens[1], tokens[2] = "<unk>", "<s>", "</s>"
    writer.add_token_list(tokens)
    writer.add_token_scores([0.0] * DSV4_VOCAB)
    writer.add_token_types([2, 3, 3] + [1] * (DSV4_VOCAB - 3))
    writer.add_bos_token_id(1)
    writer.add_eos_token_id(2)
    writer.add_unk_token_id(0)

    # numpy shape is the reverse of the gguf ne[] order.
    E, H, Q, K = DSV4_EMBD, DSV4_HEAD, DSV4_QLORA, DSV4_HEAD_K
    writer.add_tensor("token_embd.weight", drand(DSV4_VOCAB, E))
    writer.add_tensor("output_norm.weight", drand(E))
    writer.add_tensor("output.weight", drand(DSV4_VOCAB, E))
    writer.add_tensor("output_hc_fn.weight", drand(DSV4_HC, DSV4_HC * E))
    writer.add_tensor("output_hc_base.weight", drand(DSV4_HC))
    writer.add_tensor("output_hc_scale.weight", drand(1))
    mix = (2 + DSV4_HC) * DSV4_HC
    shexp = DSV4_N_FF_EXP * DSV4_N_SHEXP
    for i, ratio in enumerate([4, 128]):
        p = f"blk.{i}."
        writer.add_tensor(p + "attn_norm.weight", drand(E))
        writer.add_tensor(p + "attn_sinks.weight", drand(H))
        writer.add_tensor(p + "attn_q_a.weight", drand(Q, E))
        writer.add_tensor(p + "attn_q_a_norm.weight", drand(Q))
        writer.add_tensor(p + "attn_q_b.weight", drand(H * K, Q))
        writer.add_tensor(p + "attn_kv.weight", drand(K, E))
        writer.add_tensor(p + "attn_kv_a_norm.weight", drand(K))
        writer.add_tensor(p + "attn_output_a.weight",
                          drand(DSV4_O_GROUPS, DSV4_O_LORA,
                                H * K // DSV4_O_GROUPS))
        writer.add_tensor(p + "attn_output_b.weight",
                          drand(E, DSV4_O_GROUPS * DSV4_O_LORA))
        writer.add_tensor(p + "hc_attn_fn.weight", drand(mix, DSV4_HC * E))
        writer.add_tensor(p + "hc_attn_base.weight", drand(mix))
        writer.add_tensor(p + "hc_attn_scale.weight", drand(3))
        writer.add_tensor(p + "hc_ffn_fn.weight", drand(mix, DSV4_HC * E))
        writer.add_tensor(p + "hc_ffn_base.weight", drand(mix))
        writer.add_tensor(p + "hc_ffn_scale.weight", drand(3))
        # Compressor (coff 2 for ratio 4, else 1); ratio 4 also
        # carries the lightning-indexer tensors.
        coff = 2 if ratio == 4 else 1
        writer.add_tensor(p + "attn_compressor_kv.weight",
                          drand(coff * K, E))
        writer.add_tensor(p + "attn_compressor_gate.weight",
                          drand(coff * K, E))
        writer.add_tensor(p + "attn_compressor_ape.weight",
                          drand(ratio, coff * K))
        writer.add_tensor(p + "attn_compressor_norm.weight", drand(K))
        if ratio == 4:
            writer.add_tensor(p + "indexer.proj.weight", drand(1, E))
            writer.add_tensor(p + "indexer.attn_q_b.weight", drand(64, Q))
            writer.add_tensor(p + "indexer_compressor_kv.weight",
                              drand(128, E))
            writer.add_tensor(p + "indexer_compressor_gate.weight",
                              drand(128, E))
            writer.add_tensor(p + "indexer_compressor_ape.weight",
                              drand(4, 128))
            writer.add_tensor(p + "indexer_compressor_norm.weight",
                              drand(64))
        # MoE (hash_layer_count 0 -> exp_probs_b path).
        writer.add_tensor(p + "ffn_gate_inp.weight", drand(DSV4_N_EXP, E))
        writer.add_tensor(p + "exp_probs_b.bias", drand(DSV4_N_EXP))
        writer.add_tensor(p + "ffn_norm.weight", drand(E))
        writer.add_tensor(p + "ffn_gate_exps.weight",
                          drand(DSV4_N_EXP, DSV4_N_FF_EXP, E))
        writer.add_tensor(p + "ffn_down_exps.weight",
                          drand(DSV4_N_EXP, E, DSV4_N_FF_EXP))
        writer.add_tensor(p + "ffn_up_exps.weight",
                          drand(DSV4_N_EXP, DSV4_N_FF_EXP, E))
        writer.add_tensor(p + "ffn_gate_shexp.weight", drand(shexp, E))
        writer.add_tensor(p + "ffn_down_shexp.weight", drand(E, shexp))
        writer.add_tensor(p + "ffn_up_shexp.weight", drand(shexp, E))

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    print(f"wrote {path}")


if __name__ == "__main__":
    if sys.argv[1] == "--tok":
        main_tok(sys.argv[2])
    elif sys.argv[1] == "--dsv4":
        main_dsv4(sys.argv[2])
    else:
        main(sys.argv[1])
