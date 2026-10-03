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


if __name__ == "__main__":
    if sys.argv[1] == "--tok":
        main_tok(sys.argv[2])
    else:
        main(sys.argv[1])
