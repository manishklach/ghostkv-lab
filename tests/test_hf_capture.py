import torch
from transformers import GPT2Config, GPT2LMHeadModel

from ghostkv.hf_capture import capture_qk_tensors, compute_exact_attention_scores, flatten_attention_heads


class DummyTokenizer:
    model_max_length = 32
    eos_token = "<eos>"
    pad_token = "<pad>"

    def __call__(self, text: str, return_tensors: str, truncation: bool, max_length: int) -> dict[str, torch.Tensor]:
        token_ids = [((ord(char) % 31) + 1) for char in text][:max_length]
        if not token_ids:
            token_ids = [1]
        input_ids = torch.tensor([token_ids], dtype=torch.long)
        attention_mask = torch.ones_like(input_ids)
        return {"input_ids": input_ids, "attention_mask": attention_mask}

    def convert_ids_to_tokens(self, ids: list[int]) -> list[str]:
        return [str(token_id) for token_id in ids]


def test_capture_qk_shapes_with_local_gpt2() -> None:
    model = GPT2LMHeadModel(
        GPT2Config(
            vocab_size=64,
            n_positions=32,
            n_ctx=32,
            n_embd=16,
            n_layer=2,
            n_head=2,
        )
    )
    tokenizer = DummyTokenizer()

    capture = capture_qk_tensors(model, tokenizer, "ghostkv test prompt", layer_idx=0)

    assert capture.query_states.shape[0] == 1
    assert capture.key_states.shape[0] == 1
    assert capture.query_states.shape[1] == 2
    assert capture.key_states.shape[1] == 2
    assert capture.query_states.shape[2] == capture.key_states.shape[2]


def test_flatten_and_exact_attention_score_shapes() -> None:
    query_states = torch.randn(1, 2, 5, 4)
    key_states = torch.randn(1, 2, 5, 4)

    query_heads, key_heads = flatten_attention_heads(query_states, key_states)
    scores = compute_exact_attention_scores(query_heads, key_heads)

    assert query_heads.shape == (2, 4)
    assert key_heads.shape == (2, 5, 4)
    assert scores.shape == (2, 5)
