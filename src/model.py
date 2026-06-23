"""
Model wrappers for greedy layer pruning.

Supports any HuggingFace causal LM whose architecture follows the standard
transformer pattern: model.embed_tokens / model.layers / model.norm / lm_head.
(Llama, Mistral, Qwen, Lucie all follow this pattern.)
"""
import torch
import torch.nn as nn

try:
    from transformers.modeling_attn_mask_utils import _prepare_4d_causal_attention_mask
    _LEGACY_MASK = True
except ImportError:
    _LEGACY_MASK = False


class ModifiedModel(nn.Module):
    """
    Single-GPU wrapper that skips the transformer layers in delete_indices.
    Used for custom (project-specific) greedy-decoding evaluation.
    """

    def __init__(self, original_model, delete_indices=None, device=None):
        super().__init__()
        delete_indices = set(delete_indices or [])

        self.config = original_model.config
        self.device = device or next(original_model.parameters()).device

        self.embed_tokens = original_model.model.embed_tokens
        self.layers = nn.ModuleList([
            layer for i, layer in enumerate(original_model.model.layers)
            if i not in delete_indices
        ])
        self.norm = original_model.model.norm
        self.lm_head = original_model.lm_head
        self.vocab_size = original_model.config.vocab_size
        self.to(self.device)

    def forward(self, input_ids, attention_mask=None, position_ids=None, **kwargs):
        input_ids = input_ids.to(self.device)
        batch_size, seq_len = input_ids.shape

        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)
        else:
            attention_mask = torch.ones((batch_size, seq_len), device=self.device)

        if position_ids is None:
            position_ids = (
                torch.arange(seq_len, device=self.device)
                .unsqueeze(0)
                .expand(batch_size, -1)
            )

        hidden_states = self.embed_tokens(input_ids)

        if _LEGACY_MASK:
            causal_mask = _prepare_4d_causal_attention_mask(
                attention_mask, (batch_size, seq_len), hidden_states, 0
            )
        else:
            causal_mask = attention_mask

        for layer in self.layers:
            layer_out = layer(
                hidden_states=hidden_states,
                attention_mask=causal_mask,
                position_ids=position_ids,
            )
            hidden_states = layer_out[0]

        hidden_states = self.norm(hidden_states)
        logits = self.lm_head(hidden_states)
        return {"logits": logits}


# ─── In-place helpers for lm-eval compatibility ───────────────────────────────

def patch_model_inplace(model, delete_indices=None):
    """
    Remove layers in-place so the model remains a valid HuggingFace object
    (required by lm-evaluation-harness).

    Returns (patched_model, original_layers). Always call restore_model_inplace()
    afterwards to undo the change.
    """
    delete_indices = set(delete_indices or [])
    original_layers = model.model.layers
    kept = nn.ModuleList([
        layer for i, layer in enumerate(original_layers)
        if i not in delete_indices
    ])
    model.model.layers = kept
    model.eval()
    return model, original_layers


def restore_model_inplace(model, original_layers):
    """Undo patch_model_inplace."""
    model.model.layers = original_layers
