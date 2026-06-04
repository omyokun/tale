import inspect
from typing import Dict, Sequence

import torch
import torch.nn as nn
from transformers.modeling_attn_mask_utils import _prepare_4d_causal_attention_mask


class ModifiedDecoderModel(nn.Module):
    """Drop selected decoder layers while preserving the base model components."""

    def __init__(
        self,
        original_model: nn.Module,
        delete_indices: Sequence[int],
        device: torch.device,
    ):
        super().__init__()
        model_root = get_model_root(original_model)
        delete_set = set(delete_indices)

        self.config = original_model.config
        self.embed_tokens = model_root.embed_tokens
        self.layers = nn.ModuleList(
            [
                layer
                for idx, layer in enumerate(model_root.layers)
                if idx not in delete_set
            ]
        )
        self.norm = model_root.norm
        self.lm_head = original_model.lm_head
        self.vocab_size = original_model.config.vocab_size
        self.rotary_emb = getattr(model_root, "rotary_emb", None)
        self._layer_param_names = [
            set(inspect.signature(layer.forward).parameters.keys())
            for layer in self.layers
        ]
        self._device = device
        self.to(device)

    def _compute_position_embeddings(
        self, hidden_states: torch.Tensor, position_ids: torch.Tensor
    ):
        if self.rotary_emb is None:
            return None

        for kwargs in (
            {},
            {"position_ids": position_ids},
            {"seq_len": position_ids.shape[-1]},
        ):
            try:
                if kwargs:
                    return self.rotary_emb(hidden_states, **kwargs)
                return self.rotary_emb(hidden_states, position_ids)
            except TypeError:
                continue

        return None

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        position_embeddings=None,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        del kwargs
        input_ids = input_ids.to(self._device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(self._device)

        batch_size, seq_len = input_ids.shape
        hidden_states = self.embed_tokens(input_ids)

        if position_ids is None:
            position_ids = (
                torch.arange(seq_len, device=input_ids.device)
                .unsqueeze(0)
                .expand(batch_size, -1)
            )

        if attention_mask is None:
            attention_mask = torch.ones((batch_size, seq_len), device=self._device)

        if position_embeddings is None:
            position_embeddings = self._compute_position_embeddings(
                hidden_states, position_ids
            )

        causal_attention_mask = _prepare_4d_causal_attention_mask(
            attention_mask,
            (batch_size, seq_len),
            hidden_states,
            0,
        )

        cache_position = torch.arange(seq_len, device=input_ids.device)

        for layer, param_names in zip(self.layers, self._layer_param_names):
            layer_kwargs = {"hidden_states": hidden_states}

            if "attention_mask" in param_names:
                layer_kwargs["attention_mask"] = causal_attention_mask
            if "position_ids" in param_names:
                layer_kwargs["position_ids"] = position_ids
            if "position_embeddings" in param_names:
                layer_position_embeddings = position_embeddings
                if layer_position_embeddings is None:
                    layer_rotary = getattr(
                        getattr(layer, "self_attn", None), "rotary_emb", None
                    )
                    if layer_rotary is not None:
                        try:
                            layer_position_embeddings = layer_rotary(
                                hidden_states, position_ids
                            )
                        except TypeError:
                            try:
                                layer_position_embeddings = layer_rotary(
                                    hidden_states, position_ids=position_ids
                                )
                            except TypeError:
                                layer_position_embeddings = None

                if layer_position_embeddings is None:
                    raise RuntimeError(
                        "A decoder layer requires position_embeddings, but rotary embeddings could not be computed."
                    )
                layer_kwargs["position_embeddings"] = layer_position_embeddings
            if "cache_position" in param_names:
                layer_kwargs["cache_position"] = cache_position
            if "past_key_value" in param_names:
                layer_kwargs["past_key_value"] = None
            if "use_cache" in param_names:
                layer_kwargs["use_cache"] = False
            if "output_attentions" in param_names:
                layer_kwargs["output_attentions"] = False

            layer_outputs = layer(**layer_kwargs)
            if isinstance(layer_outputs, tuple):
                hidden_states = layer_outputs[0]
            elif hasattr(layer_outputs, "hidden_states"):
                hidden_states = layer_outputs.hidden_states
            else:
                hidden_states = layer_outputs

        hidden_states = self.norm(hidden_states)
        logits = self.lm_head(hidden_states)
        return {"logits": logits}


def get_model_root(model: nn.Module) -> nn.Module:
    model_root = getattr(model, "model", None)
    if model_root is None or not hasattr(model_root, "layers"):
        raise ValueError("Expected an AutoModelForCausalLM-style model with .model.layers.")
    return model_root


def get_model_dtype(device: torch.device) -> torch.dtype:
    if device.type != "cuda":
        return torch.float32
    if torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


def maybe_sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)

