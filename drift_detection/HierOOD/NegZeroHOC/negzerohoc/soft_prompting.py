from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class SoftPromptBatch:
    inputs_embeds: torch.Tensor
    attention_mask: torch.Tensor
    eos_positions: torch.Tensor


class SoftPromptTextEncoder(nn.Module):
    """Frozen CLIP text encoder with trainable continuous context insertion."""

    def __init__(self, clip_model, tokenizer, max_length: int = 77):
        super().__init__()
        self.clip_model = clip_model
        self.tokenizer = tokenizer
        self.max_length = int(max_length)

        for param in self.clip_model.text_model.parameters():
            param.requires_grad_(False)
        if hasattr(self.clip_model, "text_projection"):
            if isinstance(self.clip_model.text_projection, nn.Module):
                self.clip_model.text_projection.requires_grad_(False)
            elif isinstance(self.clip_model.text_projection, nn.Parameter):
                self.clip_model.text_projection.requires_grad_(False)

        self.text_model = self.clip_model.text_model
        self.token_embedding = self.text_model.embeddings.token_embedding
        self.position_embedding = self.text_model.embeddings.position_embedding
        self.text_width = int(self.token_embedding.embedding_dim)
        if isinstance(self.clip_model.text_projection, nn.Linear):
            self.projection_dim = int(self.clip_model.text_projection.out_features)
        else:
            self.projection_dim = int(self.clip_model.text_projection.shape[1])

        self.pad_token_id = int(getattr(tokenizer, "pad_token_id", 0) or 0)
        self.eos_token_id = int(getattr(tokenizer, "eos_token_id", 49407))

    @property
    def device(self) -> torch.device:
        return next(self.clip_model.parameters()).device

    def tokenize(self, texts: list[str], max_length: int | None = None) -> dict[str, torch.Tensor]:
        max_length = int(max_length or self.max_length)
        encoded = self.tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        return {k: v.to(self.device) for k, v in encoded.items()}

    @torch.no_grad()
    def encode_plain_texts(self, texts: list[str]) -> torch.Tensor:
        inputs = self.tokenize(texts, self.max_length)
        feats = self.clip_model.get_text_features(**inputs)
        return F.normalize(feats, dim=-1)

    def _build_soft_prompt_batch(self, texts: list[str], context_embeds: torch.Tensor) -> SoftPromptBatch:
        if context_embeds.dim() != 3:
            raise ValueError("context_embeds must have shape [B, M, C]")
        batch_size, context_len, width = context_embeds.shape
        if width != self.text_width:
            raise ValueError(f"context width {width} does not match CLIP text width {self.text_width}")
        if len(texts) != batch_size:
            raise ValueError(f"Expected {batch_size} texts, got {len(texts)}")
        if context_len >= self.max_length - 2:
            raise ValueError("Too many context tokens for CLIP max text length")

        fixed_length = self.max_length - context_len
        tokenized = self.tokenize(texts, fixed_length)
        input_ids = tokenized["input_ids"]
        attention_mask = tokenized["attention_mask"]

        sos_ids = input_ids[:, :1]
        tail_ids = input_ids[:, 1:]
        tail_mask = attention_mask[:, 1:]

        sos_embeds = self.token_embedding(sos_ids)
        tail_embeds = self.token_embedding(tail_ids)
        inputs_embeds = torch.cat([sos_embeds, context_embeds, tail_embeds], dim=1)

        context_mask = torch.ones(batch_size, context_len, dtype=tail_mask.dtype, device=self.device)
        merged_attention = torch.cat([attention_mask[:, :1], context_mask, tail_mask], dim=1)

        eos_positions = []
        for row in tail_ids:
            eos_locs = (row == self.eos_token_id).nonzero(as_tuple=False)
            if eos_locs.numel() > 0:
                eos_pos = 1 + context_len + int(eos_locs[0].item())
            else:
                non_pad = (row != self.pad_token_id).nonzero(as_tuple=False)
                eos_pos = 1 + context_len + int(non_pad[-1].item()) if non_pad.numel() else 0
            eos_positions.append(eos_pos)
        eos_positions = torch.tensor(eos_positions, dtype=torch.long, device=self.device)

        return SoftPromptBatch(inputs_embeds, merged_attention, eos_positions)

    def _encode_inputs_embeds(self, batch: SoftPromptBatch) -> torch.Tensor:
        try:
            outputs = self.text_model(
                inputs_embeds=batch.inputs_embeds,
                attention_mask=batch.attention_mask,
                return_dict=True,
            )
            last_hidden = outputs.last_hidden_state
        except TypeError:
            last_hidden = self._encode_inputs_embeds_manual(batch.inputs_embeds)

        pooled = last_hidden[torch.arange(last_hidden.shape[0], device=last_hidden.device), batch.eos_positions]
        if isinstance(self.clip_model.text_projection, nn.Linear):
            projected = self.clip_model.text_projection(pooled)
        else:
            projected = pooled @ self.clip_model.text_projection
        return F.normalize(projected, dim=-1)

    def _encode_inputs_embeds_manual(self, inputs_embeds: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = inputs_embeds.shape
        position_ids = torch.arange(seq_len, dtype=torch.long, device=inputs_embeds.device).unsqueeze(0)
        hidden_states = inputs_embeds + self.position_embedding(position_ids)

        causal_attention_mask = None
        try:
            from transformers.modeling_attn_mask_utils import _create_4d_causal_attention_mask

            causal_attention_mask = _create_4d_causal_attention_mask(
                (batch_size, seq_len),
                hidden_states.dtype,
                device=hidden_states.device,
            )
        except Exception:
            causal_attention_mask = None

        encoder_outputs = self.text_model.encoder(
            inputs_embeds=hidden_states,
            causal_attention_mask=causal_attention_mask,
            output_attentions=False,
            output_hidden_states=False,
        )
        return self.text_model.final_layer_norm(encoder_outputs.last_hidden_state)

    def encode_with_context(self, texts: list[str], context_embeds: torch.Tensor) -> torch.Tensor:
        batch = self._build_soft_prompt_batch(texts, context_embeds.to(self.device))
        return self._encode_inputs_embeds(batch)


def soft_prompt_smoke_test(clip_backend, max_length: int = 77, context_tokens: int = 4) -> dict:
    encoder = SoftPromptTextEncoder(
        clip_backend.model,
        clip_backend.processor.tokenizer,
        max_length=max_length,
    )
    context = nn.Parameter(torch.randn(2, context_tokens, encoder.text_width, device=encoder.device) * 0.02)
    feats = encoder.encode_with_context(["a photo of an aircraft", "a photo of a bird"], context)
    loss = feats.square().sum()
    loss.backward()
    grad_norm = float(context.grad.norm().detach().cpu())
    return {
        "ok": grad_norm > 0,
        "grad_norm": grad_norm,
        "text_width": encoder.text_width,
        "projection_dim": encoder.projection_dim,
    }
