from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
import torch.nn as nn

from .prompt_text import (
    build_child_negative_text,
    build_edge_text,
    build_parent_text,
    build_parent_unknown_text,
    node_depth,
)


@dataclass
class HierPromptConfig:
    max_length: int = 77
    global_ctx_tokens: int = 4
    depth_ctx_tokens: int = 4
    parent_ctx_tokens: int = 4
    depth_embed_dim: int = 64
    parent_generator_hidden_dim: int = 512
    init_std: float = 0.02
    ablation: str = "global_depth_parent"
    unknown_prompts: int = 1
    unknown_prototype_ctx_tokens: int = 1
    unknown_text_variant: str = "parent_conditioned"
    negative_prompts: int = 2
    negative_prototype_ctx_tokens: int = 2

    @classmethod
    def from_dict(cls, data: dict | None) -> "HierPromptConfig":
        data = dict(data or {})
        valid = {field.name for field in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in data.items() if k in valid})

    def to_dict(self) -> dict:
        return asdict(self)


class ParentContextGenerator(nn.Module):
    def __init__(
        self,
        parent_feature_dim: int,
        depth_embed_dim: int,
        hidden_dim: int,
        out_tokens: int,
        text_width: int,
    ):
        super().__init__()
        self.out_tokens = int(out_tokens)
        self.text_width = int(text_width)
        self.net = nn.Sequential(
            nn.Linear(parent_feature_dim + depth_embed_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_tokens * text_width),
        )

    def forward(self, parent_features: torch.Tensor, depth_embeds: torch.Tensor) -> torch.Tensor:
        out = self.net(torch.cat([parent_features, depth_embeds], dim=-1))
        return out.view(parent_features.shape[0], self.out_tokens, self.text_width)


class _BasePromptLearner(nn.Module):
    def __init__(self, dataset_name: str, hierarchy, text_encoder, cfg: HierPromptConfig, prefix: str):
        super().__init__()
        self.dataset_name = dataset_name
        self.hierarchy = hierarchy
        self.text_encoder = text_encoder
        self.cfg = cfg
        self.prefix = prefix
        self.max_depth = int(hierarchy.max_depth)
        self.text_width = int(text_encoder.text_width)
        self.projection_dim = int(text_encoder.projection_dim)
        self.parent_feature_cache: dict[str, torch.Tensor] = {}

        self.global_ctx = nn.Parameter(
            torch.randn(cfg.global_ctx_tokens, self.text_width) * float(cfg.init_std)
        )
        self.depth_ctx = nn.Parameter(
            torch.randn(self.max_depth + 1, cfg.depth_ctx_tokens, self.text_width) * float(cfg.init_std)
        )
        self.depth_embedding = nn.Embedding(self.max_depth + 1, cfg.depth_embed_dim)
        self.parent_generator = ParentContextGenerator(
            parent_feature_dim=self.projection_dim,
            depth_embed_dim=cfg.depth_embed_dim,
            hidden_dim=cfg.parent_generator_hidden_dim,
            out_tokens=cfg.parent_ctx_tokens,
            text_width=self.text_width,
        )

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def _parent_feature(self, parent: str) -> torch.Tensor:
        if parent not in self.parent_feature_cache:
            text = build_parent_text(self.dataset_name, self.hierarchy, parent)
            with torch.no_grad():
                feat = self.text_encoder.encode_plain_texts([text]).detach().cpu()[0]
            self.parent_feature_cache[parent] = feat
        return self.parent_feature_cache[parent].to(self.device)

    def _context_for_parents(self, parents: list[str]) -> torch.Tensor:
        chunks = []
        batch_size = len(parents)

        if self.cfg.global_ctx_tokens > 0:
            chunks.append(self.global_ctx.unsqueeze(0).expand(batch_size, -1, -1))

        ablation = self.cfg.ablation
        use_depth = ablation in {"global_depth", "global_depth_parent"}
        use_parent = ablation == "global_depth_parent"

        depths = torch.tensor(
            [min(node_depth(self.hierarchy, p), self.max_depth) for p in parents],
            dtype=torch.long,
            device=self.device,
        )
        if use_depth and self.cfg.depth_ctx_tokens > 0:
            chunks.append(self.depth_ctx[depths])

        if use_parent and self.cfg.parent_ctx_tokens > 0:
            parent_features = torch.stack([self._parent_feature(p) for p in parents], dim=0)
            depth_embeds = self.depth_embedding(depths)
            chunks.append(self.parent_generator(parent_features, depth_embeds))

        if not chunks:
            return torch.empty(batch_size, 0, self.text_width, device=self.device)
        return torch.cat(chunks, dim=1)


class PositivePromptLearner(_BasePromptLearner):
    def __init__(self, dataset_name: str, hierarchy, text_encoder, cfg: HierPromptConfig):
        super().__init__(dataset_name, hierarchy, text_encoder, cfg, prefix="positive")
        self.text_variant = "learned"
        self._plain_edge_cache: dict[tuple[tuple[str, str], ...], torch.Tensor] = {}

    def set_text_variant(self, variant: str) -> None:
        if variant not in {"learned", "plain"}:
            raise ValueError(f"Unsupported positive text variant: {variant!r}")
        self.text_variant = variant
        if variant != "plain":
            self._plain_edge_cache.clear()

    def edge_text(self, parent: str, child: str) -> str:
        return build_edge_text(self.dataset_name, self.hierarchy, parent, child)

    def encode_edges(self, parent_child_pairs: list[tuple[str, str]]) -> torch.Tensor:
        if not parent_child_pairs:
            return torch.empty(0, self.projection_dim, device=self.device)
        parents = [p for p, _ in parent_child_pairs]
        texts = [self.edge_text(p, c) for p, c in parent_child_pairs]
        if self.text_variant == "plain":
            cache_key = tuple(parent_child_pairs)
            if cache_key not in self._plain_edge_cache:
                self._plain_edge_cache[cache_key] = (
                    self.text_encoder.encode_plain_texts(texts).detach()
                )
            return self._plain_edge_cache[cache_key]
        context = self._context_for_parents(parents)
        return self.text_encoder.encode_with_context(texts, context)

    def encode_children(self, parent: str, children: list[str] | None = None) -> torch.Tensor:
        children = list(children if children is not None else self.hierarchy.parent2children[parent])
        return self.encode_edges([(parent, child) for child in children])


class UnknownPromptLearner(_BasePromptLearner):
    def __init__(self, dataset_name: str, hierarchy, text_encoder, cfg: HierPromptConfig):
        super().__init__(dataset_name, hierarchy, text_encoder, cfg, prefix="unknown")
        self.num_unknown_prompts = max(1, int(cfg.unknown_prompts))
        self.prototype_ctx_tokens = max(0, int(cfg.unknown_prototype_ctx_tokens))
        if self.num_unknown_prompts > 1 and self.prototype_ctx_tokens > 0:
            self.prototype_ctx = nn.Parameter(
                torch.randn(
                    self.num_unknown_prompts,
                    self.prototype_ctx_tokens,
                    self.text_width,
                )
                * float(cfg.init_std)
            )
        else:
            self.register_parameter("prototype_ctx", None)

    def unknown_text(self, parent: str) -> str:
        return build_parent_unknown_text(
            self.dataset_name,
            self.hierarchy,
            parent,
            variant=self.cfg.unknown_text_variant,
        )

    def encode_unknown_prototypes(self, parents: list[str]) -> torch.Tensor:
        if not parents:
            return torch.empty(
                0,
                self.num_unknown_prompts,
                self.projection_dim,
                device=self.device,
            )

        if self.num_unknown_prompts == 1:
            texts = [self.unknown_text(parent) for parent in parents]
            context = self._context_for_parents(parents)
            features = self.text_encoder.encode_with_context(texts, context)
            return features.unsqueeze(1)

        repeated_parents = [
            parent
            for parent in parents
            for _ in range(self.num_unknown_prompts)
        ]
        texts = [self.unknown_text(parent) for parent in repeated_parents]
        context = self._context_for_parents(repeated_parents)
        if self.prototype_ctx is not None:
            prototype_context = self.prototype_ctx.unsqueeze(0).expand(
                len(parents), -1, -1, -1
            )
            prototype_context = prototype_context.reshape(
                len(repeated_parents),
                self.prototype_ctx_tokens,
                self.text_width,
            )
            context = torch.cat([context, prototype_context], dim=1)
        features = self.text_encoder.encode_with_context(texts, context)
        return features.view(len(parents), self.num_unknown_prompts, -1)

    def encode_unknowns(self, parents: list[str]) -> torch.Tensor:
        features = self.encode_unknown_prototypes(parents)
        if self.num_unknown_prompts != 1:
            raise RuntimeError(
                "encode_unknowns only supports one unknown prompt; "
                "use encode_unknown_prototypes for multiple prototypes"
            )
        return features[:, 0]

    def encode_unknown(self, parent: str) -> torch.Tensor:
        return self.encode_unknowns([parent])[0]


class HierNegativePromptLearner(_BasePromptLearner):
    def __init__(self, dataset_name: str, hierarchy, text_encoder, cfg: HierPromptConfig):
        super().__init__(dataset_name, hierarchy, text_encoder, cfg, prefix="negative")
        self.num_negative_prompts = max(1, int(cfg.negative_prompts))
        self.prototype_ctx_tokens = max(0, int(cfg.negative_prototype_ctx_tokens))
        if self.num_negative_prompts > 1 and self.prototype_ctx_tokens > 0:
            self.prototype_ctx = nn.Parameter(
                torch.randn(
                    self.num_negative_prompts,
                    self.prototype_ctx_tokens,
                    self.text_width,
                )
                * float(cfg.init_std)
            )
        else:
            self.register_parameter("prototype_ctx", None)

    def negative_text(self, parent: str, child: str) -> str:
        return build_child_negative_text(
            self.dataset_name,
            self.hierarchy,
            parent,
            child,
        )

    def encode_negative_prototypes(
        self,
        parent: str,
        children: list[str],
    ) -> torch.Tensor:
        if not children:
            return torch.empty(
                0,
                self.num_negative_prompts,
                self.projection_dim,
                device=self.device,
            )

        repeated_parents = [
            parent
            for _child in children
            for _ in range(self.num_negative_prompts)
        ]
        repeated_children = [
            child
            for child in children
            for _ in range(self.num_negative_prompts)
        ]
        texts = [
            self.negative_text(parent_name, child)
            for parent_name, child in zip(repeated_parents, repeated_children)
        ]
        context = self._context_for_parents(repeated_parents)
        if self.prototype_ctx is not None:
            prototype_context = self.prototype_ctx.unsqueeze(0).expand(
                len(children), -1, -1, -1
            )
            prototype_context = prototype_context.reshape(
                len(repeated_children),
                self.prototype_ctx_tokens,
                self.text_width,
            )
            context = torch.cat([context, prototype_context], dim=1)
        features = self.text_encoder.encode_with_context(texts, context)
        return features.view(len(children), self.num_negative_prompts, -1)
