from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch
import torch.nn.functional as F


def safe_model_name(model_name: str) -> str:
    return model_name.replace("/", "_").replace("\\", "_").replace(":", "_")


@dataclass
class ClipBackend:
    model_name: str
    device: str = "cuda"
    local_files_only: bool = False

    def __post_init__(self) -> None:
        try:
            from transformers import CLIPModel, CLIPProcessor
        except ImportError as exc:
            raise RuntimeError(
                "NegZeroHOC currently expects Hugging Face transformers CLIP. "
                "Install transformers in the DD environment or add another backend."
            ) from exc

        self.device = self.device if torch.cuda.is_available() or self.device == "cpu" else "cpu"
        self.processor = CLIPProcessor.from_pretrained(
            self.model_name,
            local_files_only=self.local_files_only,
        )
        self.model = CLIPModel.from_pretrained(
            self.model_name,
            local_files_only=self.local_files_only,
        )
        self.model.eval().to(self.device)

    @torch.no_grad()
    def encode_images(self, images) -> torch.Tensor:
        inputs = self.processor(images=list(images), return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(self.device)
        feats = self.model.get_image_features(pixel_values=pixel_values)
        return F.normalize(feats, dim=-1)

    @torch.no_grad()
    def encode_texts(self, prompts: Iterable[str]) -> torch.Tensor:
        prompts = list(prompts)
        inputs = self.processor(
            text=prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        feats = self.model.get_text_features(**inputs)
        return F.normalize(feats, dim=-1)

    @torch.no_grad()
    def encode_prompt_ensemble(self, prompts: Iterable[str]) -> torch.Tensor:
        text_feats = self.encode_texts(prompts)
        return F.normalize(text_feats.mean(dim=0), dim=-1)
