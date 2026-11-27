from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class TransferableNegativePromptLearner(nn.Module):
    """NegPrompt-style contexts initialized from a frozen positive learner.

    The class-name text is exactly the positive edge text.  Each negative
    prompt is represented by one class-agnostic context offset shared across
    every hierarchy edge, matching NegPrompt's transferable-template setup.
    """

    def __init__(
        self,
        positive_learner,
        *,
        num_negative_prompts: int = 2,
        init_noise: float = 1e-2,
    ):
        super().__init__()
        self.positive_learner = positive_learner
        self.num_negative_prompts = max(1, int(num_negative_prompts))
        with torch.no_grad():
            context = positive_learner._context_for_parents(["root"])
        if context.ndim != 3 or int(context.shape[1]) == 0:
            raise ValueError(
                "Transferable negative prompts require at least one positive "
                "context token"
            )
        self.context_offsets = nn.Parameter(
            torch.empty(
                self.num_negative_prompts,
                int(context.shape[1]),
                int(context.shape[2]),
                device=context.device,
                dtype=context.dtype,
            )
        )
        nn.init.uniform_(
            self.context_offsets,
            -float(init_noise),
            float(init_noise),
        )

    def trainable_parameters(self) -> list[nn.Parameter]:
        return [self.context_offsets]

    def encode_edges(
        self,
        parent_child_pairs: list[tuple[str, str]],
    ) -> torch.Tensor:
        """Return normalized features with shape ``[edges, prompts, dim]``."""
        if not parent_child_pairs:
            return torch.empty(
                0,
                self.num_negative_prompts,
                self.positive_learner.projection_dim,
                device=self.context_offsets.device,
            )
        parents = [parent for parent, _ in parent_child_pairs]
        texts = [
            self.positive_learner.edge_text(parent, child)
            for parent, child in parent_child_pairs
        ]
        with torch.no_grad():
            positive_context = self.positive_learner._context_for_parents(
                parents
            ).detach()
        contexts = (
            positive_context.unsqueeze(1)
            + self.context_offsets.unsqueeze(0)
        )
        repeated_texts = [
            text
            for text in texts
            for _ in range(self.num_negative_prompts)
        ]
        flat_contexts = contexts.reshape(
            len(parent_child_pairs) * self.num_negative_prompts,
            contexts.shape[2],
            contexts.shape[3],
        )
        features = self.positive_learner.text_encoder.encode_with_context(
            repeated_texts,
            flat_contexts,
        )
        return features.view(
            len(parent_child_pairs),
            self.num_negative_prompts,
            -1,
        )


def negprompt_loss(
    image_features: torch.Tensor,
    positive_features: torch.Tensor,
    negative_features: torch.Tensor,
    *,
    logit_scale: float,
    beta: float,
    gamma: float,
    distance_mode: str = "attractive",
) -> tuple[torch.Tensor, dict[str, float]]:
    """The three losses from NegPrompt, plus a sign-controlled NPD ablation.

    ``attractive`` is the published NPD loss, ``-cos(negative, positive)``.
    ``repulsive`` flips only that term to directly test the hypothesis that
    negative prompts should move away from positive prompt features.
    """
    if distance_mode not in {"attractive", "repulsive"}:
        raise ValueError(
            "distance_mode must be either 'attractive' or 'repulsive'"
        )
    if float(logit_scale) <= 0.0:
        raise ValueError("logit_scale must be positive")
    if positive_features.ndim != 2:
        raise ValueError("positive_features must have shape [classes, dim]")
    if negative_features.ndim != 3:
        raise ValueError(
            "negative_features must have shape [classes, prompts, dim]"
        )
    if negative_features.shape[0] != positive_features.shape[0]:
        raise ValueError("Positive and negative class counts must match")

    images = F.normalize(image_features.float(), dim=-1)
    positives = F.normalize(positive_features.float(), dim=-1)
    negatives = F.normalize(negative_features.float(), dim=-1)
    negative_flat = negatives.flatten(0, 1)

    negative_logits = float(logit_scale) * (images @ negative_flat.t())
    # H(uniform, Softmax(S_f,neg)) in Eq. (4).
    nis_loss = -F.log_softmax(negative_logits, dim=1).mean()
    nis_excess = nis_loss - torch.log(
        nis_loss.new_tensor(float(negative_logits.shape[1]))
    )

    paired_cosines = (
        positives.unsqueeze(1).expand_as(negatives) * negatives
    ).sum(dim=-1)
    mean_positive_negative_cosine = paired_cosines.mean()
    if distance_mode == "attractive":
        npd_loss = -mean_positive_negative_cosine
    else:
        npd_loss = mean_positive_negative_cosine

    prompt_count = int(negatives.shape[1])
    if prompt_count > 1:
        pairwise = negatives @ negatives.transpose(1, 2)
        mask = ~torch.eye(
            prompt_count,
            dtype=torch.bool,
            device=pairwise.device,
        )
        nnd_loss = pairwise[:, mask].mean()
    else:
        nnd_loss = negatives.new_zeros(())

    total = nis_loss + float(beta) * npd_loss + float(gamma) * nnd_loss
    return total, {
        "loss": float(total.detach().cpu()),
        "nis_loss": float(nis_loss.detach().cpu()),
        "nis_excess": float(nis_excess.detach().cpu()),
        "npd_loss": float(npd_loss.detach().cpu()),
        "nnd_loss": float(nnd_loss.detach().cpu()),
        "positive_negative_cosine": float(
            mean_positive_negative_cosine.detach().cpu()
        ),
    }


def negprompt_probabilities(
    image_features: torch.Tensor,
    positive_features: torch.Tensor,
    negative_features: torch.Tensor,
    *,
    temperature: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Joint positive/negative softmax used by NegPrompt's MCM score."""
    if float(temperature) <= 0.0:
        raise ValueError("temperature must be positive")
    if positive_features.ndim != 2 or negative_features.ndim != 3:
        raise ValueError("Expected positive [C,D] and negative [C,K,D]")
    if negative_features.shape[0] != positive_features.shape[0]:
        raise ValueError("Positive and negative class counts must match")

    images = F.normalize(image_features.float(), dim=-1)
    positives = F.normalize(positive_features.float(), dim=-1)
    negatives = F.normalize(negative_features.float(), dim=-1)
    positive_logits = images @ positives.t() / float(temperature)
    negative_logits = torch.einsum("bd,ckd->bck", images, negatives)
    negative_logits = negative_logits / float(temperature)
    all_logits = torch.cat(
        [positive_logits, negative_logits.flatten(1)],
        dim=1,
    )
    all_probabilities = F.softmax(all_logits, dim=1)
    positive_probabilities = all_probabilities[:, : positives.shape[0]]
    negative_probabilities = all_probabilities[:, positives.shape[0] :]
    return positive_probabilities, negative_probabilities.view_as(
        negative_logits
    )


def negprompt_mcm_confidence(
    image_features: torch.Tensor,
    positive_features: torch.Tensor,
    negative_features: torch.Tensor,
    *,
    temperature: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return max positive probability and its class index."""
    positive_probabilities, _ = negprompt_probabilities(
        image_features,
        positive_features,
        negative_features,
        temperature=temperature,
    )
    return positive_probabilities.max(dim=1)
