"""Losses for contrastive alignment, masked distillation and DTCA guidance."""

from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn.functional as F


def label_aware_clip_loss(emb_a, emb_b, labels_a, labels_b, logit_scale):
    """Soft-target CLIP loss in which any two same-class samples are positives.

    Unlike diagonal-only InfoNCE it stays well defined with repeated labels in
    small, imbalanced batches.
    """
    emb_a = F.normalize(emb_a, dim=1)
    emb_b = F.normalize(emb_b, dim=1)
    logits = logit_scale * (emb_a @ emb_b.T)
    mask = torch.eq(labels_a.view(-1, 1), labels_b.view(1, -1)).float()
    mask = mask / (mask.sum(dim=1, keepdim=True) + 1e-8)
    log_probs = F.log_softmax(logits, dim=1)
    return -(mask * log_probs).sum(dim=1).mean()


def vanilla_clip_loss(emb_a, emb_b, logit_scale):
    """Symmetric InfoNCE with diagonal positives only."""
    emb_a = F.normalize(emb_a, dim=1)
    emb_b = F.normalize(emb_b, dim=1)
    logits = logit_scale * (emb_a @ emb_b.T)
    targets = torch.arange(len(logits), device=emb_a.device)
    return 0.5 * (F.cross_entropy(logits, targets) + F.cross_entropy(logits.T, targets))


def masked_distillation_loss(projected_audio, target, mask, kind: str = "cosine"):
    """Distillation onto teacher targets, averaged over samples where ``mask`` is 1.

    The mask is 0 for samples with no privileged counterpart (``Other``), so the
    3-class task can borrow from a binary teacher. ``kind`` is one of cosine, mse,
    smooth_l1, cosine_mse, or relational (pairwise-distance RKD, Park et al. 2019).
    """
    m = mask.float()
    denom = m.sum()
    if denom <= 0:
        return projected_audio.sum() * 0.0
    if kind == "cosine":
        ones = torch.ones(projected_audio.size(0), device=projected_audio.device)
        raw = F.cosine_embedding_loss(projected_audio, target, ones, reduction="none")
        return (raw * m).sum() / denom
    if kind == "mse":
        return (((projected_audio - target) ** 2).mean(dim=1) * m).sum() / denom
    if kind == "smooth_l1":
        raw = F.smooth_l1_loss(projected_audio, target, reduction="none").mean(dim=1)
        return (raw * m).sum() / denom
    if kind == "cosine_mse":
        return (masked_distillation_loss(projected_audio, target, mask, "cosine")
                + masked_distillation_loss(projected_audio, target, mask, "mse"))
    if kind == "relational":
        return _masked_relational_distillation(projected_audio, target, m)
    raise ValueError(f"unknown distillation loss kind: {kind!r}")


def _masked_relational_distillation(pred, target, m):
    """Match normalised pairwise distances among the matched samples."""
    idx = (m > 0).nonzero(as_tuple=True)[0]
    if idx.numel() < 2:
        return pred.sum() * 0.0
    p, t = pred[idx], target[idx]
    n = p.size(0)
    off = ~torch.eye(n, dtype=torch.bool, device=p.device)

    def normalized_pdist(x):
        d = torch.cdist(x, x, p=2)
        return d / d[off].mean().clamp_min(1e-8)

    return F.smooth_l1_loss(normalized_pdist(p)[off], normalized_pdist(t)[off])


def disease_consensus_loss(
    audio_shared: torch.Tensor,
    audio_labels: torch.Tensor,
    class_prototypes: torch.Tensor,
    mask: torch.Tensor,
    *,
    topic_prototypes: torch.Tensor | None = None,
    topic_temperature: float = 0.5,
    disable_disease: bool = False,
    disable_topic: bool = False,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Pull the audio shared embedding toward detached teacher prototypes.

    ``class_prototypes`` is (C, D); ``topic_prototypes``, if given, is (C, K, D)
    and each sample is soft-assigned to the K topics of its own class. Returns
    the total and a dict of its parts.
    """
    z = F.normalize(audio_shared, dim=1)
    protos = F.normalize(class_prototypes, dim=1)
    weight = mask.float()
    denom = weight.sum().clamp_min(1.0)
    safe_labels = audio_labels.clamp(min=0, max=protos.size(0) - 1)

    parts: Dict[str, torch.Tensor] = {}
    total = z.new_zeros(())

    if not disable_disease:
        target = protos[safe_labels]
        disease = ((1.0 - (z * target).sum(dim=1)) * weight).sum() / denom
        parts["disease"] = disease
        total = total + disease

    if topic_prototypes is not None and not disable_topic:
        topics = F.normalize(topic_prototypes, dim=2)
        sample_topics = topics[safe_labels]
        similarity = torch.einsum("bd,bkd->bk", z, sample_topics)
        assignment = F.softmax(similarity / max(topic_temperature, 1e-6), dim=1)
        consensus = torch.einsum("bk,bkd->bd", assignment, sample_topics)
        consensus = F.normalize(consensus, dim=1)
        topic = ((1.0 - (z * consensus).sum(dim=1)) * weight).sum() / denom
        parts["topic"] = topic
        total = total + topic

    parts["total"] = total
    return total, parts
