"""Loss functions for the unpaired audio<-privileged transfer methods.

Three mechanisms share this file, all designed so the privileged signal actually
reaches the audio encoder (unlike the throttled version this repo replaces):

1. Contrastive clinical-pivot  -> ``label_aware_clip_loss`` (and ``vanilla_clip_loss``
   for the ablation). Audio and each privileged view align to a frozen clinical-text
   manifold; same-diagnosis samples are positives.
2. Masked feature distillation -> ``masked_distillation_loss``. Cosine pull of the
   projected audio feature toward a teacher target, applied only to matched samples.
3. Disease-topic consensus (DTCA) -> ``disease_consensus_loss``. Audio's shared
   embedding is pulled toward detached teacher class prototypes (and, optionally,
   class-conditioned clinical-topic prototypes). Guidance is a plain weighted term
   with NO gradient-orthogonalisation / cosine-gate throttling; the training script
   classifies through the shared subspace so the signal cannot be routed around.
"""

from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
# Contrastive clinical-pivot
# --------------------------------------------------------------------------- #
def label_aware_clip_loss(emb_a, emb_b, labels_a, labels_b, logit_scale):
    """Soft-target CLIP loss: any two same-class samples are positives.

    Robust to duplicate labels and small imbalanced clinical batches, where the
    diagonal-only InfoNCE (``vanilla_clip_loss``) degenerates.
    """
    emb_a = F.normalize(emb_a, dim=1)
    emb_b = F.normalize(emb_b, dim=1)
    logits = logit_scale * (emb_a @ emb_b.T)
    mask = torch.eq(labels_a.view(-1, 1), labels_b.view(1, -1)).float()
    mask = mask / (mask.sum(dim=1, keepdim=True) + 1e-8)
    log_probs = F.log_softmax(logits, dim=1)
    return -(mask * log_probs).sum(dim=1).mean()


def vanilla_clip_loss(emb_a, emb_b, logit_scale):
    """Standard symmetric InfoNCE (diagonal positives only) -- ablation baseline."""
    emb_a = F.normalize(emb_a, dim=1)
    emb_b = F.normalize(emb_b, dim=1)
    logits = logit_scale * (emb_a @ emb_b.T)
    targets = torch.arange(len(logits), device=emb_a.device)
    return 0.5 * (F.cross_entropy(logits, targets) + F.cross_entropy(logits.T, targets))


# --------------------------------------------------------------------------- #
# Masked feature distillation
# --------------------------------------------------------------------------- #
def masked_distillation_loss(projected_audio, target, mask, kind: str = "cosine"):
    """Pull the projected audio feature toward a teacher target, over matched samples.

    ``mask`` is 1.0 where a valid teacher target exists (COPD/Healthy audio matched
    to a privileged profile) and 0.0 otherwise (e.g. ``Other``), so the 3-class
    audio task can borrow from a binary teacher without corrupting unmatched classes.

    ``kind`` selects the objective (all masked-averaged):
      - "cosine"     : direction only (default; the current method).
      - "mse"        : direction + magnitude (squared error).
      - "smooth_l1"  : robust version of mse.
      - "cosine_mse" : sum of the two.
      - "relational" : match the pairwise geometry BETWEEN patients rather than
                       absolute vectors (Relational KD, Park et al. 2019) -- a more
                       defensible cross-modal variant, since it transfers relative
                       structure and cannot be satisfied by regressing to a constant.
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
    """Relational KD (distance form): match the normalized pairwise-distance matrix
    of the matched audio features to that of their teacher targets."""
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


# --------------------------------------------------------------------------- #
# Disease-topic consensus (DTCA)
# --------------------------------------------------------------------------- #
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
    """Pull audio's shared embedding toward detached privileged prototypes.

    Args:
        audio_shared:     (B, D) audio shared embedding (grad flows here).
        audio_labels:     (B,) class indices; unmatched classes are excluded via ``mask``.
        class_prototypes: (C, D) per-class teacher-consensus centroids (DETACHED).
        mask:             (B,) 1.0 for privileged-relevant samples (Healthy/COPD).
        topic_prototypes: optional (C, K, D) class-conditioned clinical-topic centroids
                          (DETACHED). Audio is soft-assigned to its class's K topics.

    Returns ``(total, parts)`` where parts holds ``disease``/``topic``/``total``.
    """
    z = F.normalize(audio_shared, dim=1)
    protos = F.normalize(class_prototypes, dim=1)
    weight = mask.float()
    denom = weight.sum().clamp_min(1.0)
    safe_labels = audio_labels.clamp(min=0, max=protos.size(0) - 1)

    parts: Dict[str, torch.Tensor] = {}
    total = z.new_zeros(())

    if not disable_disease:
        target = protos[safe_labels]  # (B, D)
        disease = ((1.0 - (z * target).sum(dim=1)) * weight).sum() / denom
        parts["disease"] = disease
        total = total + disease

    if topic_prototypes is not None and not disable_topic:
        topics = F.normalize(topic_prototypes, dim=2)  # (C, K, D)
        sample_topics = topics[safe_labels]            # (B, K, D)
        similarity = torch.einsum("bd,bkd->bk", z, sample_topics)  # (B, K)
        assignment = F.softmax(similarity / max(topic_temperature, 1e-6), dim=1)
        consensus = torch.einsum("bk,bkd->bd", assignment, sample_topics)  # (B, D)
        consensus = F.normalize(consensus, dim=1)
        topic = ((1.0 - (z * consensus).sum(dim=1)) * weight).sum() / denom
        parts["topic"] = topic
        total = total + topic

    parts["total"] = total
    return total, parts
