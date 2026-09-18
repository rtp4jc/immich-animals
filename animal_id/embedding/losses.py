"""Margin-based classification heads for metric learning (ArcFace, CosFace).

Each head owns the class prototypes and returns scaled logits; the loss itself is a
standard ``CrossEntropyLoss`` applied by the trainer. Select one via ``build_head``.
"""

import math
from enum import StrEnum

import torch
import torch.nn as nn
import torch.nn.functional as F


class HeadType(StrEnum):
    ARCFACE = "arcface"
    SUBCENTER_ARCFACE = "subcenter_arcface"
    COSFACE = "cosface"


class MarginHead(nn.Module):
    """Base class for margin heads.

    Owns a trainable prototype matrix (``num_centers`` per class) and maps a batch of
    L2-normalized embeddings plus integer labels to ``(batch_size, out_features)``
    logits.

    Args:
        in_features: Size of the input embedding vectors.
        out_features: Number of classes (identities).
        s: Feature scaling factor.
        m: Margin; its interpretation depends on the subclass.
        label_smoothing: Kept on the module so trainers can read it back; the
            smoothing itself is applied by the external CrossEntropyLoss.
        num_centers: Number of sub-centers (prototypes) per class.
    """

    def __init__(
        self,
        in_features,
        out_features,
        s=30.0,
        m=0.50,
        label_smoothing=0.1,
        num_centers=1,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.label_smoothing = label_smoothing
        self.num_centers = num_centers

        self.weight = nn.Parameter(
            torch.FloatTensor(out_features * num_centers, in_features)
        )
        nn.init.xavier_uniform_(self.weight)

    def cosine_logits(self, embeddings):
        """Cosine similarity between embeddings and the L2-normalized prototypes.

        For multi-center heads the per-class score is the max over that class's
        sub-centers, which is the sub-center ArcFace pooling step.
        """
        cosine = F.linear(embeddings, F.normalize(self.weight, p=2, dim=1))

        if self.num_centers > 1:
            cosine = cosine.view(-1, self.out_features, self.num_centers)
            cosine, _ = torch.max(cosine, dim=2)

        return cosine

    def forward(self, embeddings, labels):  # pragma: no cover - abstract
        raise NotImplementedError


class ArcFaceLoss(MarginHead):
    """Additive Angular Margin Loss. https://arxiv.org/abs/1801.07698

    ``m`` is an additive angular margin. Pass ``num_centers > 1`` for sub-center
    ArcFace (see :class:`SubCenterArcFace`).
    """

    def __init__(
        self,
        in_features,
        out_features,
        s=30.0,
        m=0.50,
        label_smoothing=0.1,
        num_centers=1,
    ):
        super().__init__(
            in_features,
            out_features,
            s=s,
            m=m,
            label_smoothing=label_smoothing,
            num_centers=num_centers,
        )

        # Constants for numerical stability.
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)  # Keeps theta + m from exceeding pi.
        self.mm = math.sin(math.pi - m) * m

    def forward(self, embeddings, labels):
        cosine = self.cosine_logits(embeddings)

        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.view(-1, 1).long(), 1)
        target_cosine = cosine[one_hot.bool()]

        # cos(theta + m) via the trigonometric addition formula.
        sine = torch.sqrt(1.0 - torch.pow(target_cosine, 2))
        marginal_target_cosine = target_cosine * self.cos_m - sine * self.sin_m

        # Past theta + m > pi, fall back to the paper's linear penalty so the angle
        # cannot wrap around.
        marginal_target_cosine = torch.where(
            target_cosine > self.th, marginal_target_cosine, target_cosine - self.mm
        )

        output_logits = cosine.clone()
        output_logits[one_hot.bool()] = marginal_target_cosine
        return output_logits * self.s


class SubCenterArcFace(ArcFaceLoss):
    """ArcFace with ``k`` sub-centers per class (Deng et al., ECCV'20).

    The dominant sub-center absorbs the bulk of the clean samples while noisy ones
    latch onto a minority sub-center, making training robust to label noise and
    intra-identity appearance variance. This is the loss used by MiewID.
    https://ibug.doc.ic.ac.uk/media/uploads/documents/eccv_1445.pdf
    """

    def __init__(self, in_features, out_features, k=3, **kwargs):
        super().__init__(in_features, out_features, num_centers=k, **kwargs)
        self.k = k


class CosFaceLoss(MarginHead):
    """Large Margin Cosine Loss. https://arxiv.org/abs/1801.09414

    Subtracts an additive cosine margin ``m`` from the target-class similarity before
    scaling, rather than adding an angular margin as ArcFace does.
    """

    def __init__(self, in_features, out_features, s=30.0, m=0.35, label_smoothing=0.1):
        super().__init__(
            in_features,
            out_features,
            s=s,
            m=m,
            label_smoothing=label_smoothing,
            num_centers=1,
        )

    def forward(self, embeddings, labels):
        cosine = self.cosine_logits(embeddings)

        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.view(-1, 1).long(), 1)
        return (cosine - one_hot * self.m) * self.s


HEAD_TYPES = {
    HeadType.ARCFACE: ArcFaceLoss,
    HeadType.SUBCENTER_ARCFACE: SubCenterArcFace,
    HeadType.COSFACE: CosFaceLoss,
}

# Triplet loss lives in the sampler, not the loss: the standard PyTorch
# implementation is all that's needed here.
TripletLoss = nn.TripletMarginLoss


def build_head(head_type, embedding_dim, num_classes, **kwargs):
    """Instantiate the requested margin head.

    ``head_type`` accepts a ``HeadType`` or its string value. ``kwargs`` are forwarded
    to the head constructor; ``None`` values are dropped so its own defaults apply.

    Raises:
        ValueError: If ``head_type`` is not a recognized head.
    """
    head_cls = HEAD_TYPES[HeadType(head_type)]
    kwargs = {key: value for key, value in kwargs.items() if value is not None}
    return head_cls(embedding_dim, num_classes, **kwargs)
