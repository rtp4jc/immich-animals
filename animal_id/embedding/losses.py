"""
Defines the loss functions / margin heads for metric learning.

What it's for:
This script contains the margin-based heads that drive the embedding model's
training. Instead of just classifying, these heads teach the model to create a
geometrically meaningful embedding space, where similar items are close together
and dissimilar items are far apart.

What it does:
1. Defines a common `MarginHead` base class (an ``nn.Module``) that fixes the
   shared interface every margin head exposes: an L2-normalized class-prototype
   weight matrix and ``forward(embeddings, labels) -> logits``.
2. Implements `ArcFaceLoss` (Additive Angular Margin), `SubCenterArcFace`
   (K sub-centers per class, the MiewID recipe), and `CosFaceLoss` (additive
   cosine margin), all subclassing `MarginHead`.
3. Exposes a `HEAD_TYPES` registry plus a `build_head(...)` factory so the head
   can be selected from configuration with a single key.
4. Provides access to the standard PyTorch `TripletLoss` for comparison or
   alternative training strategies.
5. Includes a self-testing block to verify that the heads are implemented
   correctly and produce valid, non-zero loss values for non-trivial cases.

How to run it:
- This script is not typically run directly. It is imported by the embedding
  trainer.
- To run the self-tests, run from the project root:
  `python -m animal_id.embedding.losses`

How to interpret the results:
When running the self-test, the script will instantiate each head, run forward
passes with controlled dummy data, and assert the outputs have the correct shape
and produce a positive CrossEntropyLoss. A successful run indicates the heads are
correctly implemented.
"""

import math
from enum import Enum

import torch
import torch.nn as nn
import torch.nn.functional as F


class HeadType(str, Enum):
    """Enumeration of the available margin head types."""

    ARCFACE = "arcface"
    SUBCENTER_ARCFACE = "subcenter_arcface"
    COSFACE = "cosface"


class MarginHead(nn.Module):
    """
    Common base class for margin-based classification heads.

    A margin head owns a matrix of trainable class prototypes (one or more per
    class) and, given a batch of L2-normalized embeddings plus integer labels,
    returns scaled logits of shape ``(batch_size, num_classes)``. The actual
    loss is computed with a standard ``CrossEntropyLoss`` outside this module.

    Subclasses implement :meth:`forward`. The base class standardizes the
    constructor signature, the prototype weight matrix, and the scale ``s``.
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
        """
        Args:
            in_features (int): Size of the input embedding vectors.
            out_features (int): Number of classes (identities).
            s (float): Feature scaling factor.
            m (float): Margin (interpretation depends on the subclass).
            label_smoothing (float): Label smoothing factor. Retained on the
                module so trainers can read it back; the actual smoothing is
                applied by the external CrossEntropyLoss.
            num_centers (int): Number of sub-centers (prototypes) per class.
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.label_smoothing = label_smoothing
        self.num_centers = num_centers

        # The weight matrix holds the class prototypes. With ``num_centers > 1``
        # each class owns several sub-centers (sub-center ArcFace).
        self.weight = nn.Parameter(
            torch.FloatTensor(out_features * num_centers, in_features)
        )
        nn.init.xavier_uniform_(self.weight)

    def cosine_logits(self, embeddings):
        """
        Cosine similarity between (already L2-normalized) embeddings and the
        L2-normalized class prototypes, of shape ``(batch_size, out_features)``.

        For multi-center heads the per-class score is the max over that class's
        sub-centers, which is the sub-center ArcFace pooling step.
        """
        normalized_weights = F.normalize(self.weight, p=2, dim=1)
        cosine = F.linear(embeddings, normalized_weights)

        if self.num_centers > 1:
            cosine = cosine.view(-1, self.out_features, self.num_centers)
            cosine, _ = torch.max(cosine, dim=2)

        return cosine

    def forward(self, embeddings, labels):  # pragma: no cover - abstract
        raise NotImplementedError


class ArcFaceLoss(MarginHead):
    """
    Implementation of ArcFace loss (Additive Angular Margin Loss).
    This loss function is designed to increase the discriminative power of embeddings.
    Reference: https://arxiv.org/abs/1801.07698
    """

    def __init__(self, in_features, out_features, s=30.0, m=0.50, label_smoothing=0.1):
        """
        Args:
            in_features (int): Size of the input embedding vectors.
            out_features (int): Number of classes (identities).
            s (float): Feature scaling factor.
            m (float): Additive angular margin.
            label_smoothing (float): Label smoothing factor for cross entropy loss.
        """
        super().__init__(
            in_features,
            out_features,
            s=s,
            m=m,
            label_smoothing=label_smoothing,
            num_centers=1,
        )

        # Constants for numerical stability
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(
            math.pi - m
        )  # Threshold to prevent theta + m from going > pi
        self.mm = math.sin(math.pi - m) * m

    def _apply_margin(self, cosine, labels):
        """
        Apply the additive angular margin to the target-class cosine values.

        Shared by ArcFace and Sub-center ArcFace (which differ only in how the
        per-class ``cosine`` matrix is produced).
        """
        # Convert labels to one-hot format
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.view(-1, 1).long(), 1)

        # --- Additive Angular Margin --- #
        # Get the cosine of the angle between the embedding and its true class prototype
        target_cosine = cosine[one_hot.bool()]

        # Calculate the angle theta
        sine = torch.sqrt(1.0 - torch.pow(target_cosine, 2))

        # Calculate cos(theta + m)
        # This is the trigonometric addition formula: cos(a+b) = cos(a)cos(b) - sin(a)sin(b)
        marginal_target_cosine = target_cosine * self.cos_m - sine * self.sin_m

        # If theta + m > pi, use a simplified penalty (from the original paper)
        # This prevents the angle from wrapping around
        marginal_target_cosine = torch.where(
            target_cosine > self.th, marginal_target_cosine, target_cosine - self.mm
        )

        # Create the final output logits by substituting the modified cosine for the true class
        output_logits = cosine.clone()
        output_logits[one_hot.bool()] = marginal_target_cosine

        # Scale the logits before passing to CrossEntropyLoss
        output_logits *= self.s

        # The actual loss calculation is done with CrossEntropyLoss outside this module
        return output_logits

    def forward(self, embeddings, labels):
        """
        Args:
            embeddings (torch.Tensor): L2-normalized input embeddings of shape (batch_size, in_features).
            labels (torch.Tensor): Ground truth labels of shape (batch_size).
        """
        cosine = self.cosine_logits(embeddings)
        return self._apply_margin(cosine, labels)


class SubCenterArcFace(ArcFaceLoss):
    """
    Sub-center ArcFace (Deng et al., ECCV'20).

    Each class is represented by ``k`` sub-centers instead of a single
    prototype. The per-class cosine score is the maximum cosine over that
    class's ``k`` sub-centers, after which the standard ArcFace angular margin is
    applied. The dominant sub-center soaks up the bulk of (clean) samples while
    noisy / off-distribution samples can latch onto a minority sub-center,
    making training robust to label noise and intra-identity appearance
    variance. This is the loss used by MiewID.

    Reference: https://ibug.doc.ic.ac.uk/media/uploads/documents/eccv_1445.pdf
    """

    def __init__(
        self, in_features, out_features, s=30.0, m=0.50, label_smoothing=0.1, k=3
    ):
        """
        Args:
            in_features (int): Size of the input embedding vectors.
            out_features (int): Number of classes (identities).
            s (float): Feature scaling factor.
            m (float): Additive angular margin.
            label_smoothing (float): Label smoothing factor for cross entropy loss.
            k (int): Number of sub-centers per class.
        """
        # Skip ArcFaceLoss.__init__ (it hardcodes num_centers=1); go straight to
        # the grandparent MarginHead with the requested number of sub-centers,
        # then set up the same angular-margin constants ArcFace uses.
        MarginHead.__init__(
            self,
            in_features,
            out_features,
            s=s,
            m=m,
            label_smoothing=label_smoothing,
            num_centers=k,
        )
        self.k = k

        # Constants for numerical stability (mirrors ArcFaceLoss)
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    # forward() is inherited from ArcFaceLoss: cosine_logits() max-pools over the
    # k sub-centers, then _apply_margin() applies the ArcFace angular margin.


class CosFaceLoss(MarginHead):
    """
    CosFace loss (Large Margin Cosine Loss).

    Subtracts an additive cosine margin ``m`` from the target-class cosine
    similarity before scaling, rather than adding an angular margin as ArcFace
    does. Reference: https://arxiv.org/abs/1801.09414
    """

    def __init__(self, in_features, out_features, s=30.0, m=0.35, label_smoothing=0.1):
        """
        Args:
            in_features (int): Size of the input embedding vectors.
            out_features (int): Number of classes (identities).
            s (float): Feature scaling factor.
            m (float): Additive cosine margin.
            label_smoothing (float): Label smoothing factor for cross entropy loss.
        """
        super().__init__(
            in_features,
            out_features,
            s=s,
            m=m,
            label_smoothing=label_smoothing,
            num_centers=1,
        )

    def forward(self, embeddings, labels):
        """
        Args:
            embeddings (torch.Tensor): L2-normalized input embeddings of shape (batch_size, in_features).
            labels (torch.Tensor): Ground truth labels of shape (batch_size).
        """
        cosine = self.cosine_logits(embeddings)

        # Subtract the additive cosine margin from the true-class similarity.
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.view(-1, 1).long(), 1)
        output_logits = cosine - one_hot * self.m

        # Scale before passing to CrossEntropyLoss.
        output_logits *= self.s
        return output_logits


# Registry mapping a HeadType to its implementing class.
HEAD_TYPES = {
    HeadType.ARCFACE: ArcFaceLoss,
    HeadType.SUBCENTER_ARCFACE: SubCenterArcFace,
    HeadType.COSFACE: CosFaceLoss,
}


def build_head(head_type, embedding_dim, num_classes, **kwargs):
    """
    Factory that instantiates the requested margin head.

    Args:
        head_type (HeadType | str): Which head to build. Accepts a ``HeadType``
            or its string value (e.g. ``"arcface"``).
        embedding_dim (int): Size of the input embedding vectors.
        num_classes (int): Number of identity classes.
        **kwargs: Forwarded to the head constructor (e.g. ``s``, ``m``,
            ``label_smoothing``, and head-specific params like ``k``). Keys with
            a ``None`` value are dropped so the head's own defaults apply.

    Returns:
        MarginHead: The instantiated head.

    Raises:
        ValueError: If ``head_type`` is not a recognized head.
    """
    head_type = HeadType(head_type)
    head_cls = HEAD_TYPES[head_type]
    kwargs = {key: value for key, value in kwargs.items() if value is not None}
    return head_cls(embedding_dim, num_classes, **kwargs)


# For Triplet Loss, we can use the standard PyTorch implementation.
# The main complexity of Triplet Loss is in the data sampling (finding triplets),
# which is handled by a custom DataLoader sampler, not the loss function itself.
TripletLoss = nn.TripletMarginLoss


# --- Test script to verify loss functions --- #
if __name__ == "__main__":
    num_classes = 10
    embedding_dim = 512
    batch_size = 4

    dummy_embeddings = F.normalize(torch.randn(batch_size, embedding_dim), p=2, dim=1)
    dummy_labels = torch.randint(0, num_classes, (batch_size,))
    criterion = nn.CrossEntropyLoss()

    # --- Margin heads --- #
    for head_type in HeadType:
        print(f"--- Testing {head_type.value} ---")
        head = build_head(head_type, embedding_dim, num_classes)
        output_logits = head(dummy_embeddings, dummy_labels)
        print(f"Output logits shape: {output_logits.shape}")
        assert output_logits.shape == (batch_size, num_classes)

        loss = criterion(output_logits, dummy_labels)
        print(f"Calculated CrossEntropyLoss: {loss.item()}")
        assert loss.item() > 0
        loss.backward()
        print(f"{head_type.value} test passed!\n")

    # --- TripletLoss Test --- #
    print("--- Testing TripletLoss ---")
    margin = 1.0
    triplet_loss = TripletLoss(margin=margin, p=2)  # Explicitly use L2 distance

    # Test Case 1: Easy case (loss should be 0)
    anchor_easy = torch.tensor([[0.0, 1.0]])
    positive_easy = torch.tensor([[0.1, 0.9]])
    negative_easy = torch.tensor([[1.0, 0.0]])
    loss_easy = triplet_loss(anchor_easy, positive_easy, negative_easy)
    print(f"TripletLoss with easy case data: {loss_easy.item()}")
    assert loss_easy.item() == 0, "Easy case should result in zero loss"

    # Test Case 2: Hard case (loss should be > 0)
    anchor_hard = torch.tensor([[0.0, 0.0]])
    positive_hard = torch.tensor([[0.5, 0.0]])
    negative_hard = torch.tensor([[0.2, 0.0]])
    loss_hard = triplet_loss(anchor_hard, positive_hard, negative_hard)
    print(f"TripletLoss with hard case data: {loss_hard.item()}")
    assert loss_hard.item() > 0, "Hard case should result in positive loss"

    print("TripletLoss test passed!")
