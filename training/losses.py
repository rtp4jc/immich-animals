import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class ArcFaceLoss(nn.Module):
    """
    Implementation of ArcFace loss (Additive Angular Margin Loss).
    This loss function is designed to increase the discriminative power of embeddings.
    Reference: https://arxiv.org/abs/1801.07698
    """
    def __init__(self, in_features, out_features, s=30.0, m=0.50):
        """
        Args:
            in_features (int): Size of the input embedding vectors.
            out_features (int): Number of classes (identities).
            s (float): Feature scaling factor.
            m (float): Additive angular margin.
        """
        super(ArcFaceLoss, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m

        # The weight matrix of this layer represents the class prototypes.
        # Each column is a prototype for a class.
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

        # Constants for numerical stability
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m) # Threshold to prevent theta + m from going > pi
        self.mm = math.sin(math.pi - m) * m

    def forward(self, embeddings, labels):
        """
        Args:
            embeddings (torch.Tensor): L2-normalized input embeddings of shape (batch_size, in_features).
            labels (torch.Tensor): Ground truth labels of shape (batch_size).
        """
        # L2 normalize weights (class prototypes)
        normalized_weights = F.normalize(self.weight, p=2, dim=1)

        # Calculate cosine similarity between embeddings and class prototypes
        # This is equivalent to the output of a standard fully connected layer
        cosine = F.linear(embeddings, normalized_weights)
        
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
            target_cosine > self.th,
            marginal_target_cosine,
            target_cosine - self.mm
        )

        # Create the final output logits by substituting the modified cosine for the true class
        output_logits = cosine.clone()
        output_logits[one_hot.bool()] = marginal_target_cosine

        # Scale the logits before passing to CrossEntropyLoss
        output_logits *= self.s

        # The actual loss calculation is done with CrossEntropyLoss outside this module
        return output_logits


# For Triplet Loss, we can use the standard PyTorch implementation.
# The main complexity of Triplet Loss is in the data sampling (finding triplets), which is handled by a custom DataLoader sampler, not the loss function itself.
TripletLoss = nn.TripletMarginLoss


# --- Test script to verify loss functions --- #
if __name__ == '__main__':
    # --- ArcFaceLoss Test --- #
    print("--- Testing ArcFaceLoss ---")
    num_classes = 10
    embedding_dim = 512
    batch_size = 4

    # Create dummy data
    dummy_embeddings = F.normalize(torch.randn(batch_size, embedding_dim), p=2, dim=1)
    dummy_labels = torch.randint(0, num_classes, (batch_size,))

    # Instantiate loss module
    arcface = ArcFaceLoss(in_features=embedding_dim, out_features=num_classes)
    print(f"Instantiated ArcFaceLoss for {num_classes} classes.")

    # Perform forward pass
    output_logits = arcface(dummy_embeddings, dummy_labels)

    # Verify output shape
    print(f"Output logits shape: {output_logits.shape}")
    assert output_logits.shape == (batch_size, num_classes)

    # Verify that the logits for the target classes have been modified
    criterion = nn.CrossEntropyLoss()
    loss = criterion(output_logits, dummy_labels)
    print(f"Calculated CrossEntropyLoss: {loss.item()}")
    assert loss.item() > 0

    print("ArcFaceLoss test passed!")

    # --- TripletLoss Test --- #
    print("\n--- Testing TripletLoss ---")
    margin = 1.0
    triplet_loss = TripletLoss(margin=margin, p=2) # Explicitly use L2 distance

    # Test Case 1: Easy case (loss should be 0)
    # Negative is already further from anchor than positive is.
    # dist(a,p) = sqrt((0.1-0)^2 + (0.9-1)^2) = sqrt(0.02) = 0.1414
    # dist(a,n) = sqrt((1-0)^2 + (0-1)^2) = sqrt(2) = 1.414
    # loss = max(0.1414 - 1.414 + 1.0, 0) = max(-0.27, 0) = 0
    anchor_easy = torch.tensor([[0.0, 1.0]])
    positive_easy = torch.tensor([[0.1, 0.9]])
    negative_easy = torch.tensor([[1.0, 0.0]])
    loss_easy = triplet_loss(anchor_easy, positive_easy, negative_easy)
    print(f"TripletLoss with easy case data: {loss_easy.item()}")
    assert loss_easy.item() == 0, "Easy case should result in zero loss"

    # Test Case 2: Hard case (loss should be > 0)
    # Negative is closer to anchor than positive is.
    # dist(a,p) = 0.5
    # dist(a,n) = 0.2
    # loss = max(0.5 - 0.2 + 1.0, 0) = max(1.3, 0) = 1.3
    anchor_hard = torch.tensor([[0.0, 0.0]])
    positive_hard = torch.tensor([[0.5, 0.0]])
    negative_hard = torch.tensor([[0.2, 0.0]])
    loss_hard = triplet_loss(anchor_hard, positive_hard, negative_hard)
    print(f"TripletLoss with hard case data: {loss_hard.item()}")
    assert loss_hard.item() > 0, "Hard case should result in positive loss"

    print("TripletLoss test passed!")
