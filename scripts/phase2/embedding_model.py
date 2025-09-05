import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights

class EmbeddingNet(nn.Module):
    """
    A lightweight embedding network based on a pre-trained MobileNetV3-small backbone.
    The network is designed to produce a fixed-size embedding vector for metric learning.
    """
    def __init__(self, embedding_dim=512, pretrained=True):
        """
        Args:
            embedding_dim (int): The dimensionality of the output embedding vector.
            pretrained (bool): Whether to use weights pre-trained on ImageNet.
        """
        super(EmbeddingNet, self).__init__()

        # Load the pre-trained MobileNetV3-small model
        if pretrained:
            weights = MobileNet_V3_Small_Weights.IMAGENET1K_V1
        else:
            weights = None
            
        self.backbone = mobilenet_v3_small(weights=weights)

        # Isolate the feature extractor part of the backbone
        # We are taking layers up to the adaptive average pooling
        self.feature_extractor = self.backbone.features

        # Get the number of output channels from the feature extractor
        # For MobileNetV3-small, this is 576
        in_features = self.backbone.classifier[0].in_features

        # Define the new projection head
        self.projection_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(in_features, embedding_dim),
            # No activation function here, as ArcFace/CosFace will use the logits
        )

        # Initialize the weights of the new projection head
        self.projection_head[-1].apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """
        Forward pass to extract embeddings.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 3, H, W).

        Returns:
            torch.Tensor: L2-normalized embedding tensor of shape (batch_size, embedding_dim).
        """
        features = self.feature_extractor(x)
        embeddings = self.projection_head(features)
        
        # L2 normalize the embeddings
        normalized_embeddings = F.normalize(embeddings, p=2, dim=1)
        
        return normalized_embeddings

# --- Test script to verify model architecture ---
if __name__ == '__main__':
    # Create a dummy input tensor
    batch_size = 4
    img_size = 224
    dummy_input = torch.randn(batch_size, 3, img_size, img_size)

    # Instantiate the model
    print("Instantiating EmbeddingNet...")
    model = EmbeddingNet(embedding_dim=512)
    model.eval() # Set to evaluation mode

    # Perform a forward pass
    print(f"Performing forward pass with input shape: {dummy_input.shape}")
    with torch.no_grad():
        output_embeddings = model(dummy_input)

    # Verify the output shape
    print(f"Output embedding shape: {output_embeddings.shape}")
    assert output_embeddings.shape == (batch_size, 512)

    # Verify that the embeddings are L2-normalized
    norms = torch.norm(output_embeddings, p=2, dim=1)
    print(f"L2 norms of output embeddings: {norms}")
    assert torch.allclose(norms, torch.ones(batch_size), atol=1e-6)

    print("\nModel architecture test passed!")
    print(f"- Output shape is correct: {output_embeddings.shape}")
    print(f"- Embeddings are L2-normalized.")

    # Optional: Print model summary
    # try:
    #     from torchinfo import summary
    #     print("\nModel Summary:")
    #     summary(model, input_size=dummy_input.shape)
    # except ImportError:
    #     print("\nSkipping model summary: `torchinfo` not installed.")
    #     print("Install with: pip install torchinfo")
