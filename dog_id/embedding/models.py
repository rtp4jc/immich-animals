"""
Defines the embedding model architecture.

What it's for:
This script defines the neural network used to generate identity embeddings from dog images.
It uses a backbone (e.g., EfficientNet) created by the backbone factory and adds a
custom head to produce the final L2-normalized embedding vector.

What it does:
1. Defines the `EmbeddingNet` class, a PyTorch `nn.Module`.
2. Uses the `get_backbone` factory to instantiate a feature extractor and get its
   output feature dimension.
3. Adds a new "projection head" that takes the features from the backbone and maps
   them to a 512-dimensional embedding space.
4. L2-normalizes the output vector, which is crucial for metric learning.
5. Includes a self-testing block to verify the model's architecture.

How to run it:
- This script is not typically run directly. It is imported by other scripts.
- To run the self-test, run from the project root:
  `python -m dog_id.embedding.models`
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from dog_id.embedding.backbones import get_backbone

class EmbeddingNet(nn.Module):
    """
    A generic embedding network that uses a backbone from the factory.
    """
    def __init__(self, backbone_name: str, embedding_dim: int = 512, pretrained: bool = True, dropout_prob: float = 0.5):
        """
        Args:
            backbone_name (str): Name of the backbone to use (e.g., 'efficientnet_b0').
            embedding_dim (int): The dimensionality of the output embedding vector.
            pretrained (bool): Whether to use weights pre-trained on ImageNet for the backbone.
            dropout_prob (float): Probability for the dropout layer.
        """
        super(EmbeddingNet, self).__init__()

        self.feature_extractor, num_features = get_backbone(backbone_name, pretrained)

        # Define the new projection head with Dropout
        self.projection_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(p=dropout_prob),
            nn.Linear(num_features, embedding_dim),
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

    # Test with EfficientNet
    print("--- Testing with EfficientNet-B0 ---")
    model_eff = EmbeddingNet(backbone_name='efficientnet_b0')
    model_eff.eval()
    output_eff = model_eff(dummy_input)
    print(f"Output shape (EfficientNet-B0): {output_eff.shape}")
    assert output_eff.shape == (batch_size, 512)
    print("EfficientNet-B0 test passed!")

    # Test with MobileNet
    print("\n--- Testing with MobileNetV3-Small ---")
    model_mob = EmbeddingNet(backbone_name='mobilenet_v3_small')
    model_mob.eval()
    output_mob = model_mob(dummy_input)
    print(f"Output shape (MobileNetV3-Small): {output_mob.shape}")
    assert output_mob.shape == (batch_size, 512)
    print("MobileNetV3-Small test passed!")

    print("\nModel architecture tests passed!")