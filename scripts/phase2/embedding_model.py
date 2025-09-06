"""
Defines the embedding model architecture.

What it's for:
This script defines the neural network used to generate identity embeddings from dog images.
It uses a pre-trained MobileNetV3-small as a backbone for efficient feature extraction
and adds a custom head to produce the final L2-normalized embedding vector.

What it does:
1. Defines the `EmbeddingNet` class, a PyTorch `nn.Module`.
2. Loads a MobileNetV3-small model pre-trained on ImageNet.
3. Removes the original classification layer.
4. Adds a new "projection head" that takes the features from the backbone and maps
   them to a 512-dimensional embedding space.
5. L2-normalizes the output vector, which is crucial for metric learning with losses
   like ArcFace.
6. Includes a self-testing block to verify the model's architecture and output shape.

How to run it:
- This script is not typically run directly. It is imported by other scripts like
  `training/train_embedding.py`.
- To run the self-test and verify the architecture, run from the project root:
  `python scripts/phase2/embedding_model.py`

How to interpret the results:
When running the self-test, the script will:
- Print the shape of the output embedding tensor, which should be `[batch_size, 512]`.
- Print the L2 norms of the output vectors, which should all be `1.0`.
- A successful run indicates that the model is architecturally sound.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights

class EmbeddingNet(nn.Module):
    """
    A lightweight embedding network based on a pre-trained MobileNetV3-small backbone.
    The network is designed to produce a fixed-size embedding vector for metric learning.
    """
    def __init__(self, embedding_dim=512, pretrained=True, dropout_prob=0.5):
        """
        Args:
            embedding_dim (int): The dimensionality of the output embedding vector.
            pretrained (bool): Whether to use weights pre-trained on ImageNet.
            dropout_prob (float): Probability for the dropout layer.
        """
        super(EmbeddingNet, self).__init__()

        # Load the pre-trained MobileNetV3-small model
        if pretrained:
            weights = MobileNet_V3_Small_Weights.IMAGENET1K_V1
        else:
            weights = None
            
        self.backbone = mobilenet_v3_small(weights=weights)

        # Isolate the feature extractor part of the backbone
        self.feature_extractor = self.backbone.features

        # Get the number of output channels from the feature extractor
        in_features = self.backbone.classifier[0].in_features

        # Define the new projection head with Dropout
        self.projection_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(p=dropout_prob),
            nn.Linear(in_features, embedding_dim),
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

    # Instantiate the model
    print("Instantiating EmbeddingNet with Dropout...")
    model = EmbeddingNet(embedding_dim=512, dropout_prob=0.5)
    
    # Test in training mode to ensure dropout is active
    model.train()
    print("Performing forward pass in training mode...")
    with torch.no_grad():
        output_embeddings_train = model(dummy_input)

    # Test in eval mode to ensure dropout is inactive
    model.eval()
    print("Performing forward pass in eval mode...")
    with torch.no_grad():
        output_embeddings_eval = model(dummy_input)

    # In eval mode, the output should be deterministic. In train mode, it might not be.
    # We mainly care that it runs and produces the correct shape.
    print(f"Output embedding shape (eval): {output_embeddings_eval.shape}")
    assert output_embeddings_eval.shape == (batch_size, 512)

    norms = torch.norm(output_embeddings_eval, p=2, dim=1)
    print(f"L2 norms of output embeddings (eval): {norms}")
    assert torch.allclose(norms, torch.ones(batch_size), atol=1e-6)

    print("\nModel architecture test passed!")