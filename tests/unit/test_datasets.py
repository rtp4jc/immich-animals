import torch
from animal_id.common.datasets import DogIdentityDataset

def test_dataset_initialization(mock_image_dataset):
    """Test proper initialization of the dataset."""
    dataset = DogIdentityDataset(json_path=mock_image_dataset, img_size=224, is_training=True)
    assert len(dataset) == 5
    # mock_image_dataset creates 2 identities so num_classes should be 2
    assert dataset.num_classes == 2

def test_dataset_getitem(mock_image_dataset):
    """Test retrieving an item from the dataset."""
    dataset = DogIdentityDataset(json_path=mock_image_dataset, img_size=224, is_training=True)
    
    img, label = dataset[0]
    
    # Check return types
    assert isinstance(img, torch.Tensor)
    assert isinstance(label, int)
    
    # Check tensor shape (3 channels, 224x224)
    assert img.shape == (3, 224, 224)
    
    # Check label validity
    assert label in [0, 1]

def test_dataset_transforms_training(mock_image_dataset):
    """Test that training transforms are applied (smoke test)."""
    # Training has random augmentations, so multiple calls might yield different tensors
    dataset = DogIdentityDataset(json_path=mock_image_dataset, img_size=224, is_training=True)
    
    # Get the same image index twice (should return augmented versions of the same image)
    img1, _ = dataset[0]
    img2, _ = dataset[0]
    
    # In *most* cases, random augmentations will produce different tensors.
    # It's statistically possible they are identical but very unlikely.
    # However, due to ColorJitter/RandomCrop, they should differ.
    assert not torch.allclose(img1, img2)
    
    assert img1.shape == (3, 224, 224)

def test_dataset_transforms_validation(mock_image_dataset):
    """Test that validation transforms are deterministic."""
    dataset = DogIdentityDataset(json_path=mock_image_dataset, img_size=224, is_training=False)
    
    img1, _ = dataset[0]
    img2, _ = dataset[0]
    
    # Validation transform (Resize + Normalize) should be deterministic
    assert torch.allclose(img1, img2)
    assert img1.shape == (3, 224, 224)
