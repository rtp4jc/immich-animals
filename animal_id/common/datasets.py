"""PyTorch Dataset objects bridging the prepared .json annotations and training."""

import json
import logging

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

logger = logging.getLogger(__name__)


class IdentityDataset(Dataset):
    """
    Enhanced dataset with built-in data augmentations for animal identity training.
    """

    def __init__(self, json_path, img_size=224, is_training=True):
        with open(json_path) as f:
            self.annotations = json.load(f)

        # Get max label value for ArcFace (not just unique count)
        all_labels = [item["identity_label"] for item in self.annotations]
        self.num_classes = max(all_labels) + 1  # +1 because labels are 0-indexed

        # Data augmentations
        if is_training:
            self.transform = transforms.Compose(
                [
                    transforms.Resize((img_size + 32, img_size + 32)),
                    transforms.RandomCrop(img_size),
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.ColorJitter(
                        brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1
                    ),
                    transforms.RandomRotation(degrees=15),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    ),
                    transforms.RandomErasing(
                        p=0.1, scale=(0.02, 0.33), ratio=(0.3, 3.3)
                    ),
                ]
            )
        else:
            self.transform = transforms.Compose(
                [
                    transforms.Resize((img_size, img_size)),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    ),
                ]
            )

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        img_path = self.annotations[idx]["file_path"]
        identity_label = self.annotations[idx]["identity_label"]

        # Use original identity label (no remapping)

        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            logger.warning(f"Error loading image {img_path}: {e}. Skipping.")
            return self.__getitem__((idx + 1) % len(self))

        image = self.transform(image)
        return image, identity_label
