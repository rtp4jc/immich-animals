import json
from PIL import Image
from torch.utils.data import Dataset

class IdentityDataset(Dataset):
    """
    PyTorch Dataset for loading dog identity images from a JSON file.
    The JSON file is expected to be a list of dictionaries, with each
    dictionary containing 'file_path', 'identity_label', and 'breed_label'.
    """
    def __init__(self, json_path, transform=None):
        """
        Args:
            json_path (string): Path to the JSON file with annotations.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        with open(json_path, 'r') as f:
            self.annotations = json.load(f)
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        """
        Returns a tuple of (image, identity_label).
        """
        img_path = self.annotations[idx]['file_path']
        identity_label = self.annotations[idx]['identity_label']

        try:
            image = Image.open(img_path).convert('RGB')
        except FileNotFoundError:
            print(f"Warning: File not found: {img_path}. Skipping.")
            # Return the next valid item
            return self.__getitem__((idx + 1) % len(self))
        except Exception as e:
            print(f"Warning: Error loading image {img_path}: {e}. Skipping.")
            return self.__getitem__((idx + 1) % len(self))


        if self.transform:
            image = self.transform(image)

        return image, identity_label