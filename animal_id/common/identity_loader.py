"""
Utility class for loading identity validation data with optional augmentation.
"""

import json
import random
from collections import defaultdict
from typing import Dict, List, Optional

from .constants import DATA_DIR, PROJECT_ROOT


class IdentityLoader:
    """Loads and manages identity validation data with optional augmentation."""

    def __init__(self, seed: int = 42):
        """Initialize with random seed for reproducible sampling."""
        self.seed = seed
        random.seed(seed)

    def load_validation_data(
        self,
        num_images: Optional[int] = None,
        include_additional: bool = False,
        max_per_identity: Optional[int] = None,
    ) -> List[Dict[str, str]]:
        """
        Load validation data with optional additional identities.

        Args:
            num_images: Total number of images to sample (None = all)
            include_additional: Whether to include additional identities
            max_per_identity: Maximum images per identity (None = all)

        Returns:
            List of validation items with 'image_path' and 'identity_label'
        """
        # Load base validation data
        base_data = self._load_base_validation()

        if not include_additional:
            # Apply max_per_identity limit to base data if specified
            if max_per_identity is not None:
                base_data = self._limit_per_identity(base_data, max_per_identity)

            # Sample or return all
            if num_images is None:
                return base_data
            return random.sample(base_data, min(len(base_data), num_images))

        # Load additional identities
        additional_identities = self._scan_additional_identities()

        # Create augmented dataset prioritizing additional identities
        return self._create_augmented_dataset(
            base_data=base_data,
            additional_identities=additional_identities,
            num_images=num_images,
            max_per_identity=max_per_identity,
        )

    def _load_base_validation(self) -> List[Dict[str, str]]:
        """Load base validation data from identity_val.json."""
        val_json_path = DATA_DIR / "identity_val.json"

        with open(val_json_path) as f:
            val_data = json.load(f)

        # Convert to expected format
        return [
            {"image_path": item["file_path"], "identity_label": item["identity_label"]}
            for item in val_data
            if item.get("identity_label")  # Only include items with identities
        ]

    def _scan_additional_identities(self) -> Dict[str, List[str]]:
        """Scan additional_identities directory for new identities."""
        additional_dir = DATA_DIR / "additional_identities"

        if not additional_dir.exists():
            return {}

        identities = {}

        for identity_dir in additional_dir.iterdir():
            if identity_dir.is_dir():
                identity_name = identity_dir.name
                image_paths = []

                for image_file in identity_dir.iterdir():
                    if image_file.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                        # Store relative path from project root
                        rel_path = image_file.relative_to(PROJECT_ROOT)
                        image_paths.append(str(rel_path))

                if image_paths:
                    identities[identity_name] = image_paths

        return identities

    def _limit_per_identity(
        self, data: List[Dict[str, str]], max_per_identity: int
    ) -> List[Dict[str, str]]:
        """Limit number of images per identity."""
        identity_counts = defaultdict(int)
        limited_data = []

        for item in data:
            identity = item["identity_label"]
            if identity_counts[identity] < max_per_identity:
                limited_data.append(item)
                identity_counts[identity] += 1

        return limited_data

    def _create_augmented_dataset(
        self,
        base_data: List[Dict[str, str]],
        additional_identities: Dict[str, List[str]],
        num_images: Optional[int],
        max_per_identity: Optional[int],
    ) -> List[Dict[str, str]]:
        """Create augmented dataset prioritizing additional identities."""

        selected_images = []
        used_identities = set()

        # Phase 1: Add all additional identities
        for identity, image_paths in additional_identities.items():
            if max_per_identity is None:
                sampled_paths = image_paths
            else:
                sample_count = min(len(image_paths), max_per_identity)
                sampled_paths = random.sample(image_paths, sample_count)

            for image_path in sampled_paths:
                selected_images.append(
                    {"image_path": image_path, "identity_label": identity}
                )

            used_identities.add(identity)

        # Phase 2: Add base data (avoiding conflicts and respecting limits)
        base_candidates = []
        identity_counts = defaultdict(int)

        for item in base_data:
            identity = item["identity_label"]
            if identity not in used_identities:
                if (
                    max_per_identity is None
                    or identity_counts[identity] < max_per_identity
                ):
                    base_candidates.append(item)
                    identity_counts[identity] += 1

        # Add base candidates
        if num_images is None:
            # Add all base candidates
            selected_images.extend(base_candidates)
        else:
            # Fill remaining slots
            remaining_slots = num_images - len(selected_images)
            if remaining_slots > 0 and base_candidates:
                sample_count = min(len(base_candidates), remaining_slots)
                selected_images.extend(random.sample(base_candidates, sample_count))

        # Shuffle final dataset
        random.shuffle(selected_images)

        # Apply final limit if specified
        if num_images is not None:
            selected_images = selected_images[:num_images]

        return selected_images
