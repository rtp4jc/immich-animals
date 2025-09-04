import os
import json
import random
from collections import defaultdict
import xml.etree.ElementTree as ET

# --- Configuration ---
DOGFACENET_PATH = 'data/dogfacenet/DogFaceNet_224resized/after_4_bis'
STANFORD_DOGS_PATH = 'data/stanford_dogs'
OUTPUT_TRAIN_JSON = 'data/identity_train.json'
OUTPUT_VAL_JSON = 'data/identity_val.json'
VAL_SPLIT_RATIO = 0.15
MIN_IMAGES_PER_IDENTITY = 5 # Only include identities with at least this many images

def get_breed_map():
    """Scans the Stanford Dogs dataset annotations to map image filenames to breeds."""
    breed_map = {}
    annotations_path = os.path.join(STANFORD_DOGS_PATH, 'Annotation')
    if not os.path.exists(annotations_path):
        print(f"Warning: Stanford Dogs annotation path not found at {annotations_path}")
        return {}

    for breed_folder in os.listdir(annotations_path):
        breed_path = os.path.join(annotations_path, breed_folder)
        for annotation_file in os.listdir(breed_path):
            tree = ET.parse(os.path.join(breed_path, annotation_file))
            root = tree.getroot()
            filename = root.find('filename').text
            breed = root.find('object').find('name').text
            # Use a simplified key to match DogFaceNet which might not have the .jpg
            image_key = os.path.splitext(filename)[0]
            breed_map[image_key] = breed
    return breed_map

def create_identity_dataset():
    """
    Scans DogFaceNet for identities and Stanford Dogs for breed labels,
    then creates train/validation JSON files.
    """
    if not os.path.exists(DOGFACENET_PATH):
        print(f"Error: DogFaceNet path not found at {DOGFACENET_PATH}")
        print("Please download the dataset.")
        return

    print("Scanning Stanford Dogs for breed information...")
    breed_map = get_breed_map()
    print(f"Found {len(breed_map)} breed annotations.")

    print("Scanning DogFaceNet for identities...")
    identity_images = defaultdict(list)
    # DogFaceNet identities are the folder names
    identity_folders = sorted(os.listdir(DOGFACENET_PATH))
    
    for identity_id, identity_name in enumerate(identity_folders):
        identity_path = os.path.join(DOGFACENET_PATH, identity_name)
        if not os.path.isdir(identity_path):
            continue
        
        for image_file in os.listdir(identity_path):
            if image_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                file_path = os.path.join(identity_path, image_file)
                identity_images[identity_id].append(file_path)

    print(f"Found {len(identity_images)} total identities.")

    # Filter out identities with too few images
    filtered_identities = {
        identity_id: paths
        for identity_id, paths in identity_images.items()
        if len(paths) >= MIN_IMAGES_PER_IDENTITY
    }
    print(f"Filtered to {len(filtered_identities)} identities with >= {MIN_IMAGES_PER_IDENTITY} images.")

    # Create final data list and assign labels
    all_data = []
    # Create a stable mapping from original identity_id to a new contiguous one
    identity_id_map = {old_id: new_id for new_id, old_id in enumerate(filtered_identities.keys())}

    for old_identity_id, image_paths in filtered_identities.items():
        new_identity_id = identity_id_map[old_identity_id]
        for path in image_paths:
            image_key = os.path.splitext(os.path.basename(path))[0]
            # Stanford breed names often have n0..._ prefix, DogFaceNet doesn't
            lookup_key = '_'.join(image_key.split('_')[-2:]) if '_' in image_key else image_key
            breed = breed_map.get(lookup_key, 'unknown')
            
            all_data.append({
                'file_path': path.replace('\\', '/'),
                'identity_label': new_identity_id,
                'breed_label': breed
            })

    # Split data into training and validation sets
    random.shuffle(all_data)
    split_idx = int(len(all_data) * (1 - VAL_SPLIT_RATIO))
    train_data = all_data[:split_idx]
    val_data = all_data[split_idx:]

    # Save to JSON
    print(f"Writing {len(train_data)} training samples to {OUTPUT_TRAIN_JSON}")
    with open(OUTPUT_TRAIN_JSON, 'w') as f:
        json.dump(train_data, f, indent=4)

    print(f"Writing {len(val_data)} validation samples to {OUTPUT_VAL_JSON}")
    with open(OUTPUT_VAL_JSON, 'w') as f:
        json.dump(val_data, f, indent=4)

    print("Dataset preparation complete.")

if __name__ == '__main__':
    create_identity_dataset()