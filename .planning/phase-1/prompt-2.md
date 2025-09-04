# Prompt 2 — Verify dataset presence & create sample manifest

## Date / Agent run id
2025-08-29 / automatic

## Goal
Create a verification script for dataset presence and sample manifest.

## Prereqs (checked)
- [x] conda env created (user confirmed python312 active)
- [x] datasets present at required paths

## Commands run
- mkdir -p scripts\phase1 outputs\phase1\sample_images data\manifest
- python scripts\phase1\verify_datasets.py

## Files created
- scripts/phase1/verify_datasets.py
- data/manifest/phase1_samples.csv
- outputs/phase1/sample_images/coco_001.jpg through coco_010.jpg
- outputs/phase1/sample_images/stanford_dogs_001.jpg through stanford_dogs_010.jpg
- outputs/phase1/sample_images/dogfacenet_001.jpg through dogfacenet_010.jpg
- outputs/phase1/sample_images/oxford_pets_001.jpg through oxford_pets_010.jpg

## Outputs / Metrics
- COCO train2017: 118287 images
- COCO val2017: 5000 images
- COCO test2017: 40670 images
- COCO annotations: 6 JSON files
- Stanford Dogs images: 20580, annotations dir present (XML files likely)
- DogFaceNet DogFaceNet_224resized: 8363 images
- DogFaceNet DogFaceNet_alignment: 9881 images
- DogFaceNet DogFaceNet_large: 11168 images
- Oxford-IIIT Pets images: 7390, annotations dir present
- sample images: outputs/phase1/sample_images/ (40 samples total)
- manifest: E:\Code\GitHub\immich-dogs\data\manifest\phase1_samples.csv

## Issues / Errors
- Stanford Dogs and Oxford Pets annotations count as 0 JSON files (likely use XML annotation format instead)
- No actual errors - script executed successfully

## Dataset Structure Exploration

### COCO Dataset Structure
- **Subsets**: train2017 (118,287 images), val2017 (5,000 images), test2017 (40,670 images)
- **Annotations**: 6 JSON files in coco/annotations/ (Pascal VOC-style classes)
- **Format**: Object detection with bounding boxes for 80 COCO classes (dog = class 18)
- **Image naming**: 12-digit zero-padded COCO IDs (e.g., 000000032056.jpg)

### Stanford Dogs Dataset Structure
- **Organization**: Images organized by breed in WordNet synset directories (n02085620-Chihuahua)
- **Annotations**: XML files in annotation/ subdirectory per breed
- **Annotation format**: Pascal VOC XML with bounding boxes
- **Sample annotation structure** (n02085620_7.xml):
  ```xml
  <object>
    <name>Chihuahua</name>
    <bndbox>
      <xmin>71</xmin><ymin>1</ymin><xmax>192</xmax><ymax>180</ymax>
    </bndbox>
  </object>
  ```

### DogFaceNet Dataset Structure
- **Subsets**:
  - DogFaceNet_224resized/after_4_bis/ (organized by dog identity IDs 0-1180+)
  - DogFaceNet_alignment/images/
  - DogFaceNet_large/images_2/
- **Organization**: Images organized by dog identity (numeric IDs)
- **Format**: Face crops/resized images, individual identities have multiple images (e.g., 641.0.jpg, 641.1.jpg, etc.)
- **Labels**: Identity-based classification with text files:
  - `data/dogfacenet/classes_train.txt` (7666 lines, each containing dog identity ID)
  - `data/dogfacenet/classes_test.txt` (likely test set identities)
- **Sample classes_train.txt format**: Each line contains a dog identity ID, multiple images per identity

### Oxford-IIIT Pets Dataset Structure
- **Organization**: Images in images/ directory, annotations in annotations/xmls/
- **Annotations**: XML files with breed labels and bounding boxes
- **Sample annotation structure** (american_bulldog_10.xml):
  ```xml
  <object>
    <name>dog</name>
    <bndbox>
      <xmin>103</xmin><ymin>47</ymin><xmax>330</xmax><ymax>268</ymax>
    </bndbox>
  </object>
  ```
- **Labels**: Contains both cats and dogs (37 breeds total)
- **Split files**: trainval.txt, test.txt for dataset splits

### Sample Selection Analysis
**Samples appear truly random:**
- COCO: Mix of train2017 and test2017, random COCO IDs (032056, 143119, 281116, etc.)
- Stanford Dogs: Diverse breeds (collie, blenheim_spaniel, eskimo_dog, miniature_poodle, etc.)
- DogFaceNet: Across all three subdirectories, various identity IDs
- Oxford Pets: Mix of dog and cat breeds (shiba_inu, saint_bernard, wheaten_terrier, abyssinian, sphynx)

**No filtering applied** - completely random selection from available images in each dataset.

## Next steps / recommended action
- Proceed to next phase: model training setup
- Consider filtering Oxford Pets dataset to dog-only samples for dog detection task