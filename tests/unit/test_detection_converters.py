import pytest
import json
from animal_id.detection.dataset_converter import CocoDetectorDatasetConverter

@pytest.fixture
def mock_converter_env(tmp_path):
    """Setup mock directories for source data."""
    root = tmp_path / "data"
    root.mkdir()
    
    stanford = root / "stanford_dogs"
    (stanford / "annotation" / "n02085620-Chihuahua").mkdir(parents=True)
    
    xml_content = """
    <annotation>
        <folder>n02085620-Chihuahua</folder>
        <filename>n02085620_10074</filename>
        <size>
            <width>333</width>
            <height>500</height>
            <depth>3</depth>
        </size>
        <object>
            <name>dog</name>
            <bndbox>
                <xmin>25</xmin>
                <ymin>10</ymin>
                <xmax>276</xmax>
                <ymax>498</ymax>
            </bndbox>
        </object>
    </annotation>
    """
    with open(stanford / "annotation" / "n02085620-Chihuahua" / "n02085620_10074", "w") as f:
        f.write(xml_content)

    coco = root / "coco" / "annotations"
    coco.mkdir(parents=True)
    
    coco_data = {
        "images": [
            {"id": 1, "file_name": "001.jpg", "width": 640, "height": 480},
            {"id": 2, "file_name": "002.jpg", "width": 640, "height": 480}
        ],
        "categories": [
            {"id": 1, "name": "person"},
            {"id": 18, "name": "dog"}
        ],
        "annotations": [
            {"id": 101, "image_id": 1, "category_id": 18, "bbox": [100, 100, 50, 50], "iscrowd": 0},
            {"id": 102, "image_id": 2, "category_id": 1, "bbox": [0, 0, 10, 10], "iscrowd": 0}
        ]
    }
    with open(coco / "instances_train2017.json", "w") as f:
        json.dump(coco_data, f)

    output = root / "output"
    
    config = {
        'stanford_base_dir': str(stanford),
        'coco_train_json': str(coco / "instances_train2017.json"),
        'coco_val_json': str(coco / "instances_train2017.json"), # Reusing same for test
        'oxford_xml_dir': str(root / "oxford"), # Empty
        'output_dir': str(output)
    }
    
    return config

def test_stanford_parsing(mock_converter_env):
    converter = CocoDetectorDatasetConverter(mock_converter_env)
    
    images, anns, _ = converter._load_stanford_base_bboxes(
        mock_converter_env['stanford_base_dir'], set(), 0
    )
    
    assert len(images) == 1
    assert images[0]['width'] == 333
    assert len(anns) == 1
    assert anns[0]['bbox'] == [25.0, 10.0, 251.0, 488.0] # x2-x1, y2-y1

def test_coco_parsing(mock_converter_env):
    converter = CocoDetectorDatasetConverter(mock_converter_env)
    
    # Load dog images only
    pos_imgs, pos_anns, neg_imgs, _ = converter._load_coco_bbox_only(
        mock_converter_env['coco_train_json'], "split", 0, 0
    )
    
    # Should find image 1 (dog), ignore image 2 (person)
    assert len(pos_imgs) == 1
    assert pos_imgs[0]['file_name'] == "coco/images/split/001.jpg"
    assert len(pos_anns) == 1
    assert pos_anns[0]['bbox'] == [100.0, 100.0, 50.0, 50.0]
