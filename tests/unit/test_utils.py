import pytest
import time
from pathlib import Path
from unittest.mock import patch, MagicMock
from animal_id.common.utils import find_latest_run, find_latest_timestamped_run
from animal_id.common.visualization import visualize_detection_results

def test_find_latest_run(tmp_path):
    project_dir = tmp_path / "runs"
    project_dir.mkdir()
    
    (project_dir / "test_run1").mkdir()
    (project_dir / "test_run2").mkdir()
    (project_dir / "test_run10").mkdir()
    (project_dir / "other_run5").mkdir()
    
    latest = find_latest_run(project_dir, "test_run")
    assert latest.name == "test_run10"
    
    none_found = find_latest_run(project_dir, "missing")
    assert none_found is None

def test_find_latest_timestamped_run(tmp_path):
    runs_dir = tmp_path / "runs"
    runs_dir.mkdir()
    
    dir1 = runs_dir / "run1"
    dir1.mkdir()
    
    dir2 = runs_dir / "run2"
    dir2.mkdir()
    
    # Sleep to ensure timestamp difference (filesystems vary)
    # But explicitly setting mtime is safer and faster
    # Note: pathlib doesn't expose utime easily, use os.utime
    import os
    now = time.time()
    os.utime(dir1, (now - 100, now - 100))
    os.utime(dir2, (now, now))
    
    latest = find_latest_timestamped_run(runs_dir)
    assert latest.name == "run2"

@patch('matplotlib.pyplot.savefig')
@patch('matplotlib.pyplot.show')
@patch('cv2.imread')
def test_visualize_detection_smoke(mock_imread, mock_show, mock_savefig, tmp_path):
    """Smoke test for visualization function."""
    import numpy as np
    
    # Mock image
    mock_imread.return_value = np.zeros((100, 100, 3), dtype=np.uint8)
    
    detections = [
        {'bbox': [10, 10, 50, 50], 'confidence': 0.9}
    ]
    
    visualize_detection_results(
        image_path="dummy.jpg",
        detections=detections,
        output_path=tmp_path / "out.png",
        display=True
    )
    
    # Verify plotting calls occurred
    mock_savefig.assert_called_once()
    mock_show.assert_called_once()
