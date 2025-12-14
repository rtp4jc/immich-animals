from animal_id.detection.trainer import train_detector

def test_train_detector(mock_yolo_dataset, tmp_path):
    """
    Integration test for the detection training pipeline.
    Runs a single epoch on a dummy dataset to verify end-to-end execution.
    """
    # Define output directory
    project_dir = tmp_path / "runs" / "detect"
    run_name = "integration_test"
    
    # Run training
    # Note: Using imgsz=64 for speed, and device='cpu'
    results = train_detector(
        data=mock_yolo_dataset,
        epochs=1,
        imgsz=64,
        batch=2,
        device='cpu',
        project=str(project_dir),
        name=run_name,
        patience=1,
        save_period=1,
        exist_ok=True
    )
    
    # Verify results
    # 1. Check if results object is returned
    assert results is not None
    
    # 2. Check if output directory exists
    run_dir = project_dir / run_name
    assert run_dir.exists()
    
    # 3. Check for model artifacts
    # YOLO usually saves 'weights/best.pt' and 'weights/last.pt'
    weights_dir = run_dir / "weights"
    assert weights_dir.exists()
    assert (weights_dir / "last.pt").exists() or (weights_dir / "best.pt").exists()
    
    print(f"Detection integration test passed. Artifacts found in {run_dir}")
