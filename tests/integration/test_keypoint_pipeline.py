from animal_id.keypoint.trainer import train_keypoint_model


def test_train_keypoint(mock_keypoint_dataset, tmp_path):
    """
    Integration test for the keypoint training pipeline.
    Runs a single epoch on a dummy dataset to verify end-to-end execution.
    """
    # Define output directory
    project_dir = tmp_path / "runs" / "pose"
    run_name = "integration_test"

    # Run training
    # Note: Using imgsz=64 for speed, and device='cpu'
    results = train_keypoint_model(
        data=mock_keypoint_dataset,
        epochs=1,
        imgsz=64,
        batch=2,
        device="cpu",
        project=str(project_dir),
        name=run_name,
        patience=1,
        save_period=1,
        exist_ok=True,
    )

    # Verify results
    assert results is not None

    run_dir = project_dir / run_name
    assert run_dir.exists()

    # Check for model artifacts
    weights_dir = run_dir / "weights"
    assert weights_dir.exists()
    assert (weights_dir / "last.pt").exists() or (weights_dir / "best.pt").exists()

    print(f"Keypoint integration test passed. Artifacts found in {run_dir}")
