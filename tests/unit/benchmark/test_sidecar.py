from animal_id.benchmark.sidecar import changed_detections


def test_changed_detections_lists_flips_and_measures_score_drift():
    records = [
        {"path": "a.jpg", "species": "dog", "source": "commons"},
        {"path": "b.jpg", "species": "dog", "source": "commons"},
        {"path": "n.jpg", "species": None, "source": "commons"},
    ]
    baseline = {
        "a.jpg": [{"score": 0.8}, {"score": 0.35}],
        "b.jpg": [{"score": 0.6}],
        "n.jpg": [{"score": 0.2}],
    }
    preds = {
        "a.jpg": [{"score": 0.8}, {"score": 0.25}],  # second dog fell below 0.3
        "b.jpg": [{"score": 0.65}],
        "n.jpg": [{"score": 0.31}],  # new false positive
    }

    changed, drift = changed_detections(records, preds, baseline, 0.3)

    assert changed == {"dog/commons": [("a.jpg", 2, 1)], "negative": [("n.jpg", 0, 1)]}
    assert abs(drift - 0.05) < 1e-9
