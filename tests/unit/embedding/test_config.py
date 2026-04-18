from animal_id.embedding.config import TRAINING_CONFIG


def test_head_lr_meets_minimum_threshold():
    assert TRAINING_CONFIG["HEAD_LR"] >= 1e-5


def test_backbone_lr_below_head_lr():
    assert TRAINING_CONFIG["BACKBONE_LR"] < TRAINING_CONFIG["HEAD_LR"]


def test_full_train_lr_at_least_backbone_lr():
    assert TRAINING_CONFIG["FULL_TRAIN_LR"] >= TRAINING_CONFIG["BACKBONE_LR"]
