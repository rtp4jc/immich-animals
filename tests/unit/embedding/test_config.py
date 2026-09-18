from animal_id.embedding.config import TRAINING_CONFIG


def test_head_lr_meets_minimum_threshold():
    assert TRAINING_CONFIG.head_lr >= 1e-5


def test_backbone_lr_below_head_lr():
    assert TRAINING_CONFIG.backbone_lr < TRAINING_CONFIG.head_lr


def test_full_train_lr_at_least_backbone_lr():
    assert TRAINING_CONFIG.full_train_lr >= TRAINING_CONFIG.backbone_lr
