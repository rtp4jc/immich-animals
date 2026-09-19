"""The CLI replaced seven one-call wrapper scripts, so dispatch is worth locking in."""

import json
from dataclasses import replace

import pytest

from animal_id.embedding.backbones import BackboneType
from animal_id.embedding.config import DEFAULT_BACKBONE, HEAD_CONFIG, TRAINING_CONFIG
from animal_id.embedding.losses import HeadType
from scripts import train_master


@pytest.fixture
def dispatched(monkeypatch):
    """Record which command ran, without executing any pipeline."""
    calls = []
    monkeypatch.setattr(
        train_master,
        "COMMANDS",
        {
            name: (lambda args, name=name: calls.append((name, args)))
            for name in train_master.COMMANDS
        },
    )
    return calls


@pytest.mark.parametrize(
    "command",
    [
        "detection-data",
        "detection",
        "embedding-data",
        "embedding",
        "export-detector",
        "export-embedding",
        "benchmark",
        "all",
    ],
)
def test_each_subcommand_dispatches(dispatched, command):
    train_master.main([command])
    assert [name for name, _ in dispatched] == [command]


def test_no_arguments_runs_everything(dispatched):
    train_master.main([])
    assert dispatched[0][0] == "all"


def test_bare_flags_still_mean_all(dispatched):
    """`train_master.py --skip-detection` worked before the subcommands existed."""
    train_master.main(["--skip-detection", "--skip-benchmark"])
    name, args = dispatched[0]
    assert name == "all"
    assert args.skip_detection and args.skip_benchmark
    assert not args.skip_embedding


def test_benchmark_options_are_parsed(dispatched):
    train_master.main(["benchmark", "--num-images", "25", "--tag", "v2", "--no-wandb"])
    _, args = dispatched[0]
    assert args.num_images == 25
    assert args.tag == "v2"
    assert args.no_wandb


def test_every_subcommand_has_a_handler():
    parser = train_master.build_parser()
    choices = next(
        action.choices for action in parser._actions if action.dest == "command"
    )
    assert set(choices) == set(train_master.COMMANDS)


def test_unknown_subcommand_is_rejected():
    with pytest.raises(SystemExit):
        train_master.main(["not-a-command", "--nonsense-flag"])


def test_embedding_overrides_default_to_config(dispatched):
    train_master.main(["embedding"])
    _, args = dispatched[0]
    assert train_master.embedding_overrides(args) == {
        "backbone": DEFAULT_BACKBONE,
        "head": HEAD_CONFIG.head_type,
        "seed": train_master.SEED,
        "epochs": None,
    }


@pytest.mark.parametrize("command", ["embedding", "all"])
def test_embedding_overrides_are_parsed(dispatched, command):
    train_master.main(
        [
            command,
            "--backbone",
            "convnextv2_tiny",
            "--head",
            "subcenter_arcface",
            "--seed",
            "7",
            "--epochs",
            "3",
        ]
    )
    _, args = dispatched[0]
    assert train_master.embedding_overrides(args) == {
        "backbone": BackboneType.CONVNEXTV2_TINY,
        "head": HeadType.SUBCENTER_ARCFACE,
        "seed": 7,
        "epochs": 3,
    }


def test_unknown_backbone_is_rejected():
    with pytest.raises(SystemExit):
        train_master.main(["embedding", "--backbone", "not-a-backbone"])


def test_overrides_reach_the_embedding_pipeline(monkeypatch):
    calls = []
    monkeypatch.setattr(
        train_master, "run_embedding_pipeline", lambda **kwargs: calls.append(kwargs)
    )
    train_master.main(["embedding", "--backbone", "resnet50", "--seed", "1"])
    assert calls == [
        {
            "backbone": BackboneType.RESNET50,
            "head": HEAD_CONFIG.head_type,
            "seed": 1,
            "epochs": None,
        }
    ]


def test_run_config_round_trip(tmp_path):
    """Export rebuilds the trunk/head from the run dir, so the write must survive."""
    train_master.save_run_config(
        tmp_path,
        BackboneType.CONVNEXTV2_TINY,
        HeadType.SUBCENTER_ARCFACE,
        seed=7,
        training_config=replace(TRAINING_CONFIG, warmup_epochs=3, full_train_epochs=3),
    )
    saved = json.loads((tmp_path / "config.json").read_text())
    assert saved["seed"] == 7
    assert saved["training_config"]["warmup_epochs"] == 3
    assert train_master.load_run_config(tmp_path) == (
        BackboneType.CONVNEXTV2_TINY,
        HeadType.SUBCENTER_ARCFACE,
    )


def test_run_config_falls_back_to_defaults(tmp_path):
    """Pre-`--head` run dirs and missing configs must still export."""
    (tmp_path / "config.json").write_text('{"backbone": "resnet50"}')
    assert train_master.load_run_config(tmp_path) == (
        BackboneType.RESNET50,
        HEAD_CONFIG.head_type,
    )
    assert train_master.load_run_config(tmp_path / "missing") == (
        DEFAULT_BACKBONE,
        HEAD_CONFIG.head_type,
    )
