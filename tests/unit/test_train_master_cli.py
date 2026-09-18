"""The CLI replaced seven one-call wrapper scripts, so dispatch is worth locking in."""

import pytest

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
