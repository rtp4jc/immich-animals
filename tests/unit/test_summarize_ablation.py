"""Tests for the ablation decision rule."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

from summarize_ablation import (  # noqa: E402
    Candidate,
    _pick_instrument,
    build_candidates,
    latency_budget,
    select_winner,
)

from animal_id.embedding.backbones import LicenseTier  # noqa: E402


def make(backbone, mrr, latency, tier=LicenseTier.PERMISSIVE, onnx=True, std=0.0):
    return Candidate(
        backbone=backbone,
        head="arcface",
        mode="finetune",
        seeds=[42],
        mrr_mean=mrr,
        mrr_std=std,
        top1_mean=mrr - 0.02,
        top1_std=0.0,
        params_m=25.0,
        latency_ms=latency,
        latency_source="ort",
        onnx_ok=onnx,
        license_tier=tier,
    )


def test_picks_highest_mrr_among_eligible():
    winner, _ = select_winner([make("a", 0.9, 20.0), make("b", 0.8, 20.0)], 30.0)
    assert winner.backbone == "a"


def test_license_gate_rejects_noncommercial_leader():
    best = make("nc", 0.99, 10.0, tier=LicenseTier.NONCOMMERCIAL)
    winner, reasons = select_winner([best, make("ok", 0.8, 10.0)], 30.0)
    assert winner.backbone == "ok"
    assert "license" in reasons[("nc", "arcface")]


def test_latency_gate_rejects_over_budget():
    winner, reasons = select_winner(
        [make("slow", 0.99, 50.0), make("ok", 0.8, 10.0)], 30.0
    )
    assert winner.backbone == "ok"
    assert "latency" in reasons[("slow", "arcface")]


def test_onnx_gate_rejects_unexportable():
    winner, reasons = select_winner(
        [make("bad", 0.99, 10.0, onnx=False), make("ok", 0.8, 10.0)], 30.0
    )
    assert winner.backbone == "ok"
    assert reasons[("bad", "arcface")] == "ONNX export failed"


def test_only_first_failed_gate_is_reported():
    # Non-permissive AND over budget: license is checked first.
    c = make("both", 0.99, 99.0, tier=LicenseTier.NONCOMMERCIAL)
    _, reasons = select_winner([c, make("ok", 0.8, 10.0)], 30.0)
    assert "license" in reasons[("both", "arcface")]
    assert "latency" not in reasons[("both", "arcface")]


def test_missing_baseline_skips_latency_gate():
    # No resnet50 row -> no budget -> still pick the best on other axes.
    assert latency_budget([make("a", 0.9, 99.0)], 1.5) is None
    winner, _ = select_winner([make("a", 0.9, 999.0), make("b", 0.8, 1.0)], None)
    assert winner.backbone == "a"


def test_overlapping_error_bars_still_pick_a_winner_but_say_so():
    winner, reasons = select_winner(
        [make("a", 0.960, 10.0, std=0.01), make("b", 0.955, 10.0, std=0.01)], 30.0
    )
    assert winner.backbone == "a"
    assert "not a decisive loss" in reasons[("b", "arcface")]


def test_clear_win_is_not_flagged_as_noise():
    _, reasons = select_winner(
        [make("a", 0.96, 10.0, std=0.001), make("b", 0.80, 10.0, std=0.001)], 30.0
    )
    assert "not a decisive loss" not in reasons[("b", "arcface")]


def test_no_eligible_candidate_returns_none():
    winner, reasons = select_winner(
        [make("nc", 0.9, 10.0, tier=LicenseTier.NONCOMMERCIAL)], 30.0
    )
    assert winner is None
    assert "license" in reasons[("nc", "arcface")]


def test_budget_is_a_multiple_of_the_baseline():
    # Synthetic values: this pins the arithmetic, not any measured latency.
    assert latency_budget([make("resnet50", 0.8, 100.0)], 1.5) == pytest.approx(150.0)


@pytest.mark.parametrize(
    "latency,expected",
    [
        (
            {
                "a": {"ort_ms": "5", "torch_ms": "9"},
                "b": {"ort_ms": "6", "torch_ms": "8"},
            },
            "ort",
        ),
        (
            {
                "a": {"ort_ms": "", "torch_ms": "9"},
                "b": {"ort_ms": "6", "torch_ms": "8"},
            },
            "torch",
        ),
        ({"a": {"ort_ms": "", "torch_ms": ""}}, "run-torch"),
        ({}, "run-torch"),
    ],
)
def test_instrument_is_the_one_every_candidate_has(latency, expected):
    """A budget compares candidates, so all must be timed the same way."""
    assert _pick_instrument(["a", "b"], latency) == expected


def test_failed_rows_are_excluded_from_candidates():
    rows = [
        {
            "backbone": "a",
            "head": "arcface",
            "mode": "finetune",
            "seed": "42",
            "mrr": "0.9",
            "top1": "0.88",
            "params_total_M": "25",
            "cpu_ms": "30",
            "onnx_ok": "True",
            "status": "ok",
        },
        {
            "backbone": "b",
            "head": "arcface",
            "mode": "finetune",
            "seed": "42",
            "mrr": "",
            "top1": "",
            "params_total_M": "",
            "cpu_ms": "",
            "onnx_ok": "",
            "status": "failed",
        },
    ]
    candidates = build_candidates(rows, "finetune", latency={})
    assert [c.backbone for c in candidates] == ["a"]


def test_seeds_are_averaged_with_a_spread():
    rows = [
        {
            "backbone": "a",
            "head": "arcface",
            "mode": "finetune",
            "seed": str(s),
            "mrr": m,
            "top1": "0.9",
            "params_total_M": "25",
            "cpu_ms": "30",
            "onnx_ok": "True",
            "status": "ok",
        }
        for s, m in [(42, "0.90"), (1, "0.94"), (2, "0.92")]
    ]
    candidate = build_candidates(rows, "finetune", latency={})[0]
    assert candidate.seeds == [1, 2, 42]
    assert candidate.mrr_mean == pytest.approx(0.92)
    assert candidate.mrr_std > 0
