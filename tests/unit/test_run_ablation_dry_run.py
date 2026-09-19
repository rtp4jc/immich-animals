"""--dry-run must print the plan without touching the GPU or the results table."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

import run_ablation  # noqa: E402

RECORDED = (
    "timestamp,backbone,head,mode,seed,epochs,img_size,n_test_queries,mrr,top1,"
    "top5,mAP,tar@1%,tar@0.1%,params_total_M,params_trainable_M,cpu_ms,onnx_ok,"
    "tag,status,ort_cpu_ms,error\n"
    "20260625_224320,resnet50,arcface,probe,42,default,224,1078,0.8752,0.8321,"
    "0.9295,0.4658,0.596,0.3212,25.07,1.56,37.2,True,,ok,41.0,\n"
)


@pytest.fixture
def results_csv(tmp_path, monkeypatch):
    path = tmp_path / "results.csv"
    path.write_text(RECORDED)
    monkeypatch.setattr(run_ablation, "RESULTS_CSV", path)
    monkeypatch.setattr(run_ablation, "RESULTS_MD", tmp_path / "results.md")
    monkeypatch.setattr(
        run_ablation,
        "run_cell",
        lambda *a, **k: pytest.fail("--dry-run must not train"),
    )
    return path


def run_main(monkeypatch, *argv):
    monkeypatch.setattr(sys, "argv", ["run_ablation.py", *argv])
    run_ablation.main()


def test_dry_run_splits_planned_from_recorded(results_csv, monkeypatch, capsys):
    run_main(
        monkeypatch,
        "--backbone",
        "resnet50",
        "convnext_tiny",
        "--mode",
        "probe",
        "--dry-run",
    )
    out = capsys.readouterr().out

    assert "WOULD SKIP backbone=resnet50" in out
    assert "WOULD RUN backbone=convnext_tiny" in out
    assert "1 of 2 cells would run, 1 already recorded" in out
    assert results_csv.read_text() == RECORDED


def test_dry_run_with_force_plans_recorded_cells_too(results_csv, monkeypatch, capsys):
    run_main(
        monkeypatch, "--backbone", "resnet50", "--mode", "probe", "--dry-run", "--force"
    )
    out = capsys.readouterr().out

    assert "WOULD RUN backbone=resnet50" in out
    assert "1 of 1 cells would run" in out
