"""The ablation sweep runs unattended for days, so resume/skip and the failed-cell
record are the parts that must not regress."""

import csv

import pytest

from scripts import ablation_results

# Header + a row exactly as written before status/ort_cpu_ms/error existed.
LEGACY_HEADER = (
    "timestamp,backbone,head,mode,seed,epochs,img_size,n_test_queries,mrr,top1,"
    "top5,mAP,tar@1%,tar@0.1%,params_total_M,params_trainable_M,cpu_ms,onnx_ok,tag"
)
LEGACY_ROW = (
    "20260625_224320,resnet50,arcface,probe,42,default,224,1078,0.8752,0.8321,"
    "0.9295,0.4658,0.596,0.3212,25.07,1.56,37.2,True,"
)


@pytest.fixture
def legacy_csv(tmp_path):
    path = tmp_path / "results.csv"
    path.write_text(f"{LEGACY_HEADER}\n{LEGACY_ROW}\n")
    return path


def new_row(**overrides):
    row = {
        "timestamp": "20260918_120000",
        "backbone": "convnextv2_tiny",
        "head": "arcface",
        "mode": "probe",
        "seed": 42,
        "epochs": "default",
        "img_size": 224,
        "mrr": 0.95,
        "top1": 0.93,
        "top5": 0.98,
        "mAP": 0.6,
        "tar@1%": 0.81,
        "params_total_M": 28.77,
        "cpu_ms": 44.6,
        "ort_cpu_ms": 21.3,
        "onnx_ok": True,
        "status": ablation_results.STATUS_OK,
        "tag": "",
    }
    row.update(overrides)
    return row


# --- resume / skip ---------------------------------------------------------


def test_recorded_cell_is_found(legacy_csv):
    rows = ablation_results.load_rows(legacy_csv)
    assert (
        ablation_results.find_recorded(
            rows, backbone="resnet50", head="arcface", mode="probe", seed=42
        )
        is not None
    )


@pytest.mark.parametrize(
    "cell",
    [
        {"backbone": "convnextv2_tiny", "head": "arcface", "mode": "probe", "seed": 42},
        {"backbone": "resnet50", "head": "cosface", "mode": "probe", "seed": 42},
        {"backbone": "resnet50", "head": "arcface", "mode": "finetune", "seed": 42},
        {"backbone": "resnet50", "head": "arcface", "mode": "probe", "seed": 1},
    ],
)
def test_other_cells_are_not_found(legacy_csv, cell):
    rows = ablation_results.load_rows(legacy_csv)
    assert ablation_results.find_recorded(rows, **cell) is None


def test_missing_csv_records_nothing(tmp_path):
    assert ablation_results.load_rows(tmp_path / "nope.csv") == []


def test_overrides_narrow_the_match(legacy_csv):
    rows = ablation_results.load_rows(legacy_csv)
    cell = {"backbone": "resnet50", "head": "arcface", "mode": "probe", "seed": 42}
    assert ablation_results.find_recorded(rows, **cell, img_size=224) is not None
    assert ablation_results.find_recorded(rows, **cell, img_size=384) is None
    # The legacy row was run at the default epoch budget, not at 1.
    assert ablation_results.find_recorded(rows, **cell, epochs=1) is None


def test_most_recent_row_wins(legacy_csv):
    ablation_results.append_row(
        legacy_csv,
        new_row(backbone="resnet50", timestamp="20260918_130000", tag="rerun"),
    )
    rows = ablation_results.load_rows(legacy_csv)
    found = ablation_results.find_recorded(
        rows, backbone="resnet50", head="arcface", mode="probe", seed=42
    )
    assert found["tag"] == "rerun"


# --- failed cells ----------------------------------------------------------


def test_failure_row_is_recorded_and_skipped_on_resume():
    cell = {
        "timestamp": "20260918_140000",
        "backbone": "megadescriptor_l_384",
        "head": "arcface",
        "mode": "finetune",
        "seed": 1,
    }
    row = ablation_results.failure_row(cell, RuntimeError("CUDA out of memory\n  at x"))
    assert row["status"] == ablation_results.STATUS_FAILED
    assert row["error"] == "RuntimeError: CUDA out of memory at x"
    # A failure is a record, so a resume skips it instead of looping on it forever.
    assert ablation_results.find_recorded(
        [row],
        backbone="megadescriptor_l_384",
        head="arcface",
        mode="finetune",
        seed=1,
    )


def test_legacy_rows_count_as_completed(legacy_csv):
    row = ablation_results.load_rows(legacy_csv)[0]
    assert ablation_results.row_status(row) == ablation_results.STATUS_OK


def test_failed_cell_is_distinguishable_from_a_missing_one():
    failed = ablation_results.failure_row(
        {
            "timestamp": "20260918_140000",
            "backbone": "megadescriptor_l_384",
            "head": "arcface",
            "mode": "finetune",
            "seed": 1,
        },
        RuntimeError("CUDA out of memory"),
    )
    markdown = ablation_results.render_markdown([new_row(), failed])
    assert "## Failed cells" in markdown
    assert "CUDA out of memory" in markdown
    # Never in the results table itself.
    table = markdown.split("## Failed cells")[0]
    assert "megadescriptor_l_384" not in table
    assert "convnextv2_tiny" in table


def test_no_failed_section_when_everything_passed():
    assert "## Failed cells" not in ablation_results.render_markdown([new_row()])


# --- markdown with mixed old/new rows --------------------------------------


def test_markdown_renders_legacy_and_new_rows(legacy_csv):
    ablation_results.append_row(legacy_csv, new_row())
    rows = ablation_results.load_rows(legacy_csv)
    markdown = ablation_results.render_markdown(rows)

    lines = [line for line in markdown.splitlines() if line.startswith("| resnet50")]
    assert len(lines) == 1
    # Legacy rows predate the ORT column and must render as missing, not 0.
    assert f"| 37.2 | {ablation_results.MISSING} |" in lines[0]
    assert "| 44.6 | 21.3 |" in markdown


def test_markdown_sorts_finetune_first_then_by_mrr():
    rows = [
        new_row(backbone="a", mode="probe", mrr=0.99),
        new_row(backbone="b", mode="finetune", mrr=0.80),
        new_row(backbone="c", mode="finetune", mrr=0.90),
    ]
    order = [
        line.split("|")[1].strip()
        for line in ablation_results.render_markdown(rows).splitlines()
        if line.startswith("| ") and not line.startswith("| backbone")
    ]
    assert order == ["c", "b", "a"]


# --- csv schema migration --------------------------------------------------


def test_append_migrates_header_without_losing_rows(legacy_csv):
    ablation_results.append_row(legacy_csv, new_row())

    with open(legacy_csv, newline="") as f:
        reader = csv.DictReader(f)
        assert reader.fieldnames == ablation_results.CSV_FIELDS
        rows = list(reader)

    assert len(rows) == 2
    legacy, added = rows
    assert legacy["backbone"] == "resnet50"
    assert legacy["mrr"] == "0.8752"  # untouched
    assert legacy["ort_cpu_ms"] == ""  # new column, empty for old rows
    assert added["ort_cpu_ms"] == "21.3"
    assert added["status"] == ablation_results.STATUS_OK


def test_append_creates_the_file_with_a_header(tmp_path):
    path = tmp_path / "results.csv"
    ablation_results.append_row(path, new_row())
    with open(path, newline="") as f:
        assert next(csv.reader(f)) == ablation_results.CSV_FIELDS
