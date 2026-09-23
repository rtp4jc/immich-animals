"""Aggregate ablation cells into the decision the plan asks for.

Reads ``outputs/ablation/results.csv``, averages the per-seed cells of each
(backbone, head, mode), attaches the weight-license tier from the backbone
registry, and applies the plan's decision rule to name a shippable winner.

    uv run python scripts/summarize_ablation.py
    uv run python scripts/summarize_ablation.py --mode probe

The latency budget defaults to a multiple of the ResNet50 baseline rather than
an absolute millisecond figure, so it survives a change of measurement method
(torch vs ONNX Runtime).
"""

import argparse
import statistics
from dataclasses import dataclass
from pathlib import Path

from ablation_results import load_rows

from animal_id.common.license import LicenseTier
from animal_id.embedding.backbones import BackboneType, get_backbone_license
from animal_id.embedding.losses import HeadType

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_CSV = PROJECT_ROOT / "outputs" / "ablation" / "results.csv"
LATENCY_CSV = PROJECT_ROOT / "outputs" / "ablation" / "latency.csv"
BASELINE = "resnet50"


@dataclass
class Candidate:
    """One (backbone, head, mode) cell averaged over its seeds."""

    backbone: str
    head: str
    mode: str
    seeds: list[int]
    mrr_mean: float
    mrr_std: float
    top1_mean: float
    top1_std: float
    params_m: float
    latency_ms: float
    latency_source: str
    onnx_ok: bool
    license_tier: LicenseTier

    @property
    def key(self) -> tuple[str, str]:
        """Backbone AND head: the plan keeps those two questions separate."""
        return (self.backbone, self.head)

    @property
    def is_permissive(self) -> bool:
        return self.license_tier is LicenseTier.PERMISSIVE


def _f(value, default=0.0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def load_latency(csv_path=LATENCY_CSV) -> dict[str, dict]:
    """Per-backbone cost measured on one instrument (see measure_latency.py)."""
    path = Path(csv_path)
    if not path.exists():
        return {}
    import csv

    with open(path, newline="") as f:
        return {row["backbone"]: row for row in csv.DictReader(f)}


def _pick_instrument(backbones: list[str], latency: dict[str, dict]) -> str:
    """Use the timing every candidate has, so the gate compares like with like.

    A latency budget is a comparison, so one candidate timed under ORT and
    another under torch is not a comparison at all. ORT is what Immich runs, so
    prefer it, but only when no candidate would have to be measured differently.
    """
    if latency and all(latency.get(b, {}).get("ort_ms") for b in backbones):
        return "ort"
    if latency and all(latency.get(b, {}).get("torch_ms") for b in backbones):
        return "torch"
    return "run-torch"


def build_candidates(rows: list[dict], mode: str, latency=None) -> list[Candidate]:
    """Group ok rows of one mode into per-(backbone, head) candidates."""
    latency = load_latency() if latency is None else latency
    groups: dict[tuple[str, str], list[dict]] = {}
    for row in rows:
        if row.get("mode") != mode or (row.get("status") or "ok") != "ok":
            continue
        groups.setdefault((row["backbone"], row["head"]), []).append(row)

    instrument = _pick_instrument([b for b, _ in groups], latency)
    candidates = []
    for (backbone, head), cells in sorted(groups.items()):
        mrrs = [_f(c["mrr"]) for c in cells]
        top1s = [_f(c["top1"]) for c in cells]
        if instrument == "run-torch":
            measured = _f(cells[0]["cpu_ms"])
        else:
            measured = _f(latency[backbone][f"{instrument}_ms"])
        try:
            tier = get_backbone_license(BackboneType(backbone))
        except ValueError:
            tier = LicenseTier.ENCUMBERED
        candidates.append(
            Candidate(
                backbone=backbone,
                head=head,
                mode=mode,
                seeds=sorted(int(_f(c["seed"])) for c in cells),
                mrr_mean=statistics.fmean(mrrs),
                mrr_std=statistics.stdev(mrrs) if len(mrrs) > 1 else 0.0,
                top1_mean=statistics.fmean(top1s),
                top1_std=statistics.stdev(top1s) if len(top1s) > 1 else 0.0,
                params_m=_f(cells[0]["params_total_M"]),
                latency_ms=measured,
                latency_source=instrument,
                onnx_ok=all(c.get("onnx_ok") == "True" for c in cells),
                license_tier=tier,
            )
        )
    return sorted(candidates, key=lambda c: -c.mrr_mean)


def best_head(candidates: list[Candidate], backbone: str) -> str:
    """The head to ship for one backbone, preferring the simpler one on a tie.

    A plain max() always returns a winner even when the gap is noise, so an
    improvement must clear the combined seed spread to displace arcface.
    """
    for_backbone = [c for c in candidates if c.backbone == backbone]
    if not for_backbone:
        return HeadType.ARCFACE.value
    baseline = next((c for c in for_backbone if c.head == HeadType.ARCFACE.value), None)
    if baseline is None:
        return max(for_backbone, key=lambda c: c.mrr_mean).head
    best = max(for_backbone, key=lambda c: c.mrr_mean)
    if best is baseline:
        return baseline.head
    return (
        best.head
        if best.mrr_mean - baseline.mrr_mean > best.mrr_std + baseline.mrr_std
        else baseline.head
    )


def latency_budget(candidates: list[Candidate], multiple: float) -> float | None:
    """Budget = ``multiple`` x the ResNet50 baseline's measured latency."""
    for candidate in candidates:
        if candidate.backbone == BASELINE:
            return candidate.latency_ms * multiple
    return None


def select_winner(
    candidates: list[Candidate], budget_ms: float | None
) -> tuple[Candidate | None, dict[str, str]]:
    """Apply the plan's decision rule and explain every rejection.

    Returns ``(winner_or_None, {backbone: reason_it_lost})``. The reason string
    is what lands in the findings doc, so it must say *which* gate rejected a
    candidate, not merely that it lost.
    """
    reasons: dict[tuple[str, str], str] = {}
    eligible = []
    for c in candidates:
        # Gates in plan order; only the first failure is reported.
        if not c.is_permissive:
            reasons[c.key] = f"license {c.license_tier.value} - ceiling only"
        elif not c.onnx_ok:
            reasons[c.key] = "ONNX export failed"
        elif budget_ms is not None and c.latency_ms > budget_ms:
            reasons[c.key] = f"latency {c.latency_ms:.1f}ms > {budget_ms:.1f}ms budget"
        else:
            eligible.append(c)

    if not eligible:
        return None, reasons

    winner = max(eligible, key=lambda c: c.mrr_mean)
    for c in eligible:
        if c is winner:
            continue
        gap = winner.mrr_mean - c.mrr_mean
        noise = winner.mrr_std + c.mrr_std
        reason = f"MRR {gap:.3f} below {winner.backbone}/{winner.head}"
        # A gap inside the seed spread ranks but does not separate.
        if gap <= noise:
            reason += f" (within seed noise +/-{noise:.3f}; not a decisive loss)"
        reasons[c.key] = reason
    return winner, reasons


def render(candidates: list[Candidate], budget_ms, winner, reasons) -> str:
    lines = [
        "| backbone | license | seeds | MRR | Top-1 | params(M) | "
        "latency(ms) | ONNX | verdict |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for c in candidates:
        err = f" ± {c.mrr_std:.3f}" if c.mrr_std else ""
        verdict = "🏆 winner" if winner and c is winner else reasons.get(c.key, "")
        lines.append(
            f"| {c.backbone} | {c.head} | {c.license_tier.value} | {len(c.seeds)} | "
            f"{c.mrr_mean:.3f}{err} | {c.top1_mean:.3f} | {c.params_m:.1f} | "
            f"{c.latency_ms:.1f} | {'✅' if c.onnx_ok else '❌'} | {verdict} |"
        )
    if budget_ms:
        source = candidates[0].latency_source if candidates else "?"
        lines.append(f"\nLatency budget: {budget_ms:.1f} ms (measured: {source})")
    else:
        lines.append(f"\nNo {BASELINE} cell yet - latency gate skipped.")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", default="finetune", choices=["probe", "finetune"])
    parser.add_argument(
        "--best-head",
        metavar="BACKBONE",
        help="Print the head to ship for one backbone, then exit.",
    )
    parser.add_argument(
        "--latency-multiple",
        type=float,
        default=1.5,
        help="Latency ceiling as a multiple of the ResNet50 baseline (default 1.5).",
    )
    args = parser.parse_args()

    rows = load_rows(RESULTS_CSV)
    candidates = build_candidates(rows, args.mode)

    if args.best_head:
        print(best_head(candidates, args.best_head))
        return

    if not candidates:
        print(f"No completed {args.mode} cells in {RESULTS_CSV}")
        return

    budget = latency_budget(candidates, args.latency_multiple)
    winner, reasons = select_winner(candidates, budget)
    print(render(candidates, budget, winner, reasons))
    print()
    if winner:
        print(f"Shippable winner: {winner.backbone} (MRR {winner.mrr_mean:.3f})")
    else:
        print("No candidate passes every gate.")


if __name__ == "__main__":
    main()
