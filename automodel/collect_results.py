"""Collect every Automodel attempt into a single expression leaderboard."""

from __future__ import annotations

import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def iter_fit_records(value):
    if isinstance(value, dict):
        required = {
            "expression",
            "tree_nodes",
            "parameters",
            "train",
            "validation",
            "validation_fitness",
        }
        if required.issubset(value):
            yield value
            return
        for child in value.values():
            yield from iter_fit_records(child)
    elif isinstance(value, list):
        for child in value:
            yield from iter_fit_records(child)


def collect() -> list[dict]:
    rows = []
    seen = set()
    for result_path in sorted(ROOT.glob("meta_*/agent_*/attempt_*/results.json")):
        parts = result_path.relative_to(ROOT).parts
        payload = json.loads(result_path.read_text(encoding="utf-8"))
        for fit in iter_fit_records(payload):
            train = fit["train"]
            validation = fit["validation"]
            key = (
                str(result_path),
                train["baseline"],
                fit["expression"],
                tuple(fit["parameters"]),
            )
            if key in seen:
                continue
            seen.add(key)
            rows.append(
                {
                    "meta": parts[0],
                    "agent": parts[1],
                    "attempt": parts[2],
                    "baseline": train["baseline"],
                    "expression": fit["expression"],
                    "parameters": json.dumps(fit["parameters"]),
                    "tree_nodes": fit["tree_nodes"],
                    "train_rho_error": train["rho_error"],
                    "train_velocity_error": train["velocity_error"],
                    "train_data_error": train["data_error"],
                    "validation_rho_error": validation["rho_error"],
                    "validation_velocity_error": validation["velocity_error"],
                    "validation_data_error": validation["data_error"],
                    "validation_fitness": fit["validation_fitness"],
                    "feasible": fit.get("feasible"),
                    "optimizer_success": fit.get("optimizer_success"),
                    "optimizer_evaluations": fit.get("optimizer_evaluations"),
                    "seed": fit.get("seed"),
                    "fit_runtime_seconds": fit.get("fit_runtime_seconds"),
                    "peak_rss_mb": max(
                        train.get("peak_rss_mb", 0), validation.get("peak_rss_mb", 0)
                    ),
                    "source": str(result_path.relative_to(ROOT)),
                }
            )
    return rows


def main() -> None:
    rows = collect()
    json_path = ROOT / "automodel" / "leaderboard.json"
    csv_path = ROOT / "automodel" / "leaderboard.csv"
    json_path.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Collected {len(rows)} fitted baseline/expression pairs.")


if __name__ == "__main__":
    main()

