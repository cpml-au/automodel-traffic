"""Run meta-3 agent 2's explicit Hodge-star search, train/validation only."""

from __future__ import annotations

import importlib.util
import json
import os
import platform
import resource
import sys
import time
from pathlib import Path

from deap import gp

from automodel.model import BASELINES
from automodel.pipeline import I80PredictionEvaluator
from automodel.search_utils import fit_candidate, is_nonlocal_feasible
from automodel.typed_primitives import build_traffic_pset


ROOT = Path(__file__).resolve().parent
BASELINE_ORDER = (
    "greenshields",
    "idm",
    "weidmann",
    "triangular",
    "del_castillo",
)
META2_FITNESS = {
    "greenshields": 7.627391,
    "idm": 5.955012,
    "weidmann": 6.322293,
    "triangular": 6.457489,
    "del_castillo": 6.603956,
}


def load_model(attempt: int):
    path = ROOT / f"attempt_{attempt}" / "model.py"
    spec = importlib.util.spec_from_file_location(f"meta3_agent2_attempt{attempt}", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def validate_grammar(models: list[object]) -> list[str]:
    """Verify repaired Hodge names and exact typed structures in the live grammar."""

    pset, _ = build_traffic_pset()
    pset.addTerminal(0.0, float, "a")
    pset.addTerminal(0.0, float, "b")
    registered = [
        name
        for name in ("St_oneP0", "St_oneP1", "St_oneD0", "St_oneD1")
        if name in pset.mapping
    ]
    if len(registered) != 4:
        raise RuntimeError(f"Incomplete repaired Hodge-star family: {registered}")
    for model in models:
        tree = gp.PrimitiveTree.from_string(model.TYPED_GP_EXPRESSION, pset)
        if len(tree) != model.TREE_NODES:
            raise RuntimeError(
                f"Attempt {model.ATTEMPT}: logged {model.TREE_NODES} nodes, "
                f"grammar compiled {len(tree)}"
            )
        gp.compile(tree, pset)
    return registered


def write_reports(attempt_dir: Path, payload: dict) -> None:
    rows = payload["results"]
    train_lines = [
        f"# Attempt {payload['attempt']} training",
        "",
        "Constants were fitted on all I80 training times 0--63. Validation was "
        "evaluated only after fitting; the test interval remained untouched.",
        "",
        "| FD | Expression | Parameters | E_rho | E_v | E_data | Evals | "
        "Optimizer | Runtime (s) | Peak RSS (MB) | Feasible |",
        "|---|---|---|---:|---:|---:|---:|---|---:|---:|---|",
    ]
    eval_lines = [
        f"# Attempt {payload['attempt']} evaluation",
        "",
        "Selection uses full I80 validation times 64--107 and "
        "`E_fitness = E_data + 0.01*tree_nodes`; lower is better.",
        "",
        "| FD | Typed GP expression | Nodes | E_rho | E_v | E_data | E_fitness | "
        "Meta-2 fitness | Change | Runtime (s) | Peak RSS (MB) | Finite | "
        "Feasible | Seed |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---:|",
    ]
    for row in rows:
        params = ", ".join(
            f"{name}={value:.10g}"
            for name, value in zip(row["parameter_names"], row["parameters"])
        )
        train = row["train"]
        validation = row["validation"]
        train_lines.append(
            f"| {row['baseline']} | `{row['expression']}` | {params} | "
            f"{train['rho_error']:.6f} | {train['velocity_error']:.6f} | "
            f"{train['data_error']:.6f} | {row['optimizer_evaluations']} | "
            f"{'success' if row['optimizer_success'] else 'stopped'}: "
            f"{row['optimizer_message'].replace('|', '/')} | "
            f"{row['fit_runtime_seconds']:.3f} | {train['peak_rss_mb']:.2f} | "
            f"{'yes' if row['feasible'] else 'no'} |"
        )
        eval_lines.append(
            f"| {row['baseline']} | `{payload['typed_gp_expression']}` | "
            f"{row['tree_nodes']} | {validation['rho_error']:.6f} | "
            f"{validation['velocity_error']:.6f} | {validation['data_error']:.6f} | "
            f"{row['validation_fitness']:.6f} | {row['meta2_fitness']:.6f} | "
            f"{row['fitness_change_from_meta2']:+.6f} | "
            f"{validation['runtime_seconds']:.3f} | {validation['peak_rss_mb']:.2f} | "
            f"{'yes' if validation['finite'] else 'no'} | "
            f"{'yes' if row['feasible'] else 'no'} | {row['seed']} |"
        )
    train_lines.extend(
        [
            "",
            "Optimizer: SciPy Powell through the shared `fit_candidate`, two "
            "deterministic restarts (zero plus seeded uniform), 45 evaluations "
            "per restart. Bounds and seeds are recorded in `results.json`.",
            "",
            "Feasibility: shared `is_nonlocal_feasible` homogeneous-state check, "
            "followed by finite full train and validation simulations.",
            "",
        ]
    )
    eval_lines.extend(
        [
            "",
            "Registered live Hodge-star names: "
            + ", ".join(f"`{name}`" for name in payload["hodge_star_registration"])
            + ".",
            "",
            "`test_evaluated = false`; no test prediction or score was computed.",
            "",
        ]
    )
    (attempt_dir / "training.md").write_text("\n".join(train_lines), encoding="utf-8")
    (attempt_dir / "evaluation.md").write_text("\n".join(eval_lines), encoding="utf-8")


def main() -> None:
    models = [load_model(attempt) for attempt in (1, 2, 3)]
    registered = validate_grammar(models)
    evaluator = I80PredictionEvaluator()
    all_rows: list[dict] = []

    for model in models:
        run_start = time.perf_counter()
        results = []
        for baseline_index, baseline_key in enumerate(BASELINE_ORDER):
            baseline = BASELINES[baseline_key]
            seed = 8100 + model.ATTEMPT * 10 + baseline_index
            fit = fit_candidate(
                evaluator=evaluator,
                baseline=baseline,
                correction=model.correction,
                expression=model.SYMBOLIC_EXPRESSION,
                tree_nodes=model.TREE_NODES,
                parameter_names=model.PARAMETER_NAMES,
                bounds=model.BOUNDS,
                seed=seed,
                restarts=2,
                max_evaluations=45,
                feasibility_check=is_nonlocal_feasible,
            )
            row = fit.to_dict()
            row["baseline_key"] = baseline_key
            row["baseline"] = baseline.name
            row["meta2_fitness"] = META2_FITNESS[baseline_key]
            row["fitness_change_from_meta2"] = (
                fit.validation_fitness - META2_FITNESS[baseline_key]
            )
            results.append(row)
            all_rows.append({"attempt": model.ATTEMPT, **row})
            print(
                f"attempt={model.ATTEMPT} baseline={baseline_key} "
                f"train={fit.train.data_error:.6f} "
                f"validation={fit.validation.data_error:.6f} "
                f"fitness={fit.validation_fitness:.6f} "
                f"delta_meta2={row['fitness_change_from_meta2']:+.6f} "
                f"feasible={fit.feasible}",
                flush=True,
            )

        payload = {
            "attempt": model.ATTEMPT,
            "family": "explicit repaired one-dimensional Hodge-star composition",
            "symbolic_expression": model.SYMBOLIC_EXPRESSION,
            "typed_gp_expression": model.TYPED_GP_EXPRESSION,
            "tree_nodes": model.TREE_NODES,
            "parameter_names": list(model.PARAMETER_NAMES),
            "hodge_star_registration": registered,
            "hodge_star_names_used": list(model.HODGE_PRIMITIVES),
            "components_and_data": {
                "fixed_components": "I80 preprocessing, boundary/initial conditions, DEC mesh, Godunov solver, FD and calibrated FD coefficients",
                "tuned_components": "multiplicative-correction constants only",
                "training": "full chronological train times 0-63 (4,800 rows)",
                "validation": "full chronological validation times 64-107 (3,300 rows)",
                "test": "not evaluated",
            },
            "optimizer": {
                "method": "scipy.optimize.minimize/Powell via automodel.search_utils.fit_candidate",
                "objective": "full training E_data",
                "bounds": [list(bound) for bound in model.BOUNDS],
                "restarts": 2,
                "max_evaluations_per_restart": 45,
                "seed_formula": "8100 + attempt*10 + baseline_index",
            },
            "selection": {
                "split": "full validation",
                "fitness": "E_data + 0.01*tree_nodes",
                "comparison": "selected meta-2 fitness per FD",
            },
            "diagnostics": {
                "feasibility_check": "automodel.search_utils.is_nonlocal_feasible",
                "reason": "mesh metric and boundary values vary for explicit DEC features",
                "all_required_fits_completed": len(results) == len(BASELINE_ORDER),
            },
            "execution": {
                "checkout": str(Path.cwd()),
                "python": sys.executable,
                "python_version": platform.python_version(),
                "jax_platform_name": os.environ.get("JAX_PLATFORMS", ""),
                "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
                "device": "CPU",
                "total_runtime_seconds": time.perf_counter() - run_start,
                "peak_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
                "test_evaluated": False,
            },
            "results": results,
        }
        attempt_dir = ROOT / f"attempt_{model.ATTEMPT}"
        (attempt_dir / "results.json").write_text(
            json.dumps(payload, indent=2) + "\n", encoding="utf-8"
        )
        write_reports(attempt_dir, payload)

    summary = [
        "# Meta-3 agent 2 summary",
        "",
        "All 15 requested Hodge-star fits completed on full train/validation data. "
        "The held-out test split was not evaluated.",
        "",
        "| FD | Best attempt | Expression | Parameters | Validation fitness | Meta-2 | Change |",
        "|---|---:|---|---|---:|---:|---:|",
    ]
    for baseline_key in BASELINE_ORDER:
        candidates = [row for row in all_rows if row["baseline_key"] == baseline_key]
        best = min(candidates, key=lambda row: row["validation_fitness"])
        params = ", ".join(
            f"{name}={value:.10g}"
            for name, value in zip(best["parameter_names"], best["parameters"])
        )
        summary.append(
            f"| {best['baseline']} | {best['attempt']} | `{best['expression']}` | "
            f"{params} | {best['validation_fitness']:.6f} | "
            f"{best['meta2_fitness']:.6f} | {best['fitness_change_from_meta2']:+.6f} |"
        )
    summary.extend(
        [
            "",
            "All tree sizes were compiled against the repaired live typed grammar. "
            "Registered variants: `St_oneP0`, `St_oneP1`, `St_oneD0`, `St_oneD1`.",
            "",
            "`test_evaluated = false`.",
            "",
        ]
    )
    (ROOT / "summary.md").write_text("\n".join(summary), encoding="utf-8")


if __name__ == "__main__":
    main()
