import numpy as np
import jax.numpy as jnp
from jax import vmap
from dctkit.dec import cochain as C
from dctkit.dec.flat import flat
from dctkit import config
from sr_traffic.data.data import preprocess_data, build_dataset
from sr_traffic.fd import diagrams as tf_utils
from sr_traffic.utils import flat as tf_flat
from sr_traffic.utils.godunov import godunov_solver
from sr_traffic.utils.metrics import relative_tts_error
from dctkit.mesh.simplex import SimplicialComplex
from functools import partial
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as patches
import numpy.typing as npt
from scipy.stats import rankdata
from typing import Dict, Callable
import argparse
from pathlib import Path

config()

RESULTS_ROOT = Path(__file__).resolve().parents[3] / "results"


def model_plot_title(name: str) -> str:
    """Wrap the Automodel prefix so long model names remain readable."""

    return name.replace("automodel-", "automodel-\n", 1)


def rescale_rho_v_f(
    rhoP0: npt.NDArray,
    rho: npt.NDArray,
    v: npt.NDArray,
    f: npt.NDArray,
    data_info: Dict,
):
    return (
        rhoP0 * data_info["density_max"],
        rho * data_info["density_max"],
        v * data_info["V"],
        f * data_info["density_max"] * data_info["V"],
    )


def compute_errors(
    true_rho: npt.NDArray,
    true_v: npt.NDArray,
    true_f: npt.NDArray,
    model_rho: npt.NDArray,
    model_v: npt.NDArray,
    model_f: npt.NDArray,
):
    rho_err = jnp.sqrt(jnp.sum((true_rho - model_rho) ** 2)) / jnp.sqrt(
        jnp.sum(true_rho**2)
    )
    v_err = jnp.sqrt(jnp.sum((true_v - model_v) ** 2)) / jnp.sqrt(jnp.sum(true_v**2))
    f_err = jnp.sqrt(jnp.sum((true_f - model_f) ** 2)) / jnp.sqrt(jnp.sum(true_f**2))
    return rho_err, v_err, f_err


def simulate_model(
    flux_fn: Callable,
    flux_der_fn: Callable,
    data_info: Dict,
    S: SimplicialComplex,
    flats: Dict,
    step: int,
    rho_bnd_array: npt.NDArray,
):
    rho, v, f = godunov_solver(
        data_info["rho_0"],
        S,
        rho_bnd_array,
        flux_fn,
        flux_der_fn,
        data_info["delta_t_refined"],
        0,
        flats,
        data_info["num_t_points"],
    )
    flat_rho = C.CochainD1(S, flats["flat_left_v"](rho.T)[:, :, 0].T)
    rho_computedP0 = C.star(flat_rho).coeffs
    v_comp = v[:, ::step]
    f_comp = f[:, ::step]
    v_true = data_info["v"]
    f_true = data_info["flow"]
    v_comp = v_comp.at[:, 0].set(v_true[:, 0])
    v_comp = v_comp.at[0, :].set(v_true[0, :])
    v_comp = v_comp.at[-3:, :].set(v_true[-3:, :])

    f_comp = f_comp.at[:, 0].set(f_true[:, 0])
    f_comp = f_comp.at[0, :].set(f_true[0, :])
    f_comp = f_comp.at[-3:, :].set(f_true[-3:, :])
    return rho, rho_computedP0, v_comp, f_comp


def plot_diagrams(
    results: Dict,
    rhoP0: npt.NDArray,
    v: npt.NDArray,
    f: npt.NDArray,
    name_diagram: str,
    test_name: str,
    train_idx: npt.NDArray,
    test_idx: npt.NDArray,
    task: str,
):

    if name_diagram == "velocity":
        diagram = v
        diagram_idx = "v"
    elif name_diagram == "flux":
        diagram = f
        diagram_idx = "f"
    if task == "prediction":
        train_idx_slice = (slice(None), train_idx)
        test_idx_slice = (slice(None), test_idx)
    elif task == "reconstruction":
        train_idx_slice = (train_idx, slice(None))
        test_idx_slice = (test_idx, slice(None))
    models_names = list(results.keys())
    num_models = len(models_names)

    fig_dim = (3 * num_models, 4)
    fig, axes = plt.subplots(1, num_models, figsize=fig_dim)
    for i in range(num_models):
        name = models_names[i]
        axes[i].scatter(
            results[name]["rhoP0"][1:-3, 1:].flatten(),
            results[name][diagram_idx][1:-3, 1:].flatten(),
            marker=".",
            s=5,
            label="Model",
            c="#ff0000",
            zorder=1,
        )
        axes[i].scatter(
            rhoP0[train_idx_slice].flatten(),
            diagram[train_idx_slice].flatten(),
            marker=".",
            s=5,
            label="Training Data",
            c="#4757fb",
            zorder=0,
        )
        axes[i].scatter(
            rhoP0[test_idx_slice].flatten(),
            diagram[test_idx_slice].flatten(),
            marker=".",
            s=5,
            label="Test Data",
            c="#0ea4f0",
            zorder=0,
        )
        axes[i].set_xlabel(r"$\rho$ (veh/ft)")
        axes[i].set_ylabel(r"$\rho\,V(\rho)$ (veh/s)")
        axes[i].set_yticks([0, 1, 2, 3])
        axes[i].set_title(model_plot_title(name))

    handles, labels = axes[i].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        bbox_to_anchor=(0.63, 1.15),
        ncol=3,
        fancybox=True,
        shadow=True,
        fontsize=18,
        markerscale=3,
    )
    plt.tight_layout()
    plt.savefig(
        RESULTS_DIR / f"{name_diagram}_{test_name}.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.clf()


def make_rect(xy: npt.NDArray, width: float, height: float, color: str):
    return patches.Rectangle(
        xy,
        width,
        height,
        linewidth=2,
        edgecolor=color,
        facecolor="none",
        clip_on=False,
        zorder=10,
    )


def rho_v_plot(
    results,
    data_info,
    v,
    x_sampled_circ,
    test_name,
    x_ticks,
    y_ticks,
    cb_ticks,
    road_name,
    task,
):
    models_names = list(results.keys())
    num_models = len(models_names)
    data_list = [data_info["density"], v]
    data_names = ["rho", "v"]
    cbar_names = [r"$\rho$ (veh/ft)", r"$v$ (ft/s)"]

    fig, axes = plt.subplots(
        2, 1 + num_models, figsize=(3.4 * (1 + num_models), 5.4)
    )

    x_mesh, t_mesh = np.meshgrid(x_sampled_circ[:-3], data_info["t_sampled_circ"])

    cmap = "rainbow"

    if task == "prediction":
        rect_0_train = make_rect((2.5, 10), 535.0, 1500.0, "red")
        rect_1_train = make_rect((2.5, 10), 535.0, 1500.0, "red")
        rect_0_test = make_rect((542.5, 10), 355, 1500.0, "#FF7F50")
        rect_1_test = make_rect((542.5, 10), 355, 1500.0, "#FF7F50")

        rect_train = [rect_0_train, rect_1_train]
        rect_test = [rect_0_test, rect_1_test]
    elif task == "reconstruction":
        if road_name == "I80":
            x_idx = [
                50.0,
                310.0,
                430.0,
                610.0,
                770.0,
                1050.0,
                1210.0,
                1330.0,
                1430.0,
            ]
            x_magn = [40.0, 20.0, 80.0, 20.0, 20.0, 40.0, 40.0, 20.0, 40.0]
            t_start_end = [2.5, 895.0]
        elif road_name == "US101":
            x_idx = [
                10.0,
                50.0,
                310.0,
                430.0,
                610.0,
                770.0,
                1050.0,
                1230.0,
                1450.0,
                1510.0,
                1670.0,
                1750.0,
                1810.0,
            ]
            x_magn = [
                20.0,
                40.0,
                20.0,
                60.0,
                20.0,
                20.0,
                40.0,
                20.0,
                20.0,
                20.0,
                20.0,
                40.0,
                60.0,
            ]
            t_start_end = [2.5, 2695]

    for i, data_entry in enumerate(data_list):
        vmin = np.min(data_entry.T)
        vmax = np.max(data_entry.T)

        norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

        data_plot = axes[i, 0].contourf(
            t_mesh, x_mesh, data_entry[:-3].T, levels=100, cmap=cmap, norm=norm
        )
        for j in range(num_models):
            axes[i, j + 1].contourf(
                t_mesh,
                x_mesh,
                results[models_names[j]][data_names[i]][:-3].T,
                levels=100,
                cmap=cmap,
                norm=norm,
            )
            # Title
            if i == 0:
                if task == "prediction":
                    axes[i, 0].add_patch(rect_train[i])
                    axes[i, 0].add_patch(rect_test[i])
                elif task == "reconstruction":
                    for k in range(len(x_idx)):
                        rect = make_rect(
                            (t_start_end[0], x_idx[k]),
                            t_start_end[1],
                            x_magn[k],
                            "#FF00FF",
                        )
                        axes[i, 0].add_patch(rect)

                axes[i, 0].set_title("Data")
                axes[i, j + 1].set_title(model_plot_title(models_names[j]))

        # Add one colorbar per row
        fig.colorbar(
            data_plot,
            ax=axes[i, :],
            orientation="vertical",
            fraction=0.05,
            pad=0.01,
            label=cbar_names[i],
            ticks=cb_ticks[i],
        )

    # Axis labels
    for i in range(2):
        axes[i, 0].set_ylabel("x (ft)")
    for j in range(num_models + 1):
        axes[-1, j].set_xlabel("t (s)")

    for i in range(2):
        for j in range(num_models + 1):
            axes[i, j].set_xticks([])
            axes[i, j].set_yticks([])
    # Axis ticks
    for i in range(2):
        axes[i, 0].set_yticks(y_ticks)
    for j in range(num_models + 1):
        axes[-1, j].set_xticks(x_ticks)

    plt.savefig(
        RESULTS_DIR / f"rho_v_f_plot_{test_name}.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.clf()


def predicted_true_plots(results: Dict, v: npt.NDArray, f: npt.NDArray, test_name: str):
    models_names = list(results.keys())
    num_models = len(models_names)
    fig_dim = (3 * num_models, num_models)
    _, axes = plt.subplots(1, num_models, figsize=fig_dim)
    for i in range(num_models):
        axes[i].scatter(
            f.flatten(),
            results[models_names[i]]["f"].flatten(),
            marker=".",
            s=5,
            c="#0ea4f0",
        )
        axes[i].set_aspect("equal")
        axes[i].scatter(
            f.flatten(),
            f.flatten(),
            marker=".",
            s=5,
            c="#ff0000",
        )
        axes[i].set_xlabel(r"Flux true")
        axes[i].set_ylabel(r"Flux predicted")
        axes[i].set_title(model_plot_title(models_names[i]))
    plt.tight_layout()
    plt.savefig(
        RESULTS_DIR / f"pred_actual_flux_{test_name}.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.clf()

    fig_dim = (3 * num_models, num_models)
    _, axes = plt.subplots(1, num_models, figsize=fig_dim)
    for i in range(num_models):
        axes[i].scatter(
            v.flatten(),
            results[models_names[i]]["v"].flatten(),
            marker=".",
            s=5,
            c="#0ea4f0",
        )
        axes[i].set_aspect("equal")
        axes[i].scatter(
            v.flatten(),
            v.flatten(),
            marker=".",
            s=5,
            c="#ff0000",
        )
        axes[i].set_xlabel(r"Velocity true")
        axes[i].set_ylabel(r"Velocity predicted")
        axes[i].set_title(model_plot_title(models_names[i]))
    plt.tight_layout()
    plt.savefig(
        RESULTS_DIR / f"pred_actual_velocity_{test_name}.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.clf()


# Highlight best values
def format_entry(val: float, rank: float, is_best: bool):
    formatted = f"{val:.3f} ({rank:g})"
    return f"\\textbf{{{formatted}}}" if is_best else formatted


def format_markdown_entry(val: float, rank: float, is_best: bool):
    formatted = f"{val:.3f} ({rank:g})"
    return f"**{formatted}**" if is_best else formatted


def save_error_tables(
    results: Dict,
    test_idx: npt.NDArray,
    road_name: str,
    task: str,
    t_vec: npt.NDArray,
    x_vec: npt.NDArray,
    output_path: Path,
    reference_results: Dict | None = None,
):
    """Save held-out errors, test-score ranks, baseline gains, and TTS errors."""

    if task == "prediction":
        test_slice = (slice(1, -3), test_idx)
        tts_slice = (slice(None), test_idx)
        tts_t = t_vec[test_idx]
        tts_x = x_vec
    elif task == "reconstruction":
        test_slice = (test_idx, slice(None))
        tts_slice = test_slice
        tts_t = t_vec
        tts_x = x_vec[test_idx]
    else:
        raise ValueError(f"Unsupported task: {task}")

    models_for_metrics = dict(reference_results or {}) | results
    metrics = {}
    for name, model in models_for_metrics.items():
        e_rho_test, e_v_test, _ = compute_errors(
            data_info["density"][test_slice],
            v[test_slice],
            f[test_slice],
            model["rho"][test_slice],
            model["v"][test_slice],
            model["f"][test_slice],
        )
        e_data_test = 50.0 * (e_rho_test**2 + e_v_test**2)
        e_tts_test = relative_tts_error(
            np.asarray(data_info["density"])[tts_slice],
            np.asarray(model["rho"])[tts_slice],
            np.asarray(tts_t),
            np.asarray(tts_x),
        )
        metrics[name] = (
            float(e_rho_test),
            float(e_v_test),
            float(e_data_test),
            float(e_tts_test),
        )

    def family_name(name: str) -> str:
        for prefix in ("automodel-", "SR-"):
            if name.startswith(prefix):
                return name.removeprefix(prefix)
        return name

    baseline_scores = {
        name: values[2]
        for name, values in metrics.items()
        if family_name(name) == name
    }
    rows = [(name, *metrics[name]) for name in results]
    ranks = rankdata([row[3] for row in rows], method="average")
    best_rho = min(row[1] for row in rows)
    best_v = min(row[2] for row in rows)
    best_data = min(row[3] for row in rows)
    best_tts = min(row[4] for row in rows)

    markdown_rows = []
    latex_rows = []
    for i, (name, e_rho_test, e_v_test, e_data_test, e_tts_test) in enumerate(rows):
        baseline_name = family_name(name)
        if name == baseline_name:
            improvement_markdown = "—"
            improvement_latex = "--"
        else:
            improvement = 100.0 * (
                baseline_scores[baseline_name] - e_data_test
            ) / baseline_scores[baseline_name]
            improvement_markdown = f"{improvement:+.2f}%"
            improvement_latex = f"{improvement:+.2f}\\%"

        markdown_row = [
            name,
            f"**{e_rho_test:.3f}**" if e_rho_test == best_rho else f"{e_rho_test:.3f}",
            f"**{e_v_test:.3f}**" if e_v_test == best_v else f"{e_v_test:.3f}",
            format_markdown_entry(e_data_test, ranks[i], e_data_test == best_data),
            improvement_markdown,
            f"**{e_tts_test:.3f}**" if e_tts_test == best_tts else f"{e_tts_test:.3f}",
        ]
        latex_row = [
            name,
            f"\\textbf{{{e_rho_test:.3f}}}" if e_rho_test == best_rho else f"{e_rho_test:.3f}",
            f"\\textbf{{{e_v_test:.3f}}}" if e_v_test == best_v else f"{e_v_test:.3f}",
            format_entry(e_data_test, ranks[i], e_data_test == best_data),
            improvement_latex,
            f"\\textbf{{{e_tts_test:.3f}}}" if e_tts_test == best_tts else f"{e_tts_test:.3f}",
        ]
        markdown_rows.append(markdown_row)
        latex_rows.append(latex_row)

    caption = (
        "Held-out relative density, velocity, and total-travel-time errors for "
        f"the {road_name} {task} task. Models are ranked only by test E_data; "
        "positive improvement means lower E_data than the corresponding baseline."
    )
    label = f"tab:{output_path.stem}_{road_name.lower()}_{task}"

    table = (
        r"""\begin{table}[H]
        \caption{"""
        + caption
        + r"""}
        \begin{center}
            \begin{tabular}{c c c c c c}
                \toprule
                Model & $E^{\text{ts}}_\rho$ & $E^{\text{ts}}_v$ & $E^{\text{ts}}_{data}$ (rank) & vs. own baseline & $E^{\text{ts}}_{TTS}$\\
                \midrule
    """
        + "\n".join(["            " + " & ".join(row) + r"\\" for row in latex_rows])
        + r"""
                \bottomrule
            \end{tabular}
        \end{center}
        \label{"""
        + label
        + r"""}
    \end{table}
"""
    )

    output_path.write_text(table, encoding="utf-8")
    markdown_table = "\n".join(
        [
            f"## Relative errors — {road_name} {task}",
            "",
            caption,
            "",
            "| Model | $E^{\\mathrm{ts}}_\\rho$ | $E^{\\mathrm{ts}}_v$ | $E^{\\mathrm{ts}}_{data}$ (rank) | vs. own baseline | $E^{\\mathrm{ts}}_{TTS}$ |",
            "|---|---:|---:|---:|---:|---:|",
            *["| " + " | ".join(row) + " |" for row in markdown_rows],
            "",
        ]
    )
    output_path.with_suffix(".md").write_text(markdown_table, encoding="utf-8")


def save_score_tables(
    results: Dict,
    train_idx: npt.NDArray,
    test_idx: npt.NDArray,
    road_name: str,
    task: str,
    output_path: Path,
):
    """Save the README-defined unpenalized ``E_data`` for every model."""

    if task == "prediction":
        # Match ``solve`` and the README score exactly: the first and last
        # three rows are boundary/ghost locations and are not scored.
        train_slice = (slice(1, -3), train_idx)
        test_slice = (slice(1, -3), test_idx)
    else:
        train_slice = (train_idx, slice(None))
        test_slice = (test_idx, slice(None))

    rows = []
    for name, model in results.items():
        rho_train, v_train, _ = compute_errors(
            data_info["density"][train_slice],
            v[train_slice],
            f[train_slice],
            model["rho"][train_slice],
            model["v"][train_slice],
            model["f"][train_slice],
        )
        rho_test, v_test, _ = compute_errors(
            data_info["density"][test_slice],
            v[test_slice],
            f[test_slice],
            model["rho"][test_slice],
            model["v"][test_slice],
            model["f"][test_slice],
        )
        rows.append(
            (
                name,
                float(50.0 * (rho_train**2 + v_train**2)),
                float(50.0 * (rho_test**2 + v_test**2)),
            )
        )

    scores = np.asarray([[row[1], row[2]] for row in rows])
    ranks = np.argsort(np.argsort(scores, axis=0), axis=0) + 1
    markdown_rows = []
    latex_rows = []
    for i, (name, train_score, test_score) in enumerate(rows):
        markdown_rows.append(
            [
                name,
                format_markdown_entry(train_score, ranks[i, 0], ranks[i, 0] == 1),
                format_markdown_entry(test_score, ranks[i, 1], ranks[i, 1] == 1),
            ]
        )
        latex_rows.append(
            [
                name,
                format_entry(train_score, ranks[i, 0], ranks[i, 0] == 1),
                format_entry(test_score, ranks[i, 1], ranks[i, 1] == 1),
            ]
        )

    caption = (
        "Unpenalized data error on the training and test partitions for "
        f"the {road_name} {task} task. Lower is better."
    )
    markdown = "\n".join(
        [
            f"## Data-error scores — {road_name} {task}",
            "",
            caption,
            "",
            "| Model | Training $E_{data}$ | Test $E_{data}$ |",
            "|---|---:|---:|",
            *["| " + " | ".join(row) + " |" for row in markdown_rows],
            "",
        ]
    )
    output_path.with_suffix(".md").write_text(markdown, encoding="utf-8")

    latex = (
        "\\begin{table}[H]\n"
        f"\\caption{{{caption}}}\n"
        "\\begin{center}\n"
        "\\begin{tabular}{c c c}\n"
        "\\toprule\n"
        "Model & $E^{\\mathrm{tr}}_{data}$ & $E^{\\mathrm{ts}}_{data}$\\\\\n"
        "\\midrule\n"
        + "\n".join(" & ".join(row) + r"\\" for row in latex_rows)
        + "\n\\bottomrule\n"
        "\\end{tabular}\n"
        "\\end{center}\n"
        f"\\label{{tab:{output_path.stem}_{road_name.lower()}_{task}}}\n"
        "\\end{table}\n"
    )
    output_path.write_text(latex, encoding="utf-8")


parser = argparse.ArgumentParser()
parser.add_argument("--road_name", type=str, required=True)
parser.add_argument("--task", type=str, required=True)

args = parser.parse_args()

road_name = args.road_name
task = args.task
test_name = f"{road_name}_{task}"
RESULTS_DIR = RESULTS_ROOT / road_name / task
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

data_info = preprocess_data(road_name)
_, _, X_training, X_test = build_dataset(
    data_info["t_sampled_circ"],
    data_info["S"],
    data_info["density"],
    data_info["v"],
    data_info["flow"],
    task,
)

x_sampled_circ = (data_info["x_sampled"][1:] + data_info["x_sampled"][:-1]) / 2
S = data_info["S"]

# define flat
dual_edges = C.CochainD1(S, S.dual_edges_vectors)
zeros = C.CochainD0(S, jnp.zeros_like(data_info["density"][:, 0]))
I_linear_left = tf_flat.get_linear_left_interpolation
flat_linear_left_D = partial(
    flat,
    weights=None,
    edges=dual_edges,
    interp_func=I_linear_left,
    interp_func_args={"sigma": zeros},
)


def flat_left_wrap(x):
    return flat_linear_left_D(C.CochainD0(S, x)).coeffs


flat_left = vmap(flat_left_wrap)

zeros_P = C.CochainP0(S, jnp.zeros_like(data_info["vP0"][:, 0]))
zeros_D = C.CochainD0(S, jnp.zeros_like(data_info["density"][:, 0]))

all_flats = tf_flat.define_flats(S, zeros_P, zeros_D)

flats = {
    "linear_left": all_flats["flat_linear_left_D"],
    "linear_right": all_flats["flat_linear_right_D"],
    "flat_left_v": flat_left,
}

if road_name == "US101":
    if task == "prediction":
        opt_greenshields = [0.79197062, 0.51105397]
        opt_Weidmann = [0.5893936, 0.54700764, 0.541796]
        opt_triangular = [0.42595455, 1.40987626, 7.19902727]
        opt_idm = [0.13046561, 0.68253887, 0.05752636, 0.49775953]
        opt_del_castillo = [0.21205058, 0.55683342, 0.82425097, 6.70846888]

        cb_ticks = [[0, 0.1, 0.2], [2, 36, 70]]

    elif task == "reconstruction":
        opt_greenshields = [0.7639689, 0.53459349]
        opt_Weidmann = [0.56370847, 0.53356334, 0.541796]
        opt_triangular = [0.42595455, 1.31832772, 7.19902727]
        opt_idm = [0.13046561, 0.74381154, 0.05752636, 0.54561196]
        opt_del_castillo = [0.21205058, 0.50790058, 0.75568294, 6.60932084]

        cb_ticks = [[0, 0.1, 0.2], [2, 36, 70]]

    x_ticks = [0, 1350, 2700]
    y_ticks = [10, 990, 1970]
elif road_name == "I80":
    if task == "prediction":
        opt_greenshields = [0.54673127, 0.55995123]
        opt_Weidmann = [0.63190729, 0.80612097, 0.24947817]
        opt_triangular = [0.37013956, 1.48964708, 6.59672108]
        opt_idm = [0.43936351, 0.93094344, 0.16251414, 0.61353022]
        opt_del_castillo = [0.31807369, 0.46732741, 0.61532169, 2.60100492]

        cb_ticks = [[0, 0.1, 0.2], [1, 40, 75]]

    elif task == "reconstruction":
        opt_greenshields = [0.67221695, 0.53916011]
        opt_Weidmann = [0.58670242, 0.71605332, 0.32424757]
        opt_triangular = [0.37468432, 1.28975743, 7.48885539]
        opt_idm = [0.43936351, 0.95108984, 0.16251414, 0.58307378]
        opt_del_castillo = [0.28700213, 0.99060242, 0.73049573, 1.7661283]

        cb_ticks = [[0, 0.1, 0.2], [1, 30, 65]]
    x_ticks = [0, 450, 900]
    y_ticks = [10, 760, 1510]

# set-up boundary conditions in an array
rho_bnd_array = jnp.zeros((len(data_info["rho_bnd"].keys()), data_info["num_t_points"]))
for index in data_info["rho_bnd"].keys():
    rho_bnd_array = rho_bnd_array.at[int(index), :].set(
        data_info["rho_bnd"][index][: data_info["num_t_points"]]
    )

flux_greens = lambda x: tf_utils.Greenshields_flux(x, *opt_greenshields)
flux_weidmann = lambda x: tf_utils.Weidmann_flux(x, *opt_Weidmann)
flux_triang = lambda x: tf_utils.triangular_flux(x, *opt_triangular)
flux_idm = lambda x: tf_utils.IDM_flux(x, *opt_idm)
flux_del_castillo = lambda x: tf_utils.del_castillo_flux(x, *opt_del_castillo)


Greenshields_der = tf_utils.define_flux_der(S, tf_utils.Greenshields_flux)
Weidmann_der = tf_utils.define_flux_der(S, tf_utils.Weidmann_flux)
triangular_flux_der = tf_utils.define_flux_der(S, tf_utils.triangular_flux)
IDM_flux_der = tf_utils.define_flux_der(S, tf_utils.IDM_flux)
del_castillo_flux_der = tf_utils.define_flux_der(S, tf_utils.del_castillo_flux)

flux_greens_der = lambda x: Greenshields_der(x, *opt_greenshields)
flux_weidmann_der = lambda x: Weidmann_der(x, *opt_Weidmann)
flux_triang_der = lambda x: triangular_flux_der(x, *opt_triangular)
flux_idm_der = lambda x: IDM_flux_der(x, *opt_idm)
flux_del_castillo_der = lambda x: del_castillo_flux_der(x, *opt_del_castillo)


# define baseline models
step = data_info["step"]
models = {
    "Greenshields": (flux_greens, flux_greens_der, None),
    "IDM": (flux_idm, flux_idm_der, None),
    "Weidmann": (flux_weidmann, flux_weidmann_der, None),
    "Triangular": (flux_triang, flux_triang_der, None),
    "Del Castillo": (flux_del_castillo, flux_del_castillo_der, None),
}

# The discovered coefficients are specific to the I80 prediction protocol.
# Corrected flux functions use their Phase-4 refits as defaults, and their
# derivatives retain the complete nonlocal convolution Jacobian.
automodel_models = {}
if road_name == "I80" and task == "prediction":
    automodel_fluxes = {
        "automodel-Greenshields": (
            lambda x: tf_utils.Greenshields_corrected_flux(x, *opt_greenshields),
            tf_utils.define_flux_der(S, tf_utils.Greenshields_corrected_flux),
            opt_greenshields,
        ),
        "automodel-IDM": (
            lambda x: tf_utils.IDM_corrected_flux(x, *opt_idm),
            tf_utils.define_flux_der(S, tf_utils.IDM_corrected_flux),
            opt_idm,
        ),
        "automodel-Weidmann": (
            lambda x: tf_utils.Weidmann_corrected_flux(x, *opt_Weidmann),
            tf_utils.define_flux_der(S, tf_utils.Weidmann_corrected_flux),
            opt_Weidmann,
        ),
        "automodel-Triangular": (
            lambda x: tf_utils.triangular_corrected_flux(x, *opt_triangular),
            tf_utils.define_flux_der(S, tf_utils.triangular_corrected_flux),
            opt_triangular,
        ),
        "automodel-Del Castillo": (
            lambda x: tf_utils.del_castillo_corrected_flux(x, *opt_del_castillo),
            tf_utils.define_flux_der(S, tf_utils.del_castillo_corrected_flux),
            opt_del_castillo,
        ),
    }
    automodel_models = {
        name: (
            flux_fn,
            lambda x, der=der, coeffs=coeffs: der(x, *coeffs),
            None,
        )
        for name, (flux_fn, der, coeffs) in automodel_fluxes.items()
    }


def simulate_models(model_registry: Dict):
    model_results = {}
    for name, (flux_fn, flux_der_fn, _) in model_registry.items():
        rho, rhoP0_model, v_model, f_model = simulate_model(
            flux_fn, flux_der_fn, data_info, S, flats, step, rho_bnd_array
        )
        rhoP0_model, rho, v_model, f_model = rescale_rho_v_f(
            rhoP0_model, rho, v_model, f_model, data_info
        )
        model_results[name] = dict(
            rho=rho[:, ::step],
            rhoP0=rhoP0_model[:-1, ::step],
            v=v_model,
            f=f_model,
        )
    return model_results


results = simulate_models(models)
automodel_results = simulate_models(automodel_models)


# Compute true Cochains
flat_density = C.CochainD1(S, flat_left(data_info["density"].T)[:, :, 0].T)
rhoP0 = C.star(flat_density).coeffs[:-1]
flat_v = C.CochainD1(S, flat_left(data_info["v"].T)[:, :, 0].T)
vP0 = C.star(flat_v).coeffs
v = vP0[:-1]
flat_f = flat_left(data_info["flow"].T)[:, :, 0].T
fP0 = C.star(C.CochainD1(S, flat_f))
f = fP0.coeffs[:-1]

# Rescale true data
rhoP0, data_info["density"], v, f = rescale_rho_v_f(
    rhoP0, data_info["density"], v, f, data_info
)


if task == "prediction":
    train_idx = jnp.arange(X_training[0, 0], X_training[-1, 0] + 1, dtype=jnp.int64)
    test_idx = jnp.arange(X_test[0, 0], X_test[-1, 0] + 1, dtype=jnp.int64)
elif task == "reconstruction":
    num_tr = int(X_training.shape[0] / len(data_info["t_sampled_circ"]))
    num_test = int(X_test.shape[0] / len(data_info["t_sampled_circ"]))
    train_idx = X_training[:num_tr, 0].astype(np.int64) + 1
    # add missing indexes removed for the BC
    shape = data_info["density"].shape[0]
    train_idx = np.concatenate((train_idx, [0, shape - 3, shape - 2, shape - 1]))
    test_idx = X_test[:num_test, 0].astype(np.int64) + 1


x_sampled_circ *= data_info["L_dim"]
data_info["t_sampled_circ"] *= data_info["t_len"]


# plot params
fontsize = 15
plt.rcParams["font.size"] = fontsize
plt.rcParams["font.sans-serif"] = "Dejavu Sans"
plt.rcParams["font.family"] = "sans-serif"

plot_diagrams(results, rhoP0, v, f, "flux", test_name, train_idx, test_idx, task)
plot_diagrams(results, rhoP0, v, f, "velocity", test_name, train_idx, test_idx, task)

if automodel_results:
    plot_diagrams(
        automodel_results,
        rhoP0,
        v,
        f,
        "flux",
        test_name + "_automodel",
        train_idx,
        test_idx,
        task,
    )
    plot_diagrams(
        automodel_results,
        rhoP0,
        v,
        f,
        "velocity",
        test_name + "_automodel",
        train_idx,
        test_idx,
        task,
    )

rho_v_plot(
    results,
    data_info,
    v,
    x_sampled_circ,
    test_name,
    x_ticks,
    y_ticks,
    cb_ticks,
    road_name,
    task,
)
if automodel_results:
    rho_v_plot(
        automodel_results,
        data_info,
        v,
        x_sampled_circ,
        test_name + "_automodel",
        x_ticks,
        y_ticks,
        cb_ticks,
        road_name,
        task,
    )


# predicted-true plots
predicted_true_plots(results, v, f, test_name)
if automodel_results:
    predicted_true_plots(automodel_results, v, f, test_name + "_automodel")


# Save the error tables
baseline_results = results
results = baseline_results | automodel_results
save_error_tables(
    results,
    test_idx,
    road_name,
    task,
    data_info["t_sampled_circ"],
    x_sampled_circ,
    RESULTS_DIR / "error_table.tex",
)
save_score_tables(
    results,
    train_idx,
    test_idx,
    road_name,
    task,
    RESULTS_DIR / "score_table.tex",
)
if automodel_results:
    save_error_tables(
        automodel_results,
        test_idx,
        road_name,
        task,
        data_info["t_sampled_circ"],
        x_sampled_circ,
        RESULTS_DIR / "automodel_error_table.tex",
        reference_results=baseline_results,
    )
    save_score_tables(
        automodel_results,
        train_idx,
        test_idx,
        road_name,
        task,
        RESULTS_DIR / "automodel_score_table.tex",
    )
