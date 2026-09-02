"""Plot per-baseline best train and validation E_data across meta-iterations."""

from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
STAGES = ["Identity", "Meta 1", "Meta 2", "Meta 3", "Meta 4", "Meta 5"]
HISTORY = {
    "Greenshields": {
        "train": [7.203543, 7.064701, 7.047837, 6.334473, 6.334473, 6.334473],
        "validation": [9.610405, 8.045311, 7.507391, 6.703732, 6.703732, 6.703732],
    },
    "IDM": {
        "train": [8.438750, 8.438750, 8.438750, 8.265197, 7.679225, 7.679225],
        "validation": [5.945012, 5.945012, 5.945012, 5.277018, 4.728745, 4.728745],
    },
    "Weidmann": {
        "train": [7.232718, 7.205753, 7.205753, 7.205753, 7.205753, 7.205753],
        "validation": [6.480163, 6.222293, 6.222293, 6.222293, 6.222293, 6.222293],
    },
    "Triangular": {
        "train": [9.223384, 9.223384, 9.223384, 9.223384, 9.223384, 7.965049],
        "validation": [6.447489, 6.447489, 6.447489, 6.447489, 6.447489, 5.067508],
    },
    "Del Castillo": {
        "train": [7.048800, 7.048800, 7.047598, 6.364161, 6.364161, 6.364161],
        "validation": [6.628030, 6.628030, 6.583956, 5.520221, 5.520221, 5.520221],
    },
}


def main() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharex=True)
    for name, scores in HISTORY.items():
        axes[0].plot(STAGES, scores["train"], marker="o", label=name)
        axes[1].plot(STAGES, scores["validation"], marker="o", label=name)
    axes[0].set_title("Training $E_{data}$")
    axes[1].set_title("Validation $E_{data}$")
    for axis in axes:
        axis.set_ylabel("Lower is better")
        axis.grid(alpha=0.25)
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=5, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    output = ROOT / "automodel" / "progress.png"
    fig.savefig(output, dpi=180, bbox_inches="tight")
    print(output)


if __name__ == "__main__":
    main()
