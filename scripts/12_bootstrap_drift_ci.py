"""
12_bootstrap_drift_ci.py
Bootstrap 95% confidence intervals for semantic drift by adjacent legislature pair.

Reads: resultados/dissertacao_semantic_shift_adjacent.csv
Outputs:
    resultados/dissertacao_drift_bootstrap_ci.csv
    figuras/dissertacao/dissertacao_drift_with_ci.png
    resultados/dissertacao_drift_bootstrap_manifest.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_RESULTS_DIR = Path("resultados")
DEFAULT_FIGURES_DIR = Path("figuras/dissertacao")
N_BOOTSTRAP = 1_000
SEED = 42
CI_LEVEL = 0.95

PAIR_CONTEXT = {
    "52-53": "52→53\n(baseline)",
    "53-54": "53→54\n(Dilma I)",
    "54-55": "54→55\n(impeachment)",
    "55-56": "55→56\n(Bolsonaro)",
    "56-57": "56→57\n(Lula III)",
}

TRANSITION_PAIRS = {"54-55", "55-56", "56-57"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Bootstrap CI for semantic drift by adjacent legislature pair."
    )
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--figures-dir", type=Path, default=DEFAULT_FIGURES_DIR)
    parser.add_argument("--n-bootstrap", type=int, default=N_BOOTSTRAP,
                        help="Number of bootstrap resamples (default: 1000).")
    parser.add_argument("--ci", type=float, default=CI_LEVEL,
                        help="Confidence level, e.g. 0.95 (default).")
    parser.add_argument("--seed", type=int, default=SEED)
    return parser.parse_args()


def bootstrap_mean_ci(
    values: np.ndarray,
    n_bootstrap: int,
    rng: np.random.Generator,
    ci_level: float,
) -> tuple[float, float, float]:
    """Return (observed_mean, ci_lower, ci_upper) via non-parametric bootstrap."""
    boot_means = rng.choice(values, size=(n_bootstrap, len(values)), replace=True).mean(axis=1)
    alpha = (1.0 - ci_level) / 2.0
    return (
        float(values.mean()),
        float(np.percentile(boot_means, 100.0 * alpha)),
        float(np.percentile(boot_means, 100.0 * (1.0 - alpha))),
    )


def compute_bootstrap_table(
    adjacent: pd.DataFrame,
    n_bootstrap: int,
    rng: np.random.Generator,
    ci_level: float,
) -> pd.DataFrame:
    rows = []
    for pair, group in adjacent.groupby("pair"):
        values = group["cosine_distance"].dropna().values
        mean, ci_lo, ci_hi = bootstrap_mean_ci(values, n_bootstrap, rng, ci_level)
        rows.append(
            {
                "pair": pair,
                "source_leg": int(group["source_leg"].iloc[0]),
                "target_leg": int(group["target_leg"].iloc[0]),
                "n_terms": len(values),
                "mean_cosine_distance": round(mean, 6),
                "median_cosine_distance": round(float(np.median(values)), 6),
                "ci_lower": round(ci_lo, 6),
                "ci_upper": round(ci_hi, 6),
                "ci_width": round(ci_hi - ci_lo, 6),
                "is_transition": pair in TRANSITION_PAIRS,
                "n_bootstrap": n_bootstrap,
                "ci_level": ci_level,
            }
        )
    return pd.DataFrame(rows).sort_values(["source_leg", "target_leg"]).reset_index(drop=True)


def plot_drift_with_ci(result: pd.DataFrame, ci_level: float, out_path: Path) -> None:
    pairs = result["pair"].tolist()
    x = np.arange(len(pairs))
    means = result["mean_cosine_distance"].values
    ci_lo = result["ci_lower"].values
    ci_hi = result["ci_upper"].values
    is_transition = result["is_transition"].values

    fig, ax = plt.subplots(figsize=(8.5, 4.8))

    # shaded CI band
    ax.fill_between(x, ci_lo, ci_hi, alpha=0.20, color="#2c6fad",
                    label=f"{int(ci_level * 100)}% bootstrap CI")

    # error bars
    ax.errorbar(
        x, means,
        yerr=[means - ci_lo, ci_hi - means],
        fmt="none", color="#2c6fad", capsize=5, linewidth=1.4,
    )

    # main line
    ax.plot(x, means, marker="o", color="#2c6fad", linewidth=2.2,
            markersize=7, label="Mean cosine drift", zorder=3)

    # highlight transition pairs
    for i, (m, tr) in enumerate(zip(means, is_transition)):
        if tr:
            ax.axvspan(i - 0.4, i + 0.4, alpha=0.06, color="#cc3333", zorder=0)

    ax.set_xticks(x)
    ax.set_xticklabels(
        [PAIR_CONTEXT.get(p, p) for p in pairs],
        fontsize=9,
    )
    ax.set_ylabel("Mean cosine distance", fontsize=11)
    ax.set_title(
        "Semantic drift by adjacent legislature pair\n"
        f"({int(ci_level * 100)}% bootstrap confidence intervals, N terms = {result['n_terms'].iloc[0]:,})",
        fontsize=11,
    )
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_ci_width(result: pd.DataFrame, out_path: Path) -> None:
    """Secondary diagnostic: CI width shows estimation precision per pair."""
    pairs = result["pair"].tolist()
    x = np.arange(len(pairs))

    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.bar(x, result["ci_width"].values, color="#4a9a73", edgecolor="white", width=0.55)
    ax.set_xticks(x)
    ax.set_xticklabels(pairs, fontsize=9)
    ax.set_ylabel("CI width (upper − lower)", fontsize=10)
    ax.set_title("Bootstrap CI width by legislature pair\n(narrower = more precise estimate)", fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main(args: argparse.Namespace) -> None:
    args.results_dir.mkdir(parents=True, exist_ok=True)
    args.figures_dir.mkdir(parents=True, exist_ok=True)

    input_path = args.results_dir / "dissertacao_semantic_shift_adjacent.csv"
    if not input_path.exists():
        raise FileNotFoundError(
            f"Missing input: {input_path}\n"
            "Run 04_semantic_shift.py first."
        )

    adjacent = pd.read_csv(input_path)
    adjacent["pair"] = (
        adjacent["source_leg"].astype(str) + "-" + adjacent["target_leg"].astype(str)
    )

    rng = np.random.default_rng(args.seed)
    result = compute_bootstrap_table(adjacent, args.n_bootstrap, rng, args.ci)

    out_csv = args.results_dir / "dissertacao_drift_bootstrap_ci.csv"
    result.to_csv(out_csv, index=False, encoding="utf-8")

    out_fig = args.figures_dir / "dissertacao_drift_with_ci.png"
    plot_drift_with_ci(result, args.ci, out_fig)

    out_ci_width = args.figures_dir / "dissertacao_drift_ci_width.png"
    plot_ci_width(result, out_ci_width)

    manifest = {
        "input": str(input_path),
        "n_bootstrap": args.n_bootstrap,
        "ci_level": args.ci,
        "seed": args.seed,
        "outputs": {
            "bootstrap_ci_table": str(out_csv),
            "drift_with_ci_figure": str(out_fig),
            "ci_width_figure": str(out_ci_width),
        },
    }
    manifest_path = args.results_dir / "dissertacao_drift_bootstrap_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\nBootstrap CI results:")
    print(
        result[["pair", "n_terms", "mean_cosine_distance", "ci_lower", "ci_upper", "ci_width"]]
        .to_string(index=False, float_format="{:.4f}".format)
    )
    print(f"\nSaved CI table   : {out_csv}")
    print(f"Saved main figure: {out_fig}")
    print(f"Saved CI width   : {out_ci_width}")
    print(f"Saved manifest   : {manifest_path}")


if __name__ == "__main__":
    main(parse_args())
