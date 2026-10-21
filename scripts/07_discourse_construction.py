"""
Build dissertation-facing discourse-construction analyses for Dissertation.

This step translates semantic-shift outputs into substantive analytical tables:

1. Institutional discourse axes across legislature transitions.
2. Term trajectories for core institutional vocabulary.
3. Nearest-neighbor profiles by legislature for selected terms.
4. A concise interpretation memo for the results section.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from gensim.models import KeyedVectors


DEFAULT_RESULTS_DIR = Path("resultados")
DEFAULT_FIGURES_DIR = Path("figuras/dissertacao")
DEFAULT_ALIGNED_DIR = Path("modelos/dissertacao_word2vec_aligned")
TARGET_LEGS = [52, 53, 54, 55, 56, 57]

DISCOURSE_AXES = {
    "governability_and_coalition": {
        "label": "Governability and coalition politics",
        "terms": {
            "coalizao", "base", "governo", "presidente", "presidencia",
            "partido", "partidos", "oposicao", "ministro", "ministros",
        },
        "interpretation": (
            "Tracks how parliamentary discourse organizes executive-legislative "
            "coordination, governing base, opposition, and cabinet authority."
        ),
    },
    "institutional_legitimacy": {
        "label": "Institutional legitimacy and democratic order",
        "terms": {
            "democracia", "democratico", "democratica", "constituicao",
            "constitucional", "republica", "direito", "voto", "eleicao",
            "eleicoes", "federacao", "estado",
        },
        "interpretation": (
            "Captures shifts in the vocabulary through which democracy, rights, "
            "elections, constitutional order, and federation are invoked."
        ),
    },
    "judicialization_and_checks": {
        "label": "Judicialization and institutional checks",
        "terms": {
            "judiciario", "stf", "supremo", "justica", "constitucional",
            "direito", "impeachment",
        },
        "interpretation": (
            "Captures the changing semantic role of courts, legal authority, "
            "constitutional dispute, and impeachment."
        ),
    },
    "parliamentary_arena": {
        "label": "Parliamentary arena and institutional venue",
        "terms": {
            "congresso", "camara", "senado", "parlamento", "partido",
            "partidos", "voto",
        },
        "interpretation": (
            "Measures shifts in the vocabulary that names the parliamentary "
            "arena and its procedural-political actors."
        ),
    },
    "security_and_military": {
        "label": "Security and military presence",
        "terms": {"militar", "militares", "estado", "governo"},
        "interpretation": (
            "Tracks the semantic movement of military and state vocabulary in "
            "the parliamentary field."
        ),
    },
}

PAIR_CONTEXT = {
    "52-53": "pre-2011 baseline",
    "53-54": "pre-2011 to Dilma I",
    "54-55": "Dilma I to impeachment legislature",
    "55-56": "post-impeachment / Temer to Bolsonaro",
    "56-57": "Bolsonaro to Lula III",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create discourse-construction analyses for Dissertation."
    )
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--figures-dir", type=Path, default=DEFAULT_FIGURES_DIR)
    parser.add_argument("--aligned-dir", type=Path, default=DEFAULT_ALIGNED_DIR)
    parser.add_argument("--base-leg", type=int, default=52)
    parser.add_argument("--top-terms", type=int, default=25)
    parser.add_argument("--neighbor-k", type=int, default=12)
    return parser.parse_args()


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing input file: {path}")
    return pd.read_csv(path)


def load_models(aligned_dir: Path, legs: list[int], base_leg: int) -> dict[int, KeyedVectors]:
    models = {}
    for leg in legs:
        path = aligned_dir / f"w2v_leg{leg}_aligned_to_leg{base_leg}.kv"
        if not path.exists():
            raise FileNotFoundError(f"Missing aligned model: {path}")
        models[leg] = KeyedVectors.load(str(path), mmap="r")
    return models


def assign_axes(term: str) -> list[str]:
    axes = []
    for axis, spec in DISCOURSE_AXES.items():
        if term in spec["terms"]:
            axes.append(axis)
    return axes or ["other_institutional"]


def build_axis_summary(institutional: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in institutional.iterrows():
        for axis in assign_axes(str(row["term"])):
            if axis == "other_institutional":
                continue
            rows.append(
                {
                    "axis": axis,
                    "axis_label": DISCOURSE_AXES[axis]["label"],
                    "term": row["term"],
                    "pair": row["pair"],
                    "pair_context": row["pair_context"],
                    "source_leg": row["source_leg"],
                    "target_leg": row["target_leg"],
                    "cosine_distance": row["cosine_distance"],
                    "global_distance_52_57": row["cosine_distance_52_57"],
                }
            )

    expanded = pd.DataFrame(rows)
    summary = (
        expanded.groupby(["axis", "axis_label", "pair", "pair_context"], as_index=False)
        .agg(
            n_terms=("term", "nunique"),
            mean_cosine_distance=("cosine_distance", "mean"),
            median_cosine_distance=("cosine_distance", "median"),
            max_cosine_distance=("cosine_distance", "max"),
        )
        .sort_values(["axis", "pair"])
    )
    return summary


def build_term_trajectories(institutional: pd.DataFrame, final_terms: pd.DataFrame) -> pd.DataFrame:
    keep_terms = set(final_terms["term"].astype(str))
    data = institutional[institutional["term"].astype(str).isin(keep_terms)].copy()
    axis_rows = []
    for _, row in data.iterrows():
        axes = [a for a in assign_axes(str(row["term"])) if a != "other_institutional"]
        axis_rows.append(";".join(axes))
    data["discourse_axes"] = axis_rows
    cols = [
        "term",
        "discourse_axes",
        "pair",
        "pair_context",
        "source_leg",
        "target_leg",
        "cosine_distance",
        "cosine_distance_52_57",
        "term_class",
    ]
    return data[cols].sort_values(["term", "source_leg", "target_leg"])


def parse_neighbor_list(raw: list[tuple[str, float]]) -> str:
    return "; ".join(f"{term}:{score:.4f}" for term, score in raw)


def build_neighbor_profiles(
    models: dict[int, KeyedVectors],
    selected_terms: list[str],
    neighbor_k: int,
) -> pd.DataFrame:
    rows = []
    selected_set = set(selected_terms)
    for term in selected_terms:
        for leg, model in models.items():
            if term not in model:
                rows.append(
                    {
                        "term": term,
                        "leg": leg,
                        "available": False,
                        "neighbors": "",
                    }
                )
                continue
            raw = [
                (neighbor, score)
                for neighbor, score in model.most_similar(term, topn=neighbor_k + len(selected_set))
                if neighbor not in selected_set
            ][:neighbor_k]
            rows.append(
                {
                    "term": term,
                    "leg": leg,
                    "available": True,
                    "neighbors": parse_neighbor_list(raw),
                }
            )
    return pd.DataFrame(rows).sort_values(["term", "leg"])


def plot_axis_summary(axis_summary: pd.DataFrame, out_path: Path) -> None:
    pivot = axis_summary.pivot(index="pair", columns="axis_label", values="mean_cosine_distance")
    pivot = pivot.reindex(["52-53", "53-54", "54-55", "55-56", "56-57"])

    fig, ax = plt.subplots(figsize=(10, 5.5))
    for column in pivot.columns:
        ax.plot(pivot.index, pivot[column], marker="o", linewidth=2, label=column)

    ax.set_xlabel("Adjacent legislature pair")
    ax.set_ylabel("Mean cosine distance")
    ax.set_title("Semantic instability by institutional discourse axis")
    ax.grid(axis="y", alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def build_summary(axis_summary: pd.DataFrame, final_terms: pd.DataFrame) -> str:
    strongest_axis_rows = (
        axis_summary.sort_values("mean_cosine_distance", ascending=False)
        .groupby("axis_label", as_index=False)
        .first()
        .sort_values("mean_cosine_distance", ascending=False)
    )
    top_terms = final_terms.head(12)

    lines = [
        "# Dissertation discourse-construction interpretation",
        "",
        "## Central claim",
        "",
        (
            "The semantic-shift results indicate a longitudinal reorganization "
            "of parliamentary institutional discourse. The main finding is not "
            "only that words move in vector space, but that core institutional "
            "terms change the contexts in which they are used across political "
            "transitions."
        ),
        "",
        "## Strongest discourse axes",
        "",
    ]

    for _, row in strongest_axis_rows.iterrows():
        lines.append(
            f"- {row['axis_label']}: strongest in {row['pair']} "
            f"({row['pair_context']}), mean distance {row['mean_cosine_distance']:.4f}."
        )

    lines.extend(
        [
            "",
            "## Main terms for substantive interpretation",
            "",
        ]
    )

    for _, row in top_terms.iterrows():
        axes = ", ".join(assign_axes(str(row["term"])))
        lines.append(
            f"- {row['term']}: axes={axes}; strongest pair={row['strongest_pair']} "
            f"({row['strongest_pair_context']}); 52-57 distance={row['global_distance_52_57']:.4f}."
        )

    lines.extend(
        [
            "",
            "## Suggested results-section reading",
            "",
            (
                "The article should present the results as a reconstruction of "
                "how parliamentary discourse organized institutional conflict, "
                "not as a methodological demonstration. The tables identify "
                "which institutional vocabularies became unstable, when the "
                "instability intensified, and which terms should anchor the "
                "qualitative reading of speeches."
            ),
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    args.results_dir.mkdir(parents=True, exist_ok=True)
    args.figures_dir.mkdir(parents=True, exist_ok=True)

    institutional = read_csv(args.results_dir / "dissertacao_table_institutional_terms_shift.csv")
    final_terms = read_csv(args.results_dir / "dissertacao_final_interpretable_terms.csv").head(args.top_terms)

    axis_summary = build_axis_summary(institutional)
    trajectories = build_term_trajectories(institutional, final_terms)

    models = load_models(args.aligned_dir, TARGET_LEGS, args.base_leg)
    selected_terms = final_terms["term"].astype(str).tolist()
    neighbor_profiles = build_neighbor_profiles(models, selected_terms, args.neighbor_k)

    outputs = {
        "axis_summary": args.results_dir / "dissertacao_discourse_axis_summary.csv",
        "term_trajectories": args.results_dir / "dissertacao_discourse_term_trajectories.csv",
        "neighbor_profiles": args.results_dir / "dissertacao_discourse_neighbor_profiles.csv",
        "summary": args.results_dir / "dissertacao_discourse_construction_summary.md",
        "manifest": args.results_dir / "dissertacao_discourse_construction_manifest.json",
        "figure": args.figures_dir / "dissertacao_discourse_axis_trajectories.png",
    }

    axis_summary.to_csv(outputs["axis_summary"], index=False, encoding="utf-8")
    trajectories.to_csv(outputs["term_trajectories"], index=False, encoding="utf-8")
    neighbor_profiles.to_csv(outputs["neighbor_profiles"], index=False, encoding="utf-8")
    outputs["summary"].write_text(build_summary(axis_summary, final_terms), encoding="utf-8")
    plot_axis_summary(axis_summary, outputs["figure"])

    manifest = {
        "purpose": "substantive discourse-construction analyses for Dissertation",
        "inputs": {
            "institutional_shift": str(args.results_dir / "dissertacao_table_institutional_terms_shift.csv"),
            "final_terms": str(args.results_dir / "dissertacao_final_interpretable_terms.csv"),
            "aligned_models": str(args.aligned_dir),
        },
        "outputs": {key: str(path) for key, path in outputs.items()},
        "discourse_axes": {
            axis: {
                "label": spec["label"],
                "terms": sorted(spec["terms"]),
                "interpretation": spec["interpretation"],
            }
            for axis, spec in DISCOURSE_AXES.items()
        },
    }
    outputs["manifest"].write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"Saved discourse axis summary to: {outputs['axis_summary']}")
    print(f"Saved term trajectories to: {outputs['term_trajectories']}")
    print(f"Saved neighbor profiles to: {outputs['neighbor_profiles']}")
    print(f"Saved interpretation summary to: {outputs['summary']}")
    print(f"Saved axis trajectory figure to: {outputs['figure']}")


if __name__ == "__main__":
    main()
