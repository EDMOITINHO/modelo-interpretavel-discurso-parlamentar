"""
Prepare interpretable tables and figures for Dissertation semantic shift results.

This script does not retrain or realign models. It turns the CSV outputs from
04_semantic_shift.py into dissertation-oriented summaries:

1. Drift magnitude by adjacent legislature pair.
2. Drift of institutional/political concept terms linked to the paper hypotheses.
3. Candidate terms for qualitative interpretation, with discourse/proper-name
   terms flagged rather than silently removed.
4. Compact figures for the results section.

Expected inputs:

    resultados/dissertacao_semantic_shift_adjacent.csv
    resultados/dissertacao_semantic_shift_52_57.csv
    resultados/dissertacao_semantic_shift_top_terms.csv
    resultados/dissertacao_semantic_shift_neighbors.csv
    resultados/dissertacao_procrustes_alignment_metrics.csv
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_RESULTS_DIR = Path("resultados")
DEFAULT_FIGURES_DIR = Path("figuras/dissertacao")


INSTITUTIONAL_TERMS = [
    "democracia",
    "democratico",
    "democratica",
    "constituicao",
    "constitucional",
    "judiciario",
    "supremo",
    "stf",
    "congresso",
    "camara",
    "senado",
    "parlamento",
    "governo",
    "presidente",
    "presidencia",
    "republica",
    "federacao",
    "estado",
    "direito",
    "justica",
    "impeachment",
    "eleicao",
    "eleicoes",
    "voto",
    "partido",
    "partidos",
    "oposicao",
    "base",
    "coalizao",
    "ministro",
    "ministros",
    "militar",
    "militares",
]


DISCOURSE_MARKERS = {
    "acrescento",
    "adiantou",
    "argumentando",
    "concretamente",
    "convenhamos",
    "dira",
    "entendo",
    "explico",
    "imaginamos",
    "impressiona",
    "nitidamente",
    "observe",
    "percebam",
    "permitimos",
    "questiono",
    "teoricamente",
}


PROCEDURAL_TERMS = {
    "deferimento",
    "indeferimento",
    "paragrafo",
    "redacional",
    "requerimento",
    "votacao",
    "voto",
    "pauta",
    "ordem",
    "sessao",
    "plenaria",
    "comissao",
    "relator",
    "relatorio",
    "emenda",
    "projeto",
}


POLITICAL_ACTOR_TERMS = {
    "bolsonaro",
    "dilma",
    "rousseff",
    "lula",
    "temer",
    "michel",
    "pacheco",
    "maia",
    "cunha",
    "moro",
    "weber",
    "heleno",
    "marques",
    "motta",
    "amin",
    "bacelar",
    "gaspar",
    "neri",
}


PAIR_CONTEXT = {
    "52-53": "pre-2011 baseline",
    "53-54": "pre-2011 to Dilma I",
    "54-55": "Dilma I to impeachment legislature",
    "55-56": "post-impeachment / Temer to Bolsonaro",
    "56-57": "Bolsonaro to Lula III",
}


TRANSITION_PAIRS = {"54-55", "55-56", "56-57"}
PAPER_FOCUS_PAIRS = {"54-55", "55-56"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate dissertation-oriented interpretation tables for semantic shift."
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=DEFAULT_RESULTS_DIR,
        help="Directory containing semantic shift CSV files.",
    )
    parser.add_argument(
        "--figures-dir",
        type=Path,
        default=DEFAULT_FIGURES_DIR,
        help="Directory where figures will be saved.",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=100,
        help="Number of qualitative candidate terms to export.",
    )
    parser.add_argument(
        "--institutional-terms",
        nargs="*",
        default=None,
        help="Optional custom list of institutional terms. Defaults to the paper list.",
    )
    return parser.parse_args()


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing input file: {path}")
    return pd.read_csv(path)


def pair_label(source_leg: int, target_leg: int) -> str:
    return f"{int(source_leg)}-{int(target_leg)}"


def add_pair_columns(frame: pd.DataFrame) -> pd.DataFrame:
    data = frame.copy()
    data["pair"] = [
        pair_label(source, target)
        for source, target in zip(data["source_leg"], data["target_leg"])
    ]
    data["pair_context"] = data["pair"].map(PAIR_CONTEXT).fillna("")
    data["is_government_transition"] = data["pair"].isin(TRANSITION_PAIRS)
    data["is_paper_focus_transition"] = data["pair"].isin(PAPER_FOCUS_PAIRS)
    return data


def summarize_adjacent(adjacent: pd.DataFrame) -> pd.DataFrame:
    summary = (
        adjacent.groupby("pair", as_index=False)
        .agg(
            source_leg=("source_leg", "first"),
            target_leg=("target_leg", "first"),
            terms=("term", "count"),
            mean_cosine_distance=("cosine_distance", "mean"),
            median_cosine_distance=("cosine_distance", "median"),
            p90_cosine_distance=("cosine_distance", lambda x: x.quantile(0.90)),
            p95_cosine_distance=("cosine_distance", lambda x: x.quantile(0.95)),
            max_cosine_distance=("cosine_distance", "max"),
            mean_euclidean_distance=("euclidean_distance", "mean"),
        )
        .sort_values(["source_leg", "target_leg"])
    )
    summary["pair_context"] = summary["pair"].map(PAIR_CONTEXT).fillna("")
    summary["is_government_transition"] = summary["pair"].isin(TRANSITION_PAIRS)
    summary["is_paper_focus_transition"] = summary["pair"].isin(PAPER_FOCUS_PAIRS)
    return summary


def classify_term(term: str) -> str:
    if term in INSTITUTIONAL_TERMS:
        return "institutional_concept"
    if term in POLITICAL_ACTOR_TERMS:
        return "political_actor_or_name"
    if term in PROCEDURAL_TERMS:
        return "parliamentary_procedure"
    if term in DISCOURSE_MARKERS:
        return "discursive_marker"
    return "candidate_content_term"


def build_institutional_table(
    adjacent: pd.DataFrame,
    global_shift: pd.DataFrame,
    terms: list[str],
) -> pd.DataFrame:
    available_terms = sorted(set(terms).intersection(adjacent["term"].unique()))
    institutional = adjacent[adjacent["term"].isin(available_terms)].copy()

    global_cols = global_shift[
        ["term", "cosine_distance", "euclidean_distance"]
    ].rename(
        columns={
            "cosine_distance": "cosine_distance_52_57",
            "euclidean_distance": "euclidean_distance_52_57",
        }
    )

    institutional = institutional.merge(global_cols, on="term", how="left")
    institutional["term_class"] = "institutional_concept"
    return institutional.sort_values(
        ["term", "source_leg", "target_leg"]
    )


def build_qualitative_candidates(
    top_terms: pd.DataFrame,
    global_shift: pd.DataFrame,
    top_n: int,
) -> pd.DataFrame:
    candidates = top_terms.copy()
    candidates["term_class"] = candidates["term"].map(classify_term)

    global_rank = global_shift[["term", "cosine_distance"]].copy()
    global_rank = global_rank.sort_values("cosine_distance", ascending=False)
    global_rank["rank_52_57"] = np.arange(1, len(global_rank) + 1)

    candidates = candidates.merge(
        global_rank[["term", "rank_52_57"]],
        on="term",
        how="left",
    )

    candidates["recommended_for_close_reading"] = candidates["term_class"].isin(
        {"institutional_concept", "candidate_content_term", "parliamentary_procedure"}
    )
    return candidates.head(top_n)


def build_neighbors_for_candidates(
    neighbors: pd.DataFrame,
    candidates: pd.DataFrame,
) -> pd.DataFrame:
    selected_terms = candidates["term"].tolist()
    selected = neighbors[neighbors["term"].isin(selected_terms)].copy()
    selected = selected.merge(
        candidates[["term", "term_class", "recommended_for_close_reading"]],
        on="term",
        how="left",
    )
    return selected.sort_values(["term", "leg"])


def build_top_terms_per_pair(
    adjacent: pd.DataFrame,
    global_shift: pd.DataFrame,
    top_n: int = 5,
) -> pd.DataFrame:
    """
    Compact paper-ready table: top-N most shifted terms per adjacent pair,
    filtered to content terms (excluding discourse markers and proper names).
    Includes global 52→57 distance so each row is self-contained.
    """
    exclude_classes = {"political_actor_or_name", "discursive_marker"}

    global_dist = (
        global_shift[["term", "cosine_distance"]]
        .rename(columns={"cosine_distance": "global_distance_52_57"})
    )

    rows = []
    for pair, group in adjacent.groupby("pair"):
        group = group.copy()
        group["term_class"] = group["term"].map(classify_term)
        content = group[~group["term_class"].isin(exclude_classes)]
        top = content.nlargest(top_n, "cosine_distance")
        for rank, (_, row) in enumerate(top.iterrows(), start=1):
            rows.append(
                {
                    "pair": pair,
                    "pair_context": row.get("pair_context", ""),
                    "rank": rank,
                    "term": row["term"],
                    "term_class": row["term_class"],
                    "cosine_distance_adjacent": round(float(row["cosine_distance"]), 4),
                }
            )

    result = pd.DataFrame(rows)
    if not result.empty:
        result = result.merge(global_dist, on="term", how="left")
        result["global_distance_52_57"] = result["global_distance_52_57"].round(4)
    return result.sort_values(["pair", "rank"])


def write_markdown_summary(
    path: Path,
    pair_summary: pd.DataFrame,
    institutional: pd.DataFrame,
    candidates: pd.DataFrame,
) -> None:
    strongest_pair = pair_summary.sort_values(
        "mean_cosine_distance", ascending=False
    ).iloc[0]

    institutional_top = (
        institutional.sort_values("cosine_distance", ascending=False)
        .head(10)["term"]
        .drop_duplicates()
        .tolist()
    )

    candidate_top = candidates.head(15)[["term", "term_class"]]

    lines = [
        "# Dissertation semantic shift interpretation summary",
        "",
        "## Drift by legislature pair",
        "",
        (
            f"The highest mean adjacent drift is observed for pair "
            f"{strongest_pair['pair']} ({strongest_pair['pair_context']}), "
            f"with mean cosine distance {strongest_pair['mean_cosine_distance']:.4f}."
        ),
        "",
        "## Institutional terms with high drift",
        "",
        ", ".join(institutional_top) if institutional_top else "No institutional terms found.",
        "",
        "## Top qualitative candidates",
        "",
    ]

    for _, row in candidate_top.iterrows():
        lines.append(f"- {row['term']}: {row['term_class']}")

    lines.extend(
        [
            "",
            "Interpretation note: political actors and discursive markers are flagged,",
            "not discarded. They should be read as contextual indicators unless the",
            "article explicitly treats them as lexical-semantic concepts.",
            "",
        ]
    )

    path.write_text("\n".join(lines), encoding="utf-8")


def plot_pair_summary(pair_summary: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    colors = [
        "#b84a3a" if value else "#4f7cac"
        for value in pair_summary["is_paper_focus_transition"]
    ]
    ax.bar(pair_summary["pair"], pair_summary["mean_cosine_distance"], color=colors)
    ax.set_xlabel("Adjacent legislatures")
    ax.set_ylabel("Mean cosine distance")
    ax.set_title("Mean semantic drift by adjacent legislature pair")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_institutional_heatmap(institutional: pd.DataFrame, output_path: Path) -> None:
    if institutional.empty:
        return

    pivot = institutional.pivot_table(
        index="term",
        columns="pair",
        values="cosine_distance",
        aggfunc="mean",
    )
    pivot["mean"] = pivot.mean(axis=1)
    pivot = pivot.sort_values("mean", ascending=False).drop(columns=["mean"]).head(20)

    fig, ax = plt.subplots(figsize=(8.5, max(4.5, 0.35 * len(pivot))))
    image = ax.imshow(pivot.values, aspect="auto", cmap="viridis")
    ax.set_xticks(np.arange(len(pivot.columns)), labels=pivot.columns)
    ax.set_yticks(np.arange(len(pivot.index)), labels=pivot.index)
    ax.set_title("Institutional term drift by legislature pair")
    ax.set_xlabel("Adjacent legislatures")
    ax.set_ylabel("Term")
    fig.colorbar(image, ax=ax, label="Cosine distance")
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def main(args: argparse.Namespace) -> None:
    args.results_dir.mkdir(parents=True, exist_ok=True)
    args.figures_dir.mkdir(parents=True, exist_ok=True)

    adjacent = add_pair_columns(
        read_csv(args.results_dir / "dissertacao_semantic_shift_adjacent.csv")
    )
    global_shift = read_csv(args.results_dir / "dissertacao_semantic_shift_52_57.csv")
    top_terms = read_csv(args.results_dir / "dissertacao_semantic_shift_top_terms.csv")
    neighbors = read_csv(args.results_dir / "dissertacao_semantic_shift_neighbors.csv")
    alignment = read_csv(args.results_dir / "dissertacao_procrustes_alignment_metrics.csv")

    institutional_terms = args.institutional_terms or INSTITUTIONAL_TERMS

    pair_summary = summarize_adjacent(adjacent)
    institutional = build_institutional_table(
        adjacent,
        global_shift,
        institutional_terms,
    )
    candidates = build_qualitative_candidates(top_terms, global_shift, args.top_n)
    candidate_neighbors = build_neighbors_for_candidates(neighbors, candidates)

    pair_summary_path = args.results_dir / "dissertacao_table_drift_by_pair.csv"
    institutional_path = args.results_dir / "dissertacao_table_institutional_terms_shift.csv"
    candidates_path = args.results_dir / "dissertacao_table_qualitative_candidates.csv"
    neighbors_path = args.results_dir / "dissertacao_table_candidate_neighbors.csv"
    alignment_path = args.results_dir / "dissertacao_table_alignment_quality.csv"
    top_terms_per_pair_path = args.results_dir / "dissertacao_table_top_terms_per_pair.csv"
    summary_path = args.results_dir / "dissertacao_interpretation_summary.md"

    top_terms_per_pair = build_top_terms_per_pair(adjacent, global_shift, top_n=5)

    pair_summary.to_csv(pair_summary_path, index=False, encoding="utf-8")
    institutional.to_csv(institutional_path, index=False, encoding="utf-8")
    candidates.to_csv(candidates_path, index=False, encoding="utf-8")
    candidate_neighbors.to_csv(neighbors_path, index=False, encoding="utf-8")
    alignment.to_csv(alignment_path, index=False, encoding="utf-8")
    top_terms_per_pair.to_csv(top_terms_per_pair_path, index=False, encoding="utf-8")
    write_markdown_summary(summary_path, pair_summary, institutional, candidates)

    pair_fig = args.figures_dir / "dissertacao_mean_drift_by_pair.png"
    institutional_fig = args.figures_dir / "dissertacao_institutional_drift_heatmap.png"
    plot_pair_summary(pair_summary, pair_fig)
    plot_institutional_heatmap(institutional, institutional_fig)

    manifest = {
        "inputs": {
            "adjacent": str(args.results_dir / "dissertacao_semantic_shift_adjacent.csv"),
            "global": str(args.results_dir / "dissertacao_semantic_shift_52_57.csv"),
            "top_terms": str(args.results_dir / "dissertacao_semantic_shift_top_terms.csv"),
            "neighbors": str(args.results_dir / "dissertacao_semantic_shift_neighbors.csv"),
            "alignment": str(args.results_dir / "dissertacao_procrustes_alignment_metrics.csv"),
        },
        "outputs": {
            "pair_summary": str(pair_summary_path),
            "institutional_terms": str(institutional_path),
            "qualitative_candidates": str(candidates_path),
            "candidate_neighbors": str(neighbors_path),
            "alignment_quality": str(alignment_path),
            "markdown_summary": str(summary_path),
            "pair_figure": str(pair_fig),
            "institutional_figure": str(institutional_fig),
        },
        "notes": {
            "paper_focus": "longitudinal semantic drift in parliamentary discourse",
            "focus_transition_pairs": sorted(PAPER_FOCUS_PAIRS),
            "all_transition_pairs_flagged": sorted(TRANSITION_PAIRS),
            "term_classes": [
                "institutional_concept",
                "political_actor_or_name",
                "parliamentary_procedure",
                "discursive_marker",
                "candidate_content_term",
            ],
        },
    }
    (args.results_dir / "dissertacao_interpretation_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"Saved pair drift table to: {pair_summary_path}")
    print(f"Saved institutional term table to: {institutional_path}")
    print(f"Saved qualitative candidate table to: {candidates_path}")
    print(f"Saved candidate neighbor table to: {neighbors_path}")
    print(f"Saved top-5 terms per pair table to: {top_terms_per_pair_path}")
    print(f"Saved interpretation summary to: {summary_path}")
    print(f"Saved figures to: {args.figures_dir}")


if __name__ == "__main__":
    main(parse_args())
