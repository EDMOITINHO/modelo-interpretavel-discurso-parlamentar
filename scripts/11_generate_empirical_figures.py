"""
Generate complementary empirical figures for the dissertation.

Adaptations:
Fig. 1 -> overview of the computational characterization workflow.
Fig. 2 -> corpus scale: speeches by party and monthly speech activity.
Fig. 3 -> theme sea: major national themes across years.
Fig. 4 -> party-year involvement in selected themes.
Fig. 6 -> clustering of party-year observations in sensitive-group discourse.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_PREPROC_DIR = Path("data/dissertacao_preprocessado")
DEFAULT_RESULTS_DIR = Path("resultados")
DEFAULT_FIGURES_DIR = Path("figuras/dissertacao/empirical_figures")
TARGET_LEGS = [52, 53, 54, 55, 56, 57]

THEMES = {
    "institutions": {
        "label": "Democracy and institutions",
        "terms": {"democracia", "constituicao", "congresso", "camara", "senado", "governo", "voto", "judiciario"},
    },
    "federalism": {
        "label": "Federalism and municipalities",
        "terms": {"municipio", "municipios", "prefeito", "prefeitos", "estado", "estados", "federacao", "regional"},
    },
    "economy": {
        "label": "Economy, work, and income",
        "terms": {"economia", "emprego", "trabalho", "renda", "salario", "inflacao", "juros", "credito"},
    },
    "rights": {
        "label": "Rights and citizenship",
        "terms": {"direito", "direitos", "cidadania", "igualdade", "liberdade", "justica"},
    },
    "social_policy": {
        "label": "Social protection",
        "terms": {"pobreza", "fome", "desigualdade", "assistencia", "previdencia", "beneficio"},
    },
    "security": {
        "label": "Public security",
        "terms": {"seguranca", "violencia", "crime", "policia", "armas", "trafico", "homicidio"},
    },
    "education": {
        "label": "Education",
        "terms": {"educacao", "escola", "ensino", "professor", "professores", "universidade", "fundeb"},
    },
    "health": {
        "label": "Health",
        "terms": {"saude", "sus", "hospital", "medico", "medicos", "vacina", "pandemia", "covid"},
    },
    "environment": {
        "label": "Environment",
        "terms": {"ambiente", "ambiental", "amazonia", "floresta", "desmatamento", "clima", "agua"},
    },
    "infrastructure": {
        "label": "Infrastructure",
        "terms": {"infraestrutura", "transporte", "rodovia", "estrada", "energia", "saneamento", "internet"},
    },
}

SENSITIVE_GROUPS = {
    "workers": {
        "label": "Workers/unions",
        "terms": {"trabalhador", "trabalhadores", "trabalhadora", "trabalhadoras", "sindicato", "sindicatos", "greve", "servidor", "servidores"},
    },
    "lgbt_gender": {
        "label": "LGBT/gender",
        "terms": {"lgbt", "gay", "gays", "lesbica", "lesbicas", "homossexual", "transexual", "travesti", "travestis", "genero", "sexualidade", "homofobia", "transfobia"},
    },
    "women": {
        "label": "Women",
        "terms": {"mulher", "mulheres", "feminismo", "feminista", "feministas"},
    },
    "racial_indigenous": {
        "label": "Racial/indigenous",
        "terms": {"negro", "negros", "negra", "negras", "racismo", "indigena", "indigenas", "quilombola", "quilombolas"},
    },
    "religious_moral": {
        "label": "Religious/moral",
        "terms": {"religiao", "religioso", "religiosos", "igreja", "igrejas", "evangelicos", "catolicos", "aborto", "familia"},
    },
    "security_forces": {
        "label": "Security forces",
        "terms": {"policia", "policial", "policiais", "militar", "militares", "bombeiros", "guarda", "forcas", "armadas"},
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate complementary empirical figures for the dissertation.")
    parser.add_argument("--preproc-dir", type=Path, default=DEFAULT_PREPROC_DIR)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--figures-dir", type=Path, default=DEFAULT_FIGURES_DIR)
    parser.add_argument("--top-parties", type=int, default=12)
    parser.add_argument("--clusters", type=int, default=6)
    return parser.parse_args()


def read_speeches(preproc_dir: Path) -> pd.DataFrame:
    metadata = pd.read_csv(preproc_dir / "corpus_dissertacao_metadata.csv", parse_dates=["data"])
    frames = []
    for leg in TARGET_LEGS:
        path = preproc_dir / "by_legislature" / f"leg{leg}.txt"
        lines = path.read_text(encoding="utf-8").splitlines()
        sub = metadata[metadata["id_legislatura"] == leg].copy().reset_index(drop=True)
        if len(sub) != len(lines):
            raise ValueError(f"Line count mismatch for leg{leg}: metadata={len(sub)} text={len(lines)}")
        sub["clean_text"] = lines
        frames.append(sub)
    df = pd.concat(frames, ignore_index=True)
    df["year"] = df["data"].dt.year
    df["month"] = df["data"].dt.to_period("M").astype(str)
    df["party"] = df["partido_norm"].fillna(df["sigla_partido"]).fillna("UNKNOWN").astype(str)
    return df


def count_hits(tokens: list[str], dictionary: dict[str, set[str]]) -> dict[str, int]:
    counts = Counter(tokens)
    return {name: sum(counts[t] for t in spec["terms"]) for name, spec in dictionary.items()}


def collect_stats(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    corpus_rows = []
    theme_year = defaultdict(lambda: {"speeches": 0, "hits": 0, "with_theme": 0, "tokens": 0})
    party_year_theme = defaultdict(lambda: {"speeches": 0, "hits": 0, "with_theme": 0, "tokens": 0})
    party_year_sensitive = defaultdict(lambda: {"speeches": 0, "hits": 0, "with_group": 0, "tokens": 0})

    for _, row in df.iterrows():
        tokens = str(row["clean_text"]).split()
        n_tokens = len(tokens)
        year = int(row["year"]) if pd.notna(row["year"]) else -1
        month = str(row["month"])
        party = str(row["party"])
        leg = int(row["id_legislatura"])
        theme_hits = count_hits(tokens, THEMES)
        sensitive_hits = count_hits(tokens, SENSITIVE_GROUPS)

        corpus_rows.append(
            {
                "year": year,
                "month": month,
                "legislature": leg,
                "party": party,
                "tokens": n_tokens,
            }
        )

        for theme, hits in theme_hits.items():
            for bucket, key in (
                (theme_year, (year, theme)),
                (party_year_theme, (party, year, theme)),
            ):
                bucket[key]["speeches"] += 1
                bucket[key]["hits"] += hits
                bucket[key]["with_theme"] += int(hits > 0)
                bucket[key]["tokens"] += n_tokens

        for group, hits in sensitive_hits.items():
            key = (party, year, group)
            party_year_sensitive[key]["speeches"] += 1
            party_year_sensitive[key]["hits"] += hits
            party_year_sensitive[key]["with_group"] += int(hits > 0)
            party_year_sensitive[key]["tokens"] += n_tokens

    return (
        pd.DataFrame(corpus_rows),
        bucket_frame(theme_year, ["year", "theme"], THEMES, "theme"),
        bucket_frame(party_year_theme, ["party", "year", "theme"], THEMES, "theme"),
        bucket_frame(party_year_sensitive, ["party", "year", "group"], SENSITIVE_GROUPS, "group"),
    )


def bucket_frame(bucket: dict, group_cols: list[str], dictionary: dict, item_col: str) -> pd.DataFrame:
    rows = []
    for key, stats in bucket.items():
        item = key[-1]
        speeches = stats["speeches"]
        tokens = stats["tokens"]
        rows.append(
            {
                **dict(zip(group_cols, key)),
                "label": dictionary[item]["label"],
                "speeches": speeches,
                "with_item": stats["with_theme"] if "with_theme" in stats else stats["with_group"],
                "hits": stats["hits"],
                "speech_share_pct": 100 * (stats["with_theme"] if "with_theme" in stats else stats["with_group"]) / speeches if speeches else 0,
                "hits_per_10k_tokens": 10000 * stats["hits"] / tokens if tokens else 0,
            }
        )
    return pd.DataFrame(rows)


def save_fig1_overview(out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.axis("off")
    boxes = [
        ("Corpus", "439,627 speeches\nLegislatures 52--57\nParty/year metadata"),
        ("Layer 1", "National-policy\nagenda coverage"),
        ("Layer 2", "National vs.\nsectorized claims"),
        ("Layer 3", "Institutional\nsemantic instability"),
        ("Layer 4", "Sensitive-group\nconflict audit"),
        ("Outputs", "Figures, tables,\ncluster maps,\ninterpretive summaries"),
    ]
    x_positions = np.linspace(0.06, 0.86, len(boxes))
    colors = ["#d9e8f5", "#e9f2dc", "#e9f2dc", "#f6e4c8", "#f3d8da", "#e6e0f0"]
    for i, ((title, body), x, color) in enumerate(zip(boxes, x_positions, colors)):
        rect = plt.Rectangle((x, 0.42), 0.13, 0.28, facecolor=color, edgecolor="#4a4a4a", linewidth=1.1)
        ax.add_patch(rect)
        ax.text(x + 0.065, 0.63, title, ha="center", va="center", fontsize=10, fontweight="bold")
        ax.text(x + 0.065, 0.52, body, ha="center", va="center", fontsize=8)
        if i < len(boxes) - 1:
            ax.annotate("", xy=(x_positions[i + 1] - 0.01, 0.56), xytext=(x + 0.14, 0.56),
                        arrowprops=dict(arrowstyle="->", color="#4a4a4a", lw=1.2))
    ax.text(0.5, 0.82, "Computational characterization of parliamentary discourse as a social system",
            ha="center", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_fig2_scale(corpus: pd.DataFrame, out_path: Path, top_parties: int) -> pd.DataFrame:
    party_counts = corpus.groupby("party", as_index=False).agg(speeches=("party", "size")).sort_values("speeches", ascending=False)
    top = party_counts.head(top_parties)
    monthly = corpus.groupby("month", as_index=False).agg(speeches=("month", "size"))
    monthly["date"] = pd.to_datetime(monthly["month"] + "-01")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    axes[0].barh(top["party"][::-1], top["speeches"][::-1], color="#4378a8")
    axes[0].set_title("(a) Speeches by party")
    axes[0].set_xlabel("Speeches")
    axes[0].grid(axis="x", alpha=0.25)

    axes[1].plot(monthly["date"], monthly["speeches"], color="#a84f43", linewidth=1.3)
    axes[1].set_title("(b) Monthly parliamentary speech activity")
    axes[1].set_xlabel("Month")
    axes[1].set_ylabel("Speeches")
    axes[1].grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    return party_counts


def save_fig3_theme_sea(theme_year: pd.DataFrame, out_path: Path) -> None:
    data = theme_year[(theme_year["year"] > 0)].copy()
    order = list(THEMES.keys())
    y_map = {theme: i for i, theme in enumerate(order)}
    data["y"] = data["theme"].map(y_map)
    sizes = 20 + 8 * np.sqrt(data["with_item"].clip(lower=0))

    fig, ax = plt.subplots(figsize=(12, 6.2))
    scatter = ax.scatter(
        data["year"],
        data["y"],
        s=sizes,
        c=data["speech_share_pct"],
        cmap="viridis",
        alpha=0.72,
        edgecolors="none",
    )
    ax.set_yticks(range(len(order)), [THEMES[t]["label"] for t in order])
    ax.set_xlabel("Year")
    ax.set_title("Theme sea: major national-policy themes across years")
    ax.grid(axis="x", alpha=0.25)
    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label("% of speeches mentioning theme")
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def save_fig4_party_theme_heatmaps(party_year_theme: pd.DataFrame, corpus: pd.DataFrame, out_path: Path, top_parties: int) -> None:
    top_parties_list = corpus["party"].value_counts().head(top_parties).index.tolist()
    selected = ["economy", "education", "health", "security", "rights", "environment"]
    data = party_year_theme[
        party_year_theme["party"].isin(top_parties_list) & party_year_theme["theme"].isin(selected)
    ].copy()
    years = sorted(y for y in data["year"].unique() if y > 0)

    fig, axes = plt.subplots(2, 3, figsize=(13, 8), sharex=True, sharey=True)
    for ax, theme in zip(axes.ravel(), selected):
        pivot = (
            data[data["theme"] == theme]
            .pivot(index="party", columns="year", values="speech_share_pct")
            .reindex(top_parties_list)
            .reindex(columns=years)
            .fillna(0)
        )
        im = ax.imshow(pivot.values, aspect="auto", cmap="YlOrRd", vmin=0, vmax=np.nanpercentile(data["speech_share_pct"], 95))
        ax.set_title(THEMES[theme]["label"], fontsize=10)
        ax.set_xticks(range(len(years)), years, rotation=90, fontsize=6)
        ax.set_yticks(range(len(top_parties_list)), top_parties_list, fontsize=7)
    fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.72, label="% of party-year speeches")
    fig.suptitle("Party-year involvement in selected national-policy themes", y=0.98, fontsize=13)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def kmeans(X: np.ndarray, k: int, iterations: int = 100) -> np.ndarray:
    rng = np.random.default_rng(42)
    if len(X) < k:
        return np.zeros(len(X), dtype=int)
    centers = X[rng.choice(len(X), size=k, replace=False)]
    labels = np.zeros(len(X), dtype=int)
    for _ in range(iterations):
        distances = ((X[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
        new_labels = distances.argmin(axis=1)
        if np.array_equal(new_labels, labels):
            break
        labels = new_labels
        for i in range(k):
            if np.any(labels == i):
                centers[i] = X[labels == i].mean(axis=0)
    return labels


def pca2(X: np.ndarray) -> np.ndarray:
    Xc = X - X.mean(axis=0)
    _, _, vt = np.linalg.svd(Xc, full_matrices=False)
    return Xc @ vt[:2].T


def save_fig6_sensitive_clusters(party_year_sensitive: pd.DataFrame, corpus: pd.DataFrame, out_path: Path, clusters: int) -> pd.DataFrame:
    party_totals = corpus["party"].value_counts()
    keep_parties = party_totals[party_totals >= 500].index
    data = party_year_sensitive[party_year_sensitive["party"].isin(keep_parties)].copy()
    party_year_counts = (
        corpus.groupby(["party", "year"], as_index=False)
        .agg(party_year_speeches=("party", "size"))
    )
    pivot = data.pivot_table(index=["party", "year"], columns="group", values="speech_share_pct", fill_value=0)
    pivot = pivot.reindex(columns=list(SENSITIVE_GROUPS.keys()), fill_value=0)
    valid = pivot[pivot.index.get_level_values("year") > 0].copy()
    count_map = party_year_counts.set_index(["party", "year"])["party_year_speeches"]
    valid = valid[count_map.reindex(valid.index).fillna(0) >= 100]
    X = valid.values.astype(float)
    X_scaled = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-9)
    labels = kmeans(X_scaled, clusters)
    coords = pca2(X_scaled)
    out = valid.reset_index()
    out["cluster"] = labels
    out["pc1"] = coords[:, 0]
    out["pc2"] = coords[:, 1]

    fig, ax = plt.subplots(figsize=(9, 6.5))
    scatter = ax.scatter(out["pc1"], out["pc2"], c=out["cluster"], cmap="tab10", s=35, alpha=0.78)
    for _, row in out.sort_values("pc1").iloc[:: max(1, len(out)//25)].iterrows():
        ax.text(row["pc1"], row["pc2"], f"{row['party']}-{int(row['year'])}", fontsize=6, alpha=0.75)
    ax.set_title("Clustering of party-year profiles in sensitive-group discourse")
    ax.set_xlabel("PCA dimension 1")
    ax.set_ylabel("PCA dimension 2")
    ax.grid(alpha=0.25)
    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label("Cluster")
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    return out


def main() -> None:
    args = parse_args()
    args.results_dir.mkdir(parents=True, exist_ok=True)
    args.figures_dir.mkdir(parents=True, exist_ok=True)

    df = read_speeches(args.preproc_dir)
    corpus, theme_year, party_year_theme, party_year_sensitive = collect_stats(df)

    paths = {
        "fig1": args.figures_dir / "fig1_overview_workflow.png",
        "fig2": args.figures_dir / "fig2_corpus_scale.png",
        "fig3": args.figures_dir / "fig3_theme_sea_by_year.png",
        "fig4": args.figures_dir / "fig4_party_year_theme_involvement.png",
        "fig6": args.figures_dir / "fig6_sensitive_theme_clusters.png",
        "corpus_scale": args.results_dir / "dissertacao_figure_data_2_corpus_scale.csv",
        "theme_year": args.results_dir / "dissertacao_figure_data_3_theme_year.csv",
        "party_year_theme": args.results_dir / "dissertacao_figure_data_4_party_year_theme.csv",
        "sensitive_clusters": args.results_dir / "dissertacao_figure_data_6_sensitive_clusters.csv",
        "manifest": args.results_dir / "dissertacao_empirical_figures_manifest.json",
    }

    save_fig1_overview(paths["fig1"])
    party_counts = save_fig2_scale(corpus, paths["fig2"], args.top_parties)
    save_fig3_theme_sea(theme_year, paths["fig3"])
    save_fig4_party_theme_heatmaps(party_year_theme, corpus, paths["fig4"], args.top_parties)
    clusters = save_fig6_sensitive_clusters(party_year_sensitive, corpus, paths["fig6"], args.clusters)

    party_counts.to_csv(paths["corpus_scale"], index=False, encoding="utf-8")
    theme_year.to_csv(paths["theme_year"], index=False, encoding="utf-8")
    party_year_theme.to_csv(paths["party_year_theme"], index=False, encoding="utf-8")
    clusters.to_csv(paths["sensitive_clusters"], index=False, encoding="utf-8")

    manifest = {
        "purpose": "Complementary empirical figures for the dissertation",
        "n_speeches": int(df.shape[0]),
        "outputs": {k: str(v) for k, v in paths.items()},
        "figure_mapping": {
            "fig1": "overview workflow",
            "fig2": "corpus scale by party and month",
            "fig3": "theme sea across years",
            "fig4": "party-year involvement in selected themes",
            "fig6": "clustering of party-year sensitive-group discourse profiles",
        },
    }
    paths["manifest"].write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Analyzed speeches: {df.shape[0]}")
    for key in ["fig1", "fig2", "fig3", "fig4", "fig6"]:
        print(f"Saved {key}: {paths[key]}")


if __name__ == "__main__":
    main()
