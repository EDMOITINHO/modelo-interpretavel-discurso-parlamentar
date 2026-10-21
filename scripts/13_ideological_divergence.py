"""
13_ideological_divergence.py
Measure semantic divergence between Left and Right ideological blocks across legislatures.

Method:
  For each legislature L:
    1. Train Word2Vec independently on Left_L and Right_L corpora.
    2. Align Left_L onto Right_L using Orthogonal Procrustes (shared vocabulary).
    3. Compute cosine distance per shared term between aligned Left_L and Right_L.
    4. Mean across shared terms = ideological divergence score for legislature L.

Input:
  data/dissertacao_preprocessado/by_legislature_block/leg{N}_left.txt
  data/dissertacao_preprocessado/by_legislature_block/leg{N}_right.txt

Outputs:
  modelos/dissertacao_word2vec_blocks/  (trained block models)
  resultados/dissertacao_ideological_block_training_metrics.csv
  resultados/dissertacao_ideological_divergence_by_legislature.csv
  resultados/dissertacao_ideological_divergence_per_term.csv
  resultados/dissertacao_ideological_top_divergent_terms.csv
  figuras/dissertacao/dissertacao_ideological_divergence_trend.png
  figuras/dissertacao/dissertacao_ideological_divergence_heatmap.png
  resultados/dissertacao_ideological_divergence_manifest.json
"""

from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from pathlib import Path
from typing import Iterator

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from gensim.models import KeyedVectors, Word2Vec
from scipy.linalg import orthogonal_procrustes


TARGET_LEGS = [52, 53, 54, 55, 56, 57]
BLOCKS = ["left", "right"]

DEFAULT_BLOCK_DIR = Path("data/dissertacao_preprocessado/by_legislature_block")
DEFAULT_MODELS_DIR = Path("modelos/dissertacao_word2vec_blocks")
DEFAULT_RESULTS_DIR = Path("resultados")
DEFAULT_FIGURES_DIR = Path("figuras/dissertacao")

# Word2Vec hyperparameters — same as 02_train_word2vec.py
VECTOR_SIZE = 300
WINDOW = 8
EPOCHS = 10
NEGATIVE = 10
SAMPLE = 1e-4
SEED = 42
# Lower min_count for block corpora (smaller than full legislature corpora)
MIN_COUNT_BLOCK = 10

MIN_TOKENS_WARN = 500_000   # warn below this; training still proceeds
MIN_SHARED_VOCAB = 1_000    # skip a legislature pair if fewer shared terms

PAIR_CONTEXT = {
    52: "Leg 52\n(Lula I)",
    53: "Leg 53\n(Lula II)",
    54: "Leg 54\n(Dilma I)",
    55: "Leg 55\n(Dilma II /\nTemer)",
    56: "Leg 56\n(Bolsonaro)",
    57: "Leg 57\n(Lula III)",
}


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Measure ideological divergence (Left vs Right) across legislatures."
    )
    parser.add_argument("--block-dir", type=Path, default=DEFAULT_BLOCK_DIR)
    parser.add_argument("--models-dir", type=Path, default=DEFAULT_MODELS_DIR)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--figures-dir", type=Path, default=DEFAULT_FIGURES_DIR)
    parser.add_argument("--legs", type=int, nargs="+", default=TARGET_LEGS)
    parser.add_argument("--vector-size", type=int, default=VECTOR_SIZE)
    parser.add_argument("--window", type=int, default=WINDOW)
    parser.add_argument("--min-count", type=int, default=MIN_COUNT_BLOCK)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--negative", type=int, default=NEGATIVE)
    parser.add_argument("--sample", type=float, default=SAMPLE)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--max-anchors", type=int, default=15_000)
    parser.add_argument("--top-divergent", type=int, default=30,
                        help="Top-N most divergent terms to save per legislature.")
    parser.add_argument("--force-retrain", action="store_true",
                        help="Retrain even if saved model exists.")
    return parser.parse_args()


# ─────────────────────────────────────────────
# Corpus helpers
# ─────────────────────────────────────────────

class LineSentenceCorpus:
    def __init__(self, path: Path):
        self.path = path

    def __iter__(self) -> Iterator[list[str]]:
        with self.path.open("r", encoding="utf-8") as f:
            for line in f:
                tokens = line.strip().split()
                if tokens:
                    yield tokens


def stage_to_local(path: Path) -> Path:
    cache = Path(tempfile.gettempdir()) / "dissertacao_block_w2v"
    cache.mkdir(parents=True, exist_ok=True)
    local = cache / path.name
    if not local.exists() or local.stat().st_size != path.stat().st_size:
        shutil.copyfile(path, local)
    return local


def count_lines_tokens(path: Path) -> tuple[int, int]:
    lines, tokens = 0, 0
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if parts:
                lines += 1
                tokens += len(parts)
    return lines, tokens


# ─────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────

def train_or_load(
    path: Path,
    model_path: Path,
    kv_path: Path,
    args: argparse.Namespace,
    force: bool = False,
) -> tuple[KeyedVectors, dict]:
    if kv_path.exists() and not force:
        print(f"  [cache] Loading {kv_path.name}")
        kv = KeyedVectors.load(str(kv_path), mmap="r")
        speeches, tokens = count_lines_tokens(path)
        metrics = {
            "speeches": speeches,
            "tokens": tokens,
            "vocab_size": len(kv),
            "source": "cache",
        }
        return kv, metrics

    local_path = stage_to_local(path)
    speeches, tokens = count_lines_tokens(local_path)

    if tokens < MIN_TOKENS_WARN:
        print(f"  [WARNING] Only {tokens:,} tokens — results may be unreliable.")

    model = Word2Vec(
        sentences=LineSentenceCorpus(local_path),
        vector_size=args.vector_size,
        window=args.window,
        min_count=args.min_count,
        workers=args.workers,
        sg=1,
        hs=0,
        negative=args.negative,
        sample=args.sample,
        epochs=args.epochs,
        seed=args.seed,
        sorted_vocab=1,
    )
    model.save(str(model_path))
    model.wv.save(str(kv_path))

    metrics = {
        "speeches": speeches,
        "tokens": tokens,
        "vocab_size": len(model.wv),
        "source": "trained",
    }
    return model.wv, metrics


# ─────────────────────────────────────────────
# Procrustes alignment and divergence
# ─────────────────────────────────────────────

def unit_rows(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return matrix / norms


def shared_vocab(kv_a: KeyedVectors, kv_b: KeyedVectors, max_anchors: int) -> list[str]:
    set_a = set(kv_a.index_to_key)
    set_b = set(kv_b.index_to_key)
    common = set_a & set_b
    # order by frequency in kv_a (more frequent = better anchor)
    ranked = [t for t in kv_a.index_to_key if t in common]
    return ranked[:max_anchors]


def align_and_measure(
    kv_source: KeyedVectors,
    kv_target: KeyedVectors,
    anchors: list[str],
) -> tuple[np.ndarray, list[str], dict]:
    """
    Align kv_source onto kv_target via Procrustes.
    Returns per-term cosine distances, the term list, and alignment diagnostics.
    """
    x = unit_rows(np.array([kv_source[t] for t in anchors], dtype=np.float64))
    y = unit_rows(np.array([kv_target[t] for t in anchors], dtype=np.float64))

    cos_before = float(np.mean(np.sum(x * y, axis=1)))
    rotation, _ = orthogonal_procrustes(x, y)
    x_aligned = x @ rotation
    cos_after = float(np.mean(np.sum(x_aligned * y, axis=1)))

    cosine_distances = 1.0 - np.sum(x_aligned * y, axis=1)

    diagnostics = {
        "n_anchors": len(anchors),
        "cosine_similarity_before": round(cos_before, 6),
        "cosine_similarity_after": round(cos_after, 6),
        "alignment_gain": round(cos_after - cos_before, 6),
    }
    return cosine_distances, anchors, diagnostics


# ─────────────────────────────────────────────
# Main analysis
# ─────────────────────────────────────────────

def run_divergence_analysis(
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Returns:
        training_metrics  — one row per (leg, block)
        by_legislature    — divergence score per legislature
        per_term          — cosine distance per (leg, term)
    """
    training_rows = []
    leg_rows = []
    term_rows = []

    for leg in args.legs:
        left_path = args.block_dir / f"leg{leg}_left.txt"
        right_path = args.block_dir / f"leg{leg}_right.txt"

        if not left_path.exists() or not right_path.exists():
            print(f"[WARNING] Missing block files for legislature {leg} — skipping.")
            continue

        print(f"\n── Legislature {leg} ──────────────────────")

        for block, path in [("left", left_path), ("right", right_path)]:
            model_path = args.models_dir / f"w2v_leg{leg}_{block}.model"
            kv_path = args.models_dir / f"w2v_leg{leg}_{block}.kv"
            print(f"  [{block}] {path.name}")
            kv, metrics = train_or_load(
                path, model_path, kv_path, args, force=args.force_retrain
            )
            metrics.update({"id_legislatura": leg, "block": block})
            training_rows.append(metrics)

        # reload both KVs for alignment
        kv_left = KeyedVectors.load(
            str(args.models_dir / f"w2v_leg{leg}_left.kv"), mmap="r"
        )
        kv_right = KeyedVectors.load(
            str(args.models_dir / f"w2v_leg{leg}_right.kv"), mmap="r"
        )

        anchors = shared_vocab(kv_left, kv_right, args.max_anchors)
        if len(anchors) < MIN_SHARED_VOCAB:
            print(f"  [WARNING] Only {len(anchors)} shared terms — skipping legislature {leg}.")
            continue

        print(f"  Shared vocabulary for alignment: {len(anchors):,} terms")

        distances, terms, diag = align_and_measure(kv_left, kv_right, anchors)
        mean_dist = float(distances.mean())
        median_dist = float(np.median(distances))

        print(
            f"  Left→Right divergence: mean={mean_dist:.4f}  median={median_dist:.4f}  "
            f"(alignment gain: {diag['alignment_gain']:.4f})"
        )

        leg_rows.append(
            {
                "id_legislatura": leg,
                "pair_label": PAIR_CONTEXT.get(leg, str(leg)),
                "n_shared_terms": len(anchors),
                "mean_lr_divergence": round(mean_dist, 6),
                "median_lr_divergence": round(median_dist, 6),
                **diag,
            }
        )

        for term, dist in zip(terms, distances):
            term_rows.append(
                {
                    "id_legislatura": leg,
                    "term": term,
                    "cosine_distance_lr": round(float(dist), 6),
                }
            )

    return (
        pd.DataFrame(training_rows),
        pd.DataFrame(leg_rows).sort_values("id_legislatura"),
        pd.DataFrame(term_rows),
    )


def build_top_divergent(per_term: pd.DataFrame, top_n: int) -> pd.DataFrame:
    """Top-N most divergent terms per legislature, for paper-quality table."""
    return (
        per_term.sort_values(["id_legislatura", "cosine_distance_lr"], ascending=[True, False])
        .groupby("id_legislatura", as_index=False)
        .head(top_n)
        .sort_values(["id_legislatura", "cosine_distance_lr"], ascending=[True, False])
    )


# ─────────────────────────────────────────────
# Figures
# ─────────────────────────────────────────────

def plot_divergence_trend(by_leg: pd.DataFrame, out_path: Path) -> None:
    """Main figure: Left–Right ideological divergence across legislatures."""
    legs = by_leg["id_legislatura"].tolist()
    means = by_leg["mean_lr_divergence"].values
    medians = by_leg["median_lr_divergence"].values

    fig, ax = plt.subplots(figsize=(8.5, 4.8))

    ax.plot(legs, means, marker="o", color="#2c6fad", linewidth=2.2,
            markersize=7, label="Mean L–R divergence", zorder=3)
    ax.plot(legs, medians, marker="s", color="#2c6fad", linewidth=1.5,
            markersize=5, linestyle="--", alpha=0.7, label="Median L–R divergence", zorder=3)

    # annotate each point
    for leg, mean in zip(legs, means):
        ax.text(leg, mean + 0.003, f"{mean:.3f}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(legs)
    ax.set_xticklabels([PAIR_CONTEXT.get(l, str(l)) for l in legs], fontsize=8)
    ax.set_ylabel("Mean cosine distance (Left vs Right)", fontsize=11)
    ax.set_title(
        "Ideological semantic divergence between Left and Right blocks\n"
        "by legislature",
        fontsize=11,
    )
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_divergence_heatmap(per_term: pd.DataFrame, top_n: int, out_path: Path) -> None:
    """Heatmap of top-N terms by divergence across legislatures."""
    legs = sorted(per_term["id_legislatura"].unique())

    # pick terms that are consistently among the most divergent
    top_terms_all = (
        per_term.groupby("term", as_index=False)["cosine_distance_lr"]
        .mean()
        .sort_values("cosine_distance_lr", ascending=False)
        .head(top_n)["term"]
        .tolist()
    )

    pivot = (
        per_term[per_term["term"].isin(top_terms_all)]
        .pivot_table(index="term", columns="id_legislatura", values="cosine_distance_lr")
        .reindex(columns=legs)
        .fillna(0)
    )
    pivot = pivot.loc[pivot.mean(axis=1).sort_values(ascending=False).index]

    fig, ax = plt.subplots(figsize=(8, max(5.0, 0.32 * len(pivot))))
    im = ax.imshow(pivot.values, aspect="auto", cmap="YlOrRd")
    ax.set_xticks(np.arange(len(legs)), labels=[str(l) for l in legs])
    ax.set_yticks(np.arange(len(pivot.index)), labels=pivot.index, fontsize=8)
    ax.set_xlabel("Legislature")
    ax.set_title(
        f"Left–Right semantic divergence: top {top_n} most divergent terms",
        fontsize=11,
    )
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            val = pivot.iloc[i, j]
            if val > 0:
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                        fontsize=6.5, color="black" if val < 0.5 else "white")
    fig.colorbar(im, ax=ax, label="Cosine distance (Left vs Right)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


# ─────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────

def main(args: argparse.Namespace) -> None:
    args.models_dir.mkdir(parents=True, exist_ok=True)
    args.results_dir.mkdir(parents=True, exist_ok=True)
    args.figures_dir.mkdir(parents=True, exist_ok=True)

    if not args.block_dir.exists():
        raise FileNotFoundError(
            f"Block directory not found: {args.block_dir}\n"
            "Run 01_preprocess_corpus.py first."
        )

    training_metrics, by_leg, per_term = run_divergence_analysis(args)

    if by_leg.empty:
        print("\n[ERROR] No legislature pairs computed. Check block file availability.")
        return

    top_divergent = build_top_divergent(per_term, args.top_divergent)

    paths = {
        "training_metrics": args.results_dir / "dissertacao_ideological_block_training_metrics.csv",
        "by_legislature": args.results_dir / "dissertacao_ideological_divergence_by_legislature.csv",
        "per_term": args.results_dir / "dissertacao_ideological_divergence_per_term.csv",
        "top_divergent": args.results_dir / "dissertacao_ideological_top_divergent_terms.csv",
        "trend_figure": args.figures_dir / "dissertacao_ideological_divergence_trend.png",
        "heatmap_figure": args.figures_dir / "dissertacao_ideological_divergence_heatmap.png",
        "manifest": args.results_dir / "dissertacao_ideological_divergence_manifest.json",
    }

    training_metrics.to_csv(paths["training_metrics"], index=False, encoding="utf-8")
    by_leg.to_csv(paths["by_legislature"], index=False, encoding="utf-8")
    per_term.to_csv(paths["per_term"], index=False, encoding="utf-8")
    top_divergent.to_csv(paths["top_divergent"], index=False, encoding="utf-8")

    plot_divergence_trend(by_leg, paths["trend_figure"])
    plot_divergence_heatmap(per_term, args.top_divergent, paths["heatmap_figure"])

    manifest = {
        "purpose": "Left vs Right ideological semantic divergence across legislatures",
        "method": "Word2Vec per block per legislature + Orthogonal Procrustes alignment + cosine distance",
        "parameters": {
            "vector_size": args.vector_size,
            "window": args.window,
            "min_count": args.min_count,
            "epochs": args.epochs,
            "negative": args.negative,
            "sample": args.sample,
            "seed": args.seed,
            "max_anchors": args.max_anchors,
        },
        "blocks": BLOCKS,
        "legs": args.legs,
        "outputs": {k: str(v) for k, v in paths.items()},
    }
    paths["manifest"].write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n── Summary: Left–Right divergence by legislature ──────────────")
    print(
        by_leg[["id_legislatura", "n_shared_terms", "mean_lr_divergence", "median_lr_divergence",
                 "alignment_gain"]]
        .to_string(index=False, float_format="{:.4f}".format)
    )
    for k, v in paths.items():
        print(f"Saved {k}: {v}")


if __name__ == "__main__":
    main(parse_args())
