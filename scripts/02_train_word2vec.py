# ============================================================
# 02_train_word2vec.py
# Train independent Word2Vec models by legislature
# Dissertation — Semantic drift in Brazilian parliamentary speeches
# ============================================================

from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from pathlib import Path
from typing import Iterator

import pandas as pd
from gensim.models import Word2Vec


# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------

INPUT_DIR = Path("data") / "dissertacao_preprocessado" / "by_legislature"
OUT_DIR = Path("modelos") / "dissertacao_word2vec"
RESULTS_DIR = Path("resultados")

TARGET_LEGS = [52, 53, 54, 55, 56, 57]

DEFAULT_VECTOR_SIZE = 300
DEFAULT_WINDOW = 8
DEFAULT_MIN_COUNT = 20
DEFAULT_EPOCHS = 10
DEFAULT_NEGATIVE = 10
DEFAULT_SAMPLE = 1e-4
DEFAULT_SEED = 42


# ------------------------------------------------------------
# Corpus iterator
# ------------------------------------------------------------

class LineSentenceCorpus:
    """Memory-safe iterator over one-tokenized-speech-per-line text files."""

    def __init__(self, path: Path):
        self.path = path

    def __iter__(self) -> Iterator[list[str]]:
        with self.path.open("r", encoding="utf-8") as f:
            for line in f:
                tokens = line.strip().split()
                if tokens:
                    yield tokens


def stage_input_file(path: Path) -> Path:
    """Copy a Google Drive input file to local temporary storage for Gensim."""
    cache_dir = Path(tempfile.gettempdir()) / "legislatura_dissertacao_word2vec"
    cache_dir.mkdir(parents=True, exist_ok=True)
    local_path = cache_dir / path.name

    if not local_path.exists() or local_path.stat().st_size != path.stat().st_size:
        print(f"  Copying corpus to local temporary storage: {local_path}")
        shutil.copyfile(path, local_path)

    return local_path


def count_lines_and_tokens(path: Path) -> tuple[int, int]:
    lines = 0
    tokens = 0
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if parts:
                lines += 1
                tokens += len(parts)
    return lines, tokens


def train_model(path: Path, args: argparse.Namespace) -> tuple[Word2Vec, dict]:
    local_path = stage_input_file(path)
    corpus = LineSentenceCorpus(local_path)
    speeches, tokens = count_lines_and_tokens(local_path)

    model = Word2Vec(
        sentences=corpus,
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
        compute_loss=args.compute_loss,
    )

    training_loss = None
    if args.compute_loss:
        training_loss = float(model.get_latest_training_loss())

    metrics = {
        "speeches": speeches,
        "tokens": tokens,
        "vocab_size": len(model.wv),
        "vector_size": args.vector_size,
        "window": args.window,
        "min_count": args.min_count,
        "epochs": args.epochs,
        "negative": args.negative,
        "sample": args.sample,
        "seed": args.seed,
        "training_loss": training_loss,
    }
    return model, metrics


def adjacent_vocab_overlap(models: dict[int, Word2Vec]) -> pd.DataFrame:
    rows = []
    for left, right in zip(TARGET_LEGS[:-1], TARGET_LEGS[1:]):
        vocab_left = set(models[left].wv.index_to_key)
        vocab_right = set(models[right].wv.index_to_key)
        common = vocab_left & vocab_right
        union = vocab_left | vocab_right
        rows.append(
            {
                "legislature_from": left,
                "legislature_to": right,
                "vocab_from": len(vocab_left),
                "vocab_to": len(vocab_right),
                "common_vocab": len(common),
                "union_vocab": len(union),
                "jaccard": round(len(common) / len(union), 6) if union else 0.0,
                "common_vocab_pct_from": round(100.0 * len(common) / len(vocab_left), 3)
                if vocab_left
                else 0.0,
                "common_vocab_pct_to": round(100.0 * len(common) / len(vocab_right), 3)
                if vocab_right
                else 0.0,
            }
        )
    return pd.DataFrame(rows)


def save_anchor_vocab(models: dict[int, Word2Vec], min_rank: int, path: Path) -> None:
    """Save a stable anchor vocabulary shared by all legislatures.

    The later Procrustes step can use either adjacent shared vocabularies or
    this stricter global anchor set. min_rank keeps only terms that are frequent
    enough in every model.
    """

    ranked_sets = []
    for model in models.values():
        ranked_sets.append(set(model.wv.index_to_key[:min_rank]))

    anchors = sorted(set.intersection(*ranked_sets))
    path.write_text("\n".join(anchors) + "\n", encoding="utf-8")


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

def main(args: argparse.Namespace) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    models: dict[int, Word2Vec] = {}
    metrics_rows = []

    for leg in TARGET_LEGS:
        path = args.input_dir / f"leg{leg}.txt"
        if not path.exists():
            raise FileNotFoundError(f"Missing preprocessed file: {path}")

        print(f"\nTraining Word2Vec for legislature {leg}: {path}")
        model, metrics = train_model(path, args)

        model_path = OUT_DIR / f"w2v_leg{leg}.model"
        vectors_path = OUT_DIR / f"w2v_leg{leg}.kv"
        vocab_path = OUT_DIR / f"vocab_leg{leg}.txt"

        model.save(str(model_path))
        model.wv.save(str(vectors_path))
        vocab_path.write_text("\n".join(str(token) for token in model.wv.index_to_key) + "\n", encoding="utf-8",)

        metrics["id_legislatura"] = leg
        metrics["model_path"] = str(model_path)
        metrics["vectors_path"] = str(vectors_path)
        metrics["vocab_path"] = str(vocab_path)
        metrics_rows.append(metrics)
        models[leg] = model

        print(
            f"  speeches={metrics['speeches']:,} "
            f"tokens={metrics['tokens']:,} "
            f"vocab={metrics['vocab_size']:,}"
        )

    metrics_df = pd.DataFrame(metrics_rows)
    metrics_path = RESULTS_DIR / "dissertacao_word2vec_training_metrics.csv"
    metrics_df.to_csv(metrics_path, index=False, encoding="utf-8-sig")

    overlap = adjacent_vocab_overlap(models)
    overlap_path = RESULTS_DIR / "dissertacao_word2vec_adjacent_vocab_overlap.csv"
    overlap.to_csv(overlap_path, index=False, encoding="utf-8-sig")

    anchors_path = OUT_DIR / f"anchor_vocab_top{args.anchor_topn}_all_legs.txt"
    save_anchor_vocab(models, args.anchor_topn, anchors_path)

    manifest = {
        "input_dir": str(args.input_dir),
        "output_dir": str(OUT_DIR),
        "target_legislatures": TARGET_LEGS,
        "algorithm": "Word2Vec skip-gram with negative sampling",
        "parameters": {
            "vector_size": args.vector_size,
            "window": args.window,
            "min_count": args.min_count,
            "epochs": args.epochs,
            "workers": args.workers,
            "negative": args.negative,
            "sample": args.sample,
            "seed": args.seed,
        },
        "outputs": {
            "training_metrics": str(metrics_path),
            "adjacent_vocab_overlap": str(overlap_path),
            "anchor_vocab": str(anchors_path),
        },
    }
    manifest_path = RESULTS_DIR / "dissertacao_word2vec_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("\nTraining finished.")
    print(f"Models: {OUT_DIR}")
    print(f"Metrics: {metrics_path}")
    print(f"Adjacent overlap: {overlap_path}")
    print(f"Global anchor vocabulary: {anchors_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train independent Word2Vec models for Dissertation."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=INPUT_DIR,
        help=f"Directory containing legXX.txt files. Default: {INPUT_DIR}",
    )
    parser.add_argument("--vector-size", type=int, default=DEFAULT_VECTOR_SIZE)
    parser.add_argument("--window", type=int, default=DEFAULT_WINDOW)
    parser.add_argument("--min-count", type=int, default=DEFAULT_MIN_COUNT)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--negative", type=int, default=DEFAULT_NEGATIVE)
    parser.add_argument("--sample", type=float, default=DEFAULT_SAMPLE)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--compute-loss",
        action="store_true",
        help="Ask gensim to compute training loss. This can slow training.",
    )
    parser.add_argument(
        "--anchor-topn",
        type=int,
        default=20_000,
        help="Top-N frequent terms per model used to build global anchor vocabulary.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
