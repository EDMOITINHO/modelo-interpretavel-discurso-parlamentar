"""
Align Word2Vec models from Dissertation across legislatures with Orthogonal Procrustes.

This script assumes that 02_train_word2vec.py has already produced
KeyedVectors files at:

    modelos/dissertacao_word2vec/w2v_leg52.kv
    ...
    modelos/dissertacao_word2vec/w2v_leg57.kv

It aligns each legislature to the previous one and writes aligned vectors plus
diagnostic metrics for adjacent legislature pairs.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from gensim.models import KeyedVectors
from scipy.linalg import orthogonal_procrustes


TARGET_LEGS = [52, 53, 54, 55, 56, 57]

DEFAULT_MODELS_DIR = Path("modelos/dissertacao_word2vec")
DEFAULT_OUTPUT_DIR = Path("modelos/dissertacao_word2vec_aligned")
DEFAULT_RESULTS_DIR = Path("resultados")
DEFAULT_ANCHORS_PATH = DEFAULT_MODELS_DIR / "anchor_vocab_top20000_all_legs.txt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Align Dissertation Word2Vec models with Orthogonal Procrustes."
    )
    parser.add_argument(
        "--models-dir",
        type=Path,
        default=DEFAULT_MODELS_DIR,
        help="Directory containing w2v_legXX.kv files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where aligned KeyedVectors will be saved.",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=DEFAULT_RESULTS_DIR,
        help="Directory where alignment metrics will be saved.",
    )
    parser.add_argument(
        "--anchors-path",
        type=Path,
        default=DEFAULT_ANCHORS_PATH,
        help="Text file with one anchor term per line.",
    )
    parser.add_argument(
        "--legs",
        type=int,
        nargs="+",
        default=TARGET_LEGS,
        help="Legislatures to align, in chronological order.",
    )
    parser.add_argument(
        "--max-anchors",
        type=int,
        default=20_000,
        help="Maximum number of anchor words used in each adjacent alignment.",
    )
    parser.add_argument(
        "--min-anchors",
        type=int,
        default=5_000,
        help="Minimum anchor count required for a pairwise alignment.",
    )
    parser.add_argument(
        "--save-pairwise",
        action="store_true",
        help="Also save models aligned only to the immediately previous legislature.",
    )
    return parser.parse_args()


def load_kv(path: Path) -> KeyedVectors:
    if not path.exists():
        raise FileNotFoundError(f"Missing model file: {path}")
    return KeyedVectors.load(str(path), mmap="r")


def read_anchor_terms(path: Path) -> list[str]:
    if not path.exists():
        raise FileNotFoundError(f"Missing anchor vocabulary file: {path}")

    terms = []
    seen = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        term = line.strip()
        if term and term not in seen:
            terms.append(term)
            seen.add(term)
    return terms


def shared_anchors(
    anchors: Iterable[str],
    source: KeyedVectors,
    target: KeyedVectors,
    max_anchors: int,
) -> list[str]:
    selected = []
    for term in anchors:
        if term in source.key_to_index and term in target.key_to_index:
            selected.append(term)
            if len(selected) >= max_anchors:
                break
    return selected


def unit_rows(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return matrix / norms


def cosine_mean(a: np.ndarray, b: np.ndarray) -> float:
    a_norm = unit_rows(a)
    b_norm = unit_rows(b)
    return float(np.mean(np.sum(a_norm * b_norm, axis=1)))


def copy_with_vectors(template: KeyedVectors, vectors: np.ndarray) -> KeyedVectors:
    aligned = KeyedVectors(vector_size=template.vector_size)
    words = [str(word) for word in template.index_to_key]
    aligned.add_vectors(words, vectors.astype(np.float32))
    return aligned


def transform_vectors(model: KeyedVectors, rotation: np.ndarray) -> np.ndarray:
    vectors = np.asarray(model.vectors, dtype=np.float64)
    transformed = vectors @ rotation
    return unit_rows(transformed)


def fit_rotation(
    source: KeyedVectors,
    target: KeyedVectors,
    anchors: list[str],
) -> tuple[np.ndarray, dict[str, float]]:
    x = unit_rows(np.asarray([source[word] for word in anchors], dtype=np.float64))
    y = unit_rows(np.asarray([target[word] for word in anchors], dtype=np.float64))

    before_cosine = cosine_mean(x, y)
    rotation, scale = orthogonal_procrustes(x, y)
    x_aligned = x @ rotation
    after_cosine = cosine_mean(x_aligned, y)

    residual = float(np.linalg.norm(x_aligned - y, ord="fro") / np.sqrt(len(anchors)))

    metrics = {
        "anchors": float(len(anchors)),
        "cosine_before": before_cosine,
        "cosine_after": after_cosine,
        "cosine_gain": after_cosine - before_cosine,
        "procrustes_scale": float(scale),
        "rmse_frobenius": residual,
    }
    return rotation, metrics


def save_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def main(args: argparse.Namespace) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.results_dir.mkdir(parents=True, exist_ok=True)

    anchors = read_anchor_terms(args.anchors_path)
    if not anchors:
        raise ValueError(f"No anchors found in {args.anchors_path}")

    models = {
        leg: load_kv(args.models_dir / f"w2v_leg{leg}.kv")
        for leg in args.legs
    }

    base_leg = args.legs[0]
    base_model = models[base_leg]
    base_vectors = unit_rows(np.asarray(base_model.vectors, dtype=np.float64))
    copy_with_vectors(base_model, base_vectors).save(
        str(args.output_dir / f"w2v_leg{base_leg}_aligned_to_leg{base_leg}.kv")
    )

    cumulative_rotations: dict[int, np.ndarray] = {
        base_leg: np.eye(base_model.vector_size, dtype=np.float64)
    }
    rows = []

    for previous_leg, current_leg in zip(args.legs[:-1], args.legs[1:]):
        previous_model = models[previous_leg]
        current_model = models[current_leg]

        pair_anchors = shared_anchors(
            anchors,
            source=current_model,
            target=previous_model,
            max_anchors=args.max_anchors,
        )
        if len(pair_anchors) < args.min_anchors:
            raise ValueError(
                f"Only {len(pair_anchors)} anchors available for "
                f"leg{current_leg}->leg{previous_leg}; minimum is {args.min_anchors}."
            )

        rotation_to_previous, metrics = fit_rotation(
            source=current_model,
            target=previous_model,
            anchors=pair_anchors,
        )

        cumulative_rotations[current_leg] = (
            rotation_to_previous @ cumulative_rotations[previous_leg]
        )

        pairwise_vectors = transform_vectors(current_model, rotation_to_previous)
        pairwise_model = copy_with_vectors(current_model, pairwise_vectors)

        if args.save_pairwise:
            pairwise_model.save(
                str(
                    args.output_dir
                    / f"w2v_leg{current_leg}_aligned_to_leg{previous_leg}.kv"
                )
            )

        cumulative_vectors = transform_vectors(
            current_model,
            cumulative_rotations[current_leg],
        )
        cumulative_model = copy_with_vectors(current_model, cumulative_vectors)
        cumulative_model.save(
            str(args.output_dir / f"w2v_leg{current_leg}_aligned_to_leg{base_leg}.kv")
        )

        row = {
            "source_leg": current_leg,
            "target_leg": previous_leg,
            "base_leg": base_leg,
            "source_vocab": len(current_model.index_to_key),
            "target_vocab": len(previous_model.index_to_key),
            "anchor_file": str(args.anchors_path),
            "max_anchors": args.max_anchors,
            **metrics,
        }
        rows.append(row)

        print(
            f"Aligned leg{current_leg} -> leg{previous_leg}: "
            f"anchors={int(metrics['anchors'])}, "
            f"cosine {metrics['cosine_before']:.4f} -> {metrics['cosine_after']:.4f}"
        )

    metrics_path = args.results_dir / "dissertacao_procrustes_alignment_metrics.csv"
    pd.DataFrame(rows).to_csv(metrics_path, index=False, encoding="utf-8")

    manifest = {
        "legs": args.legs,
        "base_leg": base_leg,
        "models_dir": str(args.models_dir),
        "output_dir": str(args.output_dir),
        "anchors_path": str(args.anchors_path),
        "max_anchors": args.max_anchors,
        "min_anchors": args.min_anchors,
        "metrics_path": str(metrics_path),
        "alignment": "adjacent orthogonal Procrustes with cumulative rotations to base_leg",
    }
    save_json(args.output_dir / "alignment_manifest.json", manifest)

    print(f"\nSaved aligned vectors to: {args.output_dir}")
    print(f"Saved metrics to: {metrics_path}")


if __name__ == "__main__":
    main(parse_args())
