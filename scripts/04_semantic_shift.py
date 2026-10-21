"""
Measure semantic shift in aligned Word2Vec models for Dissertation.

Expected input from 03_align_procrustes.py:

    modelos/dissertacao_word2vec_aligned/w2v_leg52_aligned_to_leg52.kv
    modelos/dissertacao_word2vec_aligned/w2v_leg53_aligned_to_leg52.kv
    ...
    modelos/dissertacao_word2vec_aligned/w2v_leg57_aligned_to_leg52.kv

Outputs:

    resultados/dissertacao_semantic_shift_adjacent.csv
    resultados/dissertacao_semantic_shift_52_57.csv
    resultados/dissertacao_semantic_shift_top_terms.csv
    resultados/dissertacao_semantic_shift_neighbors.csv
    resultados/dissertacao_semantic_shift_manifest.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from gensim.models import KeyedVectors


TARGET_LEGS = [52, 53, 54, 55, 56, 57]

DEFAULT_ALIGNED_DIR = Path("modelos/dissertacao_word2vec_aligned")
DEFAULT_RESULTS_DIR = Path("resultados")
DEFAULT_ANCHORS_PATH = Path("modelos/dissertacao_word2vec/anchor_vocab_top20000_all_legs.txt")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute semantic shift metrics from aligned Dissertation Word2Vec models."
    )
    parser.add_argument(
        "--aligned-dir",
        type=Path,
        default=DEFAULT_ALIGNED_DIR,
        help="Directory containing aligned KeyedVectors from script 03.",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=DEFAULT_RESULTS_DIR,
        help="Directory where CSV outputs will be saved.",
    )
    parser.add_argument(
        "--anchors-path",
        type=Path,
        default=DEFAULT_ANCHORS_PATH,
        help="Anchor vocabulary file used to restrict the default comparison vocabulary.",
    )
    parser.add_argument(
        "--legs",
        type=int,
        nargs="+",
        default=TARGET_LEGS,
        help="Legislatures to compare, in chronological order.",
    )
    parser.add_argument(
        "--base-leg",
        type=int,
        default=52,
        help="Base legislature used in aligned model filenames.",
    )
    parser.add_argument(
        "--vocab-source",
        choices=["anchors", "intersection"],
        default="anchors",
        help="Use anchor file or full vocabulary intersection as comparison vocabulary.",
    )
    parser.add_argument(
        "--max-terms",
        type=int,
        default=20_000,
        help="Maximum number of terms to evaluate from the selected vocabulary.",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=200,
        help="Number of top shifted terms saved in the compact top-terms file.",
    )
    parser.add_argument(
        "--neighbors-top-n",
        type=int,
        default=50,
        help="Number of most shifted terms for which nearest neighbors will be exported.",
    )
    parser.add_argument(
        "--neighbor-k",
        type=int,
        default=10,
        help="Number of nearest neighbors exported per selected term and legislature.",
    )
    return parser.parse_args()


def load_kv(path: Path) -> KeyedVectors:
    if not path.exists():
        raise FileNotFoundError(f"Missing aligned model file: {path}")
    return KeyedVectors.load(str(path), mmap="r")


def read_terms(path: Path) -> list[str]:
    if not path.exists():
        raise FileNotFoundError(f"Missing vocabulary file: {path}")

    terms = []
    seen = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        term = line.strip()
        if term and term not in seen:
            terms.append(term)
            seen.add(term)
    return terms


def model_path(aligned_dir: Path, leg: int, base_leg: int) -> Path:
    return aligned_dir / f"w2v_leg{leg}_aligned_to_leg{base_leg}.kv"


def common_terms_from_anchors(
    anchors_path: Path,
    models: dict[int, KeyedVectors],
    max_terms: int,
) -> list[str]:
    candidates = read_terms(anchors_path)
    return common_terms(candidates, models, max_terms)


def common_terms_from_intersection(
    models: dict[int, KeyedVectors],
    max_terms: int,
) -> list[str]:
    base_leg = sorted(models)[0]
    candidates = [str(term) for term in models[base_leg].index_to_key]
    return common_terms(candidates, models, max_terms)


def common_terms(
    candidates: list[str],
    models: dict[int, KeyedVectors],
    max_terms: int,
) -> list[str]:
    selected = []
    for term in candidates:
        if all(term in model.key_to_index for model in models.values()):
            selected.append(term)
            if len(selected) >= max_terms:
                break
    return selected


def unit_vector(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm == 0.0:
        return vector
    return vector / norm


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a_unit = unit_vector(a)
    b_unit = unit_vector(b)
    return float(np.dot(a_unit, b_unit))


def displacement(source: KeyedVectors, target: KeyedVectors, term: str) -> dict[str, float]:
    source_vector = np.asarray(source[term], dtype=np.float64)
    target_vector = np.asarray(target[term], dtype=np.float64)
    cosine = cosine_similarity(source_vector, target_vector)
    return {
        "cosine_similarity": cosine,
        "cosine_distance": 1.0 - cosine,
        "euclidean_distance": float(np.linalg.norm(source_vector - target_vector)),
    }


def compute_pair_shift(
    models: dict[int, KeyedVectors],
    terms: list[str],
    source_leg: int,
    target_leg: int,
) -> pd.DataFrame:
    rows = []
    source = models[source_leg]
    target = models[target_leg]

    for term in terms:
        metrics = displacement(source, target, term)
        rows.append(
            {
                "term": term,
                "source_leg": source_leg,
                "target_leg": target_leg,
                **metrics,
            }
        )

    return pd.DataFrame(rows).sort_values(
        ["cosine_distance", "euclidean_distance"],
        ascending=[False, False],
    )


def aggregate_adjacent_shift(adjacent: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        adjacent.groupby("term", as_index=False)
        .agg(
            mean_cosine_distance=("cosine_distance", "mean"),
            max_cosine_distance=("cosine_distance", "max"),
            sum_cosine_distance=("cosine_distance", "sum"),
            mean_euclidean_distance=("euclidean_distance", "mean"),
            max_euclidean_distance=("euclidean_distance", "max"),
        )
        .sort_values(
            ["mean_cosine_distance", "max_cosine_distance"],
            ascending=[False, False],
        )
    )
    return grouped


def nearest_neighbors(
    models: dict[int, KeyedVectors],
    terms: list[str],
    neighbor_k: int,
) -> pd.DataFrame:
    rows = []
    for term in terms:
        for leg, model in models.items():
            if term not in model.key_to_index:
                continue
            neighbors = model.most_similar(term, topn=neighbor_k)
            rows.append(
                {
                    "term": term,
                    "leg": leg,
                    "neighbors": "; ".join(
                        f"{str(neighbor)}:{score:.4f}"
                        for neighbor, score in neighbors
                    ),
                }
            )
    return pd.DataFrame(rows)


def save_manifest(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def main(args: argparse.Namespace) -> None:
    args.results_dir.mkdir(parents=True, exist_ok=True)

    models = {
        leg: load_kv(model_path(args.aligned_dir, leg, args.base_leg))
        for leg in args.legs
    }

    if args.vocab_source == "anchors":
        terms = common_terms_from_anchors(args.anchors_path, models, args.max_terms)
    else:
        terms = common_terms_from_intersection(models, args.max_terms)

    if not terms:
        raise ValueError("No common terms found across aligned models.")

    print(f"Common terms selected: {len(terms)}")

    adjacent_frames = []
    for source_leg, target_leg in zip(args.legs[:-1], args.legs[1:]):
        frame = compute_pair_shift(models, terms, source_leg, target_leg)
        adjacent_frames.append(frame)
        print(
            f"Computed semantic shift leg{source_leg}->leg{target_leg}: "
            f"{len(frame)} terms"
        )

    adjacent = pd.concat(adjacent_frames, ignore_index=True)
    adjacent_path = args.results_dir / "dissertacao_semantic_shift_adjacent.csv"
    adjacent.to_csv(adjacent_path, index=False, encoding="utf-8")

    first_leg = args.legs[0]
    last_leg = args.legs[-1]
    global_shift = compute_pair_shift(models, terms, first_leg, last_leg)
    global_path = args.results_dir / f"dissertacao_semantic_shift_{first_leg}_{last_leg}.csv"
    global_shift.to_csv(global_path, index=False, encoding="utf-8")

    adjacent_summary = aggregate_adjacent_shift(adjacent)
    top_terms = (
        adjacent_summary.head(args.top_n)
        .merge(
            global_shift[
                ["term", "cosine_distance", "euclidean_distance"]
            ].rename(
                columns={
                    "cosine_distance": f"cosine_distance_{first_leg}_{last_leg}",
                    "euclidean_distance": f"euclidean_distance_{first_leg}_{last_leg}",
                }
            ),
            on="term",
            how="left",
        )
    )
    top_terms_path = args.results_dir / "dissertacao_semantic_shift_top_terms.csv"
    top_terms.to_csv(top_terms_path, index=False, encoding="utf-8")

    neighbor_terms = top_terms["term"].head(args.neighbors_top_n).tolist()
    neighbors = nearest_neighbors(models, neighbor_terms, args.neighbor_k)
    neighbors_path = args.results_dir / "dissertacao_semantic_shift_neighbors.csv"
    neighbors.to_csv(neighbors_path, index=False, encoding="utf-8")

    manifest = {
        "legs": args.legs,
        "base_leg": args.base_leg,
        "aligned_dir": str(args.aligned_dir),
        "vocab_source": args.vocab_source,
        "anchors_path": str(args.anchors_path),
        "max_terms": args.max_terms,
        "terms_used": len(terms),
        "top_n": args.top_n,
        "neighbors_top_n": args.neighbors_top_n,
        "neighbor_k": args.neighbor_k,
        "outputs": {
            "adjacent": str(adjacent_path),
            "global": str(global_path),
            "top_terms": str(top_terms_path),
            "neighbors": str(neighbors_path),
        },
    }
    save_manifest(args.results_dir / "dissertacao_semantic_shift_manifest.json", manifest)

    print(f"\nSaved adjacent shift ranking to: {adjacent_path}")
    print(f"Saved global shift ranking to: {global_path}")
    print(f"Saved compact top-term table to: {top_terms_path}")
    print(f"Saved neighbor table to: {neighbors_path}")


if __name__ == "__main__":
    main(parse_args())
