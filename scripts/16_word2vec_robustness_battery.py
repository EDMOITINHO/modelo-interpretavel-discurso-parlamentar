# ============================================================
# 16_word2vec_robustness_battery.py
# Robustness battery for the semantic-drift finding (Workstream C)
# Robustness battery for the semantic-drift analysis in the dissertation.
#
# Checks, for the six legislatures 52-57:
#   A) multi-seed:        retrain with seed in {7, 123} (baseline uses seed=42)
#   B) min_count:          retrain with min_count in {50, 100} (baseline uses 20)
#   C) null model:         split each legislature's corpus into two random
#                          halves, train each half, align, measure "internal"
#                          same-legislature drift as a noise floor
#   D) token-balanced:     subsample every legislature to ~9.27M tokens
#                          (legislature 57's size) before training
#
# For A/B/D: after training the 6-legislature set for a config, aligns
# adjacent pairs via orthogonal Procrustes (fresh anchor vocabulary per
# config) and reports mean cosine-distance drift per adjacent pair --
# directly comparable to the baseline's 0.3999 (52-53) ... 0.5047 (56-57).
#
# Writes incrementally to
#   resultados/dissertacao_word2vec_robustness_battery.csv
# so partial progress survives interruption.
# ============================================================

from __future__ import annotations

import random
import shutil
import tempfile
import time
from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd
from gensim.models import Word2Vec, KeyedVectors
from scipy.linalg import orthogonal_procrustes

TARGET_LEGS = [52, 53, 54, 55, 56, 57]

INPUT_DIR = Path("data") / "dissertacao_preprocessado" / "by_legislature"

# Everything below reads/writes to LOCAL disk, not the Google-Drive-backed
# project directory. Training reads the corpus once per epoch (10 epochs),
# and streaming that repeatedly off a Drive-synced path stalled badly after
# several hours (confirmed: process alive, CPU time not advancing). Only the
# small final results CSV is written back to the project's resultados/ dir.
LOCAL_CACHE_DIR = Path(tempfile.gettempdir()) / "legislatura_dissertacao_robustness"
STAGED_INPUT_DIR = LOCAL_CACHE_DIR / "staged_input"
SCRATCH_DIR = LOCAL_CACHE_DIR / "scratch"
OUT_DIR = LOCAL_CACHE_DIR / "modelos"
RESULTS_PATH = Path("resultados") / "dissertacao_word2vec_robustness_battery.csv"


def stage_input_file(leg: int) -> Path:
    """Copy a legislature's corpus file from the Drive-backed project dir to
    local disk once, reusing the local copy on subsequent calls."""
    src = INPUT_DIR / f"leg{leg}.txt"
    dst = STAGED_INPUT_DIR / f"leg{leg}.txt"
    if not dst.exists() or dst.stat().st_size != src.stat().st_size:
        dst.parent.mkdir(parents=True, exist_ok=True)
        print(f"  staging leg{leg}.txt to local disk...")
        shutil.copyfile(src, dst)
    return dst

VECTOR_SIZE = 300
WINDOW = 8
EPOCHS = 10
NEGATIVE = 10
SAMPLE = 1e-4
WORKERS = 8
BASELINE_MIN_COUNT = 20
BASELINE_SEED = 42
TOKEN_BUDGET = 9_269_792  # legislature 57's token count


# ------------------------------------------------------------
# Corpus helpers
# ------------------------------------------------------------

class LineSentenceCorpus:
    def __init__(self, path: Path):
        self.path = path

    def __iter__(self) -> Iterator[list[str]]:
        with self.path.open("r", encoding="utf-8") as f:
            for line in f:
                tokens = line.strip().split()
                if tokens:
                    yield tokens


def read_lines(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8") as f:
        return [line for line in f if line.strip()]


def write_lines(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.writelines(lines)


def split_half(leg: int, seed: int = BASELINE_SEED) -> tuple[Path, Path]:
    out_a = SCRATCH_DIR / f"leg{leg}_half_a.txt"
    out_b = SCRATCH_DIR / f"leg{leg}_half_b.txt"
    if out_a.exists() and out_b.exists():
        return out_a, out_b
    lines = read_lines(stage_input_file(leg))
    rng = random.Random(seed)
    idx = list(range(len(lines)))
    rng.shuffle(idx)
    mid = len(idx) // 2
    a_lines = [lines[i] for i in idx[:mid]]
    b_lines = [lines[i] for i in idx[mid:]]
    write_lines(out_a, a_lines)
    write_lines(out_b, b_lines)
    return out_a, out_b


def subsample_to_budget(leg: int, token_budget: int, seed: int = BASELINE_SEED) -> Path:
    out_path = SCRATCH_DIR / f"leg{leg}_balanced_{token_budget}.txt"
    if out_path.exists():
        return out_path
    lines = read_lines(stage_input_file(leg))
    rng = random.Random(seed)
    idx = list(range(len(lines)))
    rng.shuffle(idx)
    # Take shuffled speeches in order and stop once the budget is reached
    # (allowing slight overshoot on the last speech). Skipping speeches that
    # would overshoot -- instead of just stopping -- would systematically
    # bias the subsample toward shorter speeches.
    kept = []
    total = 0
    for i in idx:
        kept.append(lines[i])
        total += len(lines[i].split())
        if total >= token_budget:
            break
    write_lines(out_path, kept)
    return out_path


# ------------------------------------------------------------
# Training
# ------------------------------------------------------------

def train_one(path: Path, seed: int, min_count: int) -> KeyedVectors:
    corpus = LineSentenceCorpus(path)
    model = Word2Vec(
        sentences=corpus,
        vector_size=VECTOR_SIZE,
        window=WINDOW,
        min_count=min_count,
        workers=WORKERS,
        sg=1,
        hs=0,
        negative=NEGATIVE,
        sample=SAMPLE,
        epochs=EPOCHS,
        seed=seed,
        sorted_vocab=1,
    )
    return model.wv


# ------------------------------------------------------------
# Alignment + drift (fresh anchors per config, no dependency on
# the baseline pipeline's saved anchor file)
# ------------------------------------------------------------

def unit_rows(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return matrix / norms


def build_anchor_vocab(models: dict, top_n: int = 15_000) -> list[str]:
    ranked_sets = [set(kv.index_to_key[:top_n]) for kv in models.values()]
    return sorted(set.intersection(*ranked_sets))


def shared_terms(anchors: list[str], source: KeyedVectors, target: KeyedVectors, max_terms: int) -> list[str]:
    out = []
    for term in anchors:
        if term in source.key_to_index and term in target.key_to_index:
            out.append(term)
            if len(out) >= max_terms:
                break
    return out


def mean_adjacent_drift(source: KeyedVectors, target: KeyedVectors, terms: list[str]) -> float:
    x = unit_rows(np.asarray([source[t] for t in terms], dtype=np.float64))
    y = unit_rows(np.asarray([target[t] for t in terms], dtype=np.float64))
    rotation, _ = orthogonal_procrustes(x, y)
    x_aligned = unit_rows(x @ rotation)
    cosine_sim = np.sum(x_aligned * y, axis=1)
    cosine_dist = 1.0 - cosine_sim
    return float(np.mean(cosine_dist))


def adjacent_pair_drift(models: dict, legs_in_order: list, max_terms: int = 15_000, min_terms: int = 500) -> list[dict]:
    anchors = build_anchor_vocab(models)
    rows = []
    for source_leg, target_leg in zip(legs_in_order[1:], legs_in_order[:-1]):
        terms = shared_terms(anchors, models[source_leg], models[target_leg], max_terms)
        if len(terms) < min_terms:
            print(f"    WARNING: only {len(terms)} shared terms for {source_leg}->{target_leg}, skipping")
            continue
        drift = mean_adjacent_drift(models[source_leg], models[target_leg], terms)
        rows.append({"pair": f"{target_leg}-{source_leg}", "n_terms": len(terms), "mean_drift": drift})
    return rows


# ------------------------------------------------------------
# Result logging
# ------------------------------------------------------------

def append_results(rows: list[dict]) -> None:
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    if RESULTS_PATH.exists():
        df.to_csv(RESULTS_PATH, mode="a", header=False, index=False, encoding="utf-8-sig")
    else:
        df.to_csv(RESULTS_PATH, index=False, encoding="utf-8-sig")
    print(df.to_string(index=False))


def already_done(config: str) -> bool:
    if not RESULTS_PATH.exists():
        return False
    df = pd.read_csv(RESULTS_PATH)
    return config in set(df["config"])


# ------------------------------------------------------------
# Battery stages
# ------------------------------------------------------------

def run_full_corpus_config(config_name: str, seed: int, min_count: int) -> None:
    if already_done(config_name):
        print(f"[skip] {config_name} already in results")
        return
    t0 = time.perf_counter()
    print(f"\n=== {config_name} (seed={seed}, min_count={min_count}) ===")
    models = {}
    for leg in TARGET_LEGS:
        t_leg = time.perf_counter()
        models[leg] = train_one(stage_input_file(leg), seed=seed, min_count=min_count)
        print(f"  leg{leg}: vocab={len(models[leg]):,} ({time.perf_counter()-t_leg:.0f}s)")
        models[leg].save(str(OUT_DIR / f"{config_name}_leg{leg}.kv"))
    rows = adjacent_pair_drift(models, TARGET_LEGS)
    for r in rows:
        r["config"] = config_name
        r["seed"] = seed
        r["min_count"] = min_count
    append_results(rows)
    print(f"=== {config_name} done in {time.perf_counter()-t0:.0f}s ===")


def run_null_model() -> None:
    for leg in TARGET_LEGS:
        config_name = f"null_leg{leg}"
        if already_done(config_name):
            print(f"[skip] {config_name} already in results")
            continue
        t0 = time.perf_counter()
        print(f"\n=== {config_name} (same-legislature split-half baseline) ===")
        path_a, path_b = split_half(leg)
        kv_a = train_one(path_a, seed=BASELINE_SEED, min_count=BASELINE_MIN_COUNT)
        kv_b = train_one(path_b, seed=BASELINE_SEED, min_count=BASELINE_MIN_COUNT)
        kv_a.save(str(OUT_DIR / f"{config_name}_half_a.kv"))
        kv_b.save(str(OUT_DIR / f"{config_name}_half_b.kv"))
        models = {"a": kv_a, "b": kv_b}
        rows = adjacent_pair_drift(models, ["a", "b"])
        for r in rows:
            r["config"] = config_name
            r["seed"] = BASELINE_SEED
            r["min_count"] = BASELINE_MIN_COUNT
            r["pair"] = f"leg{leg}_internal"
        append_results(rows)
        print(f"=== {config_name} done in {time.perf_counter()-t0:.0f}s ===")


def run_token_balanced() -> None:
    config_name = "token_balanced"
    if already_done(config_name):
        print(f"[skip] {config_name} already in results")
        return
    t0 = time.perf_counter()
    print(f"\n=== {config_name} (all legislatures subsampled to {TOKEN_BUDGET:,} tokens) ===")
    models = {}
    for leg in TARGET_LEGS:
        t_leg = time.perf_counter()
        path = subsample_to_budget(leg, TOKEN_BUDGET)
        models[leg] = train_one(path, seed=BASELINE_SEED, min_count=BASELINE_MIN_COUNT)
        print(f"  leg{leg}: vocab={len(models[leg]):,} ({time.perf_counter()-t_leg:.0f}s)")
        models[leg].save(str(OUT_DIR / f"{config_name}_leg{leg}.kv"))
    rows = adjacent_pair_drift(models, TARGET_LEGS)
    for r in rows:
        r["config"] = config_name
        r["seed"] = BASELINE_SEED
        r["min_count"] = BASELINE_MIN_COUNT
    append_results(rows)
    print(f"=== {config_name} done in {time.perf_counter()-t0:.0f}s ===")


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    SCRATCH_DIR.mkdir(parents=True, exist_ok=True)

    battery_start = time.perf_counter()

    # A) multi-seed (baseline seed=42 already exists in main pipeline outputs)
    run_full_corpus_config("seed7", seed=7, min_count=BASELINE_MIN_COUNT)
    run_full_corpus_config("seed123", seed=123, min_count=BASELINE_MIN_COUNT)

    # B) min_count sensitivity
    run_full_corpus_config("min_count50", seed=BASELINE_SEED, min_count=50)
    run_full_corpus_config("min_count100", seed=BASELINE_SEED, min_count=100)

    # C) null model (same-legislature split-half noise floor)
    run_null_model()

    # D) token-balanced subsample
    run_token_balanced()

    print(f"\nFULL BATTERY DONE in {(time.perf_counter()-battery_start)/3600:.2f}h")
    print(f"Results: {RESULTS_PATH}")


if __name__ == "__main__":
    main()
