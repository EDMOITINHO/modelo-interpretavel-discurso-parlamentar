# ============================================================
# 15_adversarial_validation_sample.py
# Stratified manual-annotation sample for the adversarial audit layer
# Manual-validation sample for the social-groups audit layer.
#
# Draws a stratified random sample from the candidate-adversarial pool
# produced by 10_adversarial_group_discourse.py, stratified by
# target group x legislature, and exports readable text windows for
# offline manual coding (attack / defense / neutral / ambiguous).
#
# This script prepares the stratified sample used for manual annotation.
# The final dissertation summarizes the returned labels descriptively as
# attack, defense, neutral, or ambiguous; this is not a classifier evaluation.
# ============================================================

from __future__ import annotations

import random
from pathlib import Path

import pandas as pd

TARGET_LEGS = [52, 53, 54, 55, 56, 57]
PREPROC_DIR = Path("data/dissertacao_preprocessado")
RESULTS_DIR = Path("resultados")
WINDOW = 12  # must match 10_adversarial_group_discourse.py

TARGET_GROUPS_TERMS = {
    "workers": {
        "trabalhador", "trabalhadores", "trabalhadora", "trabalhadoras",
        "sindicato", "sindicatos", "sindical", "sindicais", "greve",
        "greves", "servidor", "servidores", "funcionario", "funcionarios",
    },
    "lgbt_identity": {
        "lgbt", "lgbtq", "gay", "gays", "lesbica", "lesbicas",
        "homossexual", "homossexuais", "transexual", "transexuais",
        "transgenero", "transgeneros", "travesti", "travestis",
        "genero", "sexualidade", "orientacao", "homofobia", "transfobia",
    },
    "women": {"mulher", "mulheres", "feminismo", "feminista", "feministas"},
    "racial_indigenous": {
        "negro", "negros", "negra", "negras", "racismo", "indigena",
        "indigenas", "quilombola", "quilombolas",
    },
}

TARGET_PER_GROUP = 150
RANDOM_SEED = 42


def read_speeches() -> pd.DataFrame:
    metadata = pd.read_csv(PREPROC_DIR / "corpus_dissertacao_metadata.csv")
    frames = []
    for leg in TARGET_LEGS:
        lines = (PREPROC_DIR / "by_legislature" / f"leg{leg}.txt").read_text(encoding="utf-8").splitlines()
        sub = metadata[metadata["id_legislatura"] == leg].copy().reset_index(drop=True)
        if len(sub) != len(lines):
            raise ValueError(f"Line count mismatch for leg{leg}: metadata={len(sub)} text={len(lines)}")
        sub["clean_text"] = lines
        frames.append(sub)
    return pd.concat(frames, ignore_index=True)


def stratified_sample(examples: pd.DataFrame) -> pd.DataFrame:
    rng = random.Random(RANDOM_SEED)
    picked_frames = []
    for group in TARGET_GROUPS_TERMS:
        group_pool = examples[examples["target_group"] == group]
        if group_pool.empty:
            print(f"  WARNING: no candidates for group={group}")
            continue
        legs_present = sorted(group_pool["id_legislatura"].unique())
        per_leg_target = max(1, TARGET_PER_GROUP // len(legs_present))
        leg_frames = []
        for leg in legs_present:
            leg_pool = group_pool[group_pool["id_legislatura"] == leg]
            n = min(per_leg_target, len(leg_pool))
            idx = rng.sample(list(leg_pool.index), n)
            leg_frames.append(leg_pool.loc[idx])
        group_sample = pd.concat(leg_frames, ignore_index=True)
        # Top up to TARGET_PER_GROUP if some strata were thin, from remaining pool.
        shortfall = TARGET_PER_GROUP - len(group_sample)
        if shortfall > 0:
            remaining = group_pool.drop(index=pd.concat(leg_frames).index if leg_frames else [])
            remaining = remaining[~remaining.index.isin(group_sample.index)]
            extra_n = min(shortfall, len(remaining))
            if extra_n > 0:
                extra_idx = rng.sample(list(remaining.index), extra_n)
                group_sample = pd.concat([group_sample, remaining.loc[extra_idx]], ignore_index=True)
        print(f"  {group}: sampled {len(group_sample)} across {len(legs_present)} legislatures")
        picked_frames.append(group_sample)
    return pd.concat(picked_frames, ignore_index=True)


def extract_text_window(tokens: list[str], target_terms: set[str], window: int) -> str:
    positions = [i for i, tok in enumerate(tokens) if tok in target_terms]
    if not positions:
        return ""
    snippets = []
    covered = set()
    for pos in positions:
        start = max(0, pos - window)
        end = min(len(tokens), pos + window + 1)
        span = (start, end)
        if span in covered:
            continue
        covered.add(span)
        snippet = " ".join(tokens[start:pos]) + " [[" + tokens[pos] + "]] " + " ".join(tokens[pos + 1:end])
        snippets.append(snippet)
        if len(snippets) >= 3:  # cap windows per speech for annotator readability
            break
    return "  (...)  ".join(snippets)


def main() -> None:
    print("Loading candidate-adversarial examples...")
    examples_path = RESULTS_DIR / "dissertacao_adversarial_group_discourse_examples.csv"
    examples = pd.read_csv(examples_path)
    print(f"  candidate pool: {len(examples):,} rows")

    print("Drawing stratified sample...")
    sample = stratified_sample(examples)
    print(f"Total sampled: {len(sample)}")

    print("Reconstructing text windows from corpus...")
    speeches = read_speeches()

    rows = []
    for _, ex in sample.iterrows():
        speech_id = int(ex["speech_id"])
        speech_row = speeches.loc[speech_id]
        tokens = str(speech_row["clean_text"]).split()
        target_terms = TARGET_GROUPS_TERMS[ex["target_group"]]
        text_window = extract_text_window(tokens, target_terms, WINDOW)
        rows.append(
            {
                "speech_id": speech_id,
                "legislature": int(ex["id_legislatura"]),
                "party": ex.get("sigla_partido", ""),
                "ideological_block": ex.get("bloco_ideologico", ""),
                "target_group": ex["target_group"],
                "target_label": ex["target_label"],
                "matched_target_terms": ex["target_terms"],
                "matched_adversarial_terms": ex["trigger_terms"],
                "text_window": text_window,
                "label_1": "",
                "label_2": "",
                "notes": "",
            }
        )

    out = pd.DataFrame(rows).sample(frac=1.0, random_state=RANDOM_SEED).reset_index(drop=True)
    out_path = RESULTS_DIR / "dissertacao_adversarial_validation_sample.csv"
    out.to_csv(out_path, index=False, encoding="utf-8-sig")

    print(f"\nSaved {len(out)} rows to: {out_path}")
    print("\nStrata counts (target_group x legislature):")
    print(out.groupby(["target_group", "legislature"]).size().unstack(fill_value=0))
    print(
        "\nAnnotation instructions: for each row, read text_window (target term "
        "marked as [[term]]) and set label_1 to one of: attack, defense, neutral, "
        "ambiguous. label_2 is for a second independent annotator (optional, for "
        "inter-annotator agreement)."
    )


if __name__ == "__main__":
    main()
