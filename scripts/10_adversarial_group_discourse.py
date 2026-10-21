"""
Detect candidate adversarial discourse around workers and identity groups.

The output should be read as an audit layer, not as a definitive hate-speech or
attack classifier. A speech is flagged when target-group terms occur near
negative, delegitimizing, threat, or rights-restriction vocabulary.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd


DEFAULT_PREPROC_DIR = Path("data/dissertacao_preprocessado")
DEFAULT_RESULTS_DIR = Path("resultados")
TARGET_LEGS = [52, 53, 54, 55, 56, 57]
WINDOW = 12

TARGET_GROUPS = {
    "workers": {
        "label": "Workers and unions",
        "terms": {
            "trabalhador", "trabalhadores", "trabalhadora", "trabalhadoras",
            "sindicato", "sindicatos", "sindical", "sindicais", "greve",
            "greves", "servidor", "servidores", "funcionario", "funcionarios",
        },
    },
    "lgbt_identity": {
        "label": "LGBT and gender/sexuality identity",
        "terms": {
            "lgbt", "lgbtq", "gay", "gays", "lesbica", "lesbicas",
            "homossexual", "homossexuais", "transexual", "transexuais",
            "transgenero", "transgeneros", "travesti", "travestis",
            "genero", "sexualidade", "orientacao", "homofobia", "transfobia",
        },
    },
    "women": {
        "label": "Women",
        "terms": {"mulher", "mulheres", "feminismo", "feminista", "feministas"},
    },
    "racial_indigenous": {
        "label": "Racial, indigenous, and quilombola groups",
        "terms": {
            "negro", "negros", "negra", "negras", "racismo", "indigena",
            "indigenas", "quilombola", "quilombolas",
        },
    },
}

ADVERSARIAL_LEXICON = {
    "delegitimization": {
        "baderneiro", "baderneiros", "vagabundo", "vagabundos", "parasita",
        "parasitas", "privilegio", "privilegios", "ideologia", "ideologico",
        "ideologica", "doutrinacao", "doutrinar", "imposicao", "militancia",
        "militante", "militantes", "radical", "radicais",
    },
    "threat_criminalization": {
        "ameaca", "ameacas", "crime", "criminoso", "criminosos", "invasao",
        "invasores", "terrorismo", "terrorista", "terroristas", "violencia",
        "violento", "violentos", "desordem", "ilegal", "ilegais",
    },
    "rights_restriction": {
        "proibir", "proibicao", "proibido", "impedir", "barrar", "restringir",
        "restricao", "retirar", "retirada", "cortar", "cortes", "limitar",
        "revogar", "revogacao", "criminalizar", "punir", "punicao",
    },
    "moral_panic": {
        "aberracao", "absurdo", "destruir", "destruicao", "corrupcao",
        "degeneracao", "imoral", "imoralidade", "familia", "criancas",
        "perverter", "perversao", "vergonha", "escandalo",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit candidate adversarial group discourse.")
    parser.add_argument("--preproc-dir", type=Path, default=DEFAULT_PREPROC_DIR)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--window", type=int, default=WINDOW)
    return parser.parse_args()


def read_speeches(preproc_dir: Path) -> pd.DataFrame:
    metadata = pd.read_csv(preproc_dir / "corpus_dissertacao_metadata.csv")
    frames = []
    for leg in TARGET_LEGS:
        lines = (preproc_dir / "by_legislature" / f"leg{leg}.txt").read_text(encoding="utf-8").splitlines()
        sub = metadata[metadata["id_legislatura"] == leg].copy().reset_index(drop=True)
        if len(sub) != len(lines):
            raise ValueError(f"Line count mismatch for leg{leg}: metadata={len(sub)} text={len(lines)}")
        sub["clean_text"] = lines
        frames.append(sub)
    return pd.concat(frames, ignore_index=True)


def default_stats() -> dict[str, int]:
    return {
        "speeches": 0,
        "target_speeches": 0,
        "candidate_adversarial_speeches": 0,
        "target_mentions": 0,
        "window_adversarial_hits": 0,
    }


def nearby_adversarial(tokens: list[str], idx: int, window: int) -> tuple[int, set[str]]:
    start = max(0, idx - window)
    end = min(len(tokens), idx + window + 1)
    context = tokens[start:end]
    categories = set()
    hits = 0
    for category, terms in ADVERSARIAL_LEXICON.items():
        count = sum(1 for token in context if token in terms)
        if count:
            hits += count
            categories.add(category)
    return hits, categories


def analyze(df: pd.DataFrame, window: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    overall = defaultdict(default_stats)
    by_leg = defaultdict(default_stats)
    by_block = defaultdict(default_stats)
    examples = []
    examples_per_stratum: Counter = Counter()
    MAX_EXAMPLES_PER_LEG_GROUP = 1000

    for speech_id, row in df.iterrows():
        tokens = str(row["clean_text"]).split()
        token_counts = Counter(tokens)
        leg = int(row["id_legislatura"])
        block = str(row.get("bloco_ideologico", ""))

        for group, spec in TARGET_GROUPS.items():
            target_positions = [i for i, token in enumerate(tokens) if token in spec["terms"]]
            target_mentions = len(target_positions)
            has_target = target_mentions > 0
            adversarial_hits = 0
            categories = set()
            trigger_terms = Counter()
            target_terms = Counter({term: token_counts[term] for term in spec["terms"] if token_counts[term]})

            if has_target:
                for pos in target_positions:
                    hits, cats = nearby_adversarial(tokens, pos, window)
                    adversarial_hits += hits
                    categories.update(cats)
                    start = max(0, pos - window)
                    end = min(len(tokens), pos + window + 1)
                    for category_terms in ADVERSARIAL_LEXICON.values():
                        for token in tokens[start:end]:
                            if token in category_terms:
                                trigger_terms[token] += 1

            is_candidate = has_target and adversarial_hits > 0

            for bucket, key in (
                (overall, (group, spec["label"])),
                (by_leg, (leg, group, spec["label"])),
                (by_block, (leg, block, group, spec["label"])),
            ):
                stats = bucket[key]
                stats["speeches"] += 1
                stats["target_speeches"] += int(has_target)
                stats["candidate_adversarial_speeches"] += int(is_candidate)
                stats["target_mentions"] += target_mentions
                stats["window_adversarial_hits"] += adversarial_hits

            stratum_key = (leg, group)
            if is_candidate and examples_per_stratum[stratum_key] < MAX_EXAMPLES_PER_LEG_GROUP:
                examples_per_stratum[stratum_key] += 1
                examples.append(
                    {
                        "speech_id": speech_id,
                        "id_legislatura": leg,
                        "governo": row.get("governo", ""),
                        "sigla_partido": row.get("sigla_partido", ""),
                        "bloco_ideologico": block,
                        "target_group": group,
                        "target_label": spec["label"],
                        "target_terms": ";".join(sorted(target_terms)),
                        "adversarial_categories": ";".join(sorted(categories)),
                        "trigger_terms": ";".join(term for term, _ in trigger_terms.most_common(12)),
                        "target_mentions": target_mentions,
                        "window_adversarial_hits": adversarial_hits,
                    }
                )

    return (
        bucket_to_frame(overall, []),
        bucket_to_frame(by_leg, ["id_legislatura"]),
        bucket_to_frame(by_block, ["id_legislatura", "bloco_ideologico"]),
        pd.DataFrame(examples),
    )


def bucket_to_frame(bucket: dict[tuple, dict[str, int]], group_cols: list[str]) -> pd.DataFrame:
    rows = []
    for key, stats in bucket.items():
        speeches = stats["speeches"]
        target_speeches = stats["target_speeches"]
        rows.append(
            {
                **dict(zip(group_cols, key[: len(group_cols)])),
                "target_group": key[len(group_cols)],
                "target_label": key[len(group_cols) + 1],
                "speeches": speeches,
                "target_speeches": target_speeches,
                "candidate_adversarial_speeches": stats["candidate_adversarial_speeches"],
                "target_mentions": stats["target_mentions"],
                "window_adversarial_hits": stats["window_adversarial_hits"],
                "target_speech_share_pct": 100 * target_speeches / speeches if speeches else 0,
                "candidate_share_all_speeches_pct": 100 * stats["candidate_adversarial_speeches"] / speeches
                if speeches
                else 0,
                "candidate_share_target_speeches_pct": 100 * stats["candidate_adversarial_speeches"] / target_speeches
                if target_speeches
                else 0,
            }
        )
    return pd.DataFrame(rows).sort_values(
        group_cols + ["candidate_share_all_speeches_pct"],
        ascending=[True] * len(group_cols) + [False],
    )


def plot_adversarial_longitudinal(by_leg: pd.DataFrame, out_path: Path) -> None:
    """
    Line plot: candidate adversarial context rate (% of all speeches) by
    legislature for each target group. Reveals which legislatures show
    the sharpest increase in adversarial framing.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    group_colors = {
        "workers":          ("#2c6fad", "o", "Workers and unions"),
        "women":            ("#cc3333", "s", "Women"),
        "racial_indigenous":("#4a9a73", "^", "Racial/indigenous/quilombola"),
        "lgbt_identity":    ("#a05cb0", "D", "LGBT/gender/sexuality"),
    }

    legs = sorted(by_leg["id_legislatura"].unique())

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=False)

    # Panel A: candidate adversarial % of all speeches
    for group, (color, marker, label) in group_colors.items():
        sub = by_leg[by_leg["target_group"] == group].sort_values("id_legislatura")
        if sub.empty:
            continue
        axes[0].plot(
            sub["id_legislatura"],
            sub["candidate_share_all_speeches_pct"],
            marker=marker, color=color, linewidth=2, label=label,
        )
    axes[0].set_xticks(legs)
    axes[0].set_xlabel("Legislature")
    axes[0].set_ylabel("Candidate adversarial speeches (% of all speeches)")
    axes[0].set_title("(a) Adversarial context rate by legislature")
    axes[0].grid(axis="y", alpha=0.25)
    axes[0].spines[["top", "right"]].set_visible(False)
    axes[0].legend(frameon=False, fontsize=8)

    # Panel B: candidate adversarial % of target speeches (conditional rate)
    for group, (color, marker, label) in group_colors.items():
        sub = by_leg[by_leg["target_group"] == group].sort_values("id_legislatura")
        if sub.empty:
            continue
        axes[1].plot(
            sub["id_legislatura"],
            sub["candidate_share_target_speeches_pct"],
            marker=marker, color=color, linewidth=2, label=label,
        )
    axes[1].set_xticks(legs)
    axes[1].set_xlabel("Legislature")
    axes[1].set_ylabel("Adversarial speeches (% of speeches mentioning group)")
    axes[1].set_title("(b) Conditional adversarial rate by legislature")
    axes[1].grid(axis="y", alpha=0.25)
    axes[1].spines[["top", "right"]].set_visible(False)
    axes[1].legend(frameon=False, fontsize=8)

    fig.suptitle(
        "Candidate adversarial discourse around social groups by legislature\n"
        "(window = 12 tokens each side; not a definitive hate-speech classifier)",
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def write_summary(overall: pd.DataFrame, by_leg: pd.DataFrame, path: Path) -> None:
    lines = [
        "# Candidate adversarial discourse around social groups",
        "",
        "## Method note",
        "",
        (
            "These are candidate cases, not definitive classifications. A speech "
            "is flagged when a target-group term appears within a local window "
            "near delegitimizing, criminalizing, rights-restriction, or moral-panic vocabulary."
        ),
        "",
        "## Overall",
        "",
    ]
    for _, row in overall.iterrows():
        lines.append(
            f"- {row['target_label']}: target appears in {row['target_speech_share_pct']:.2f}% "
            f"of speeches; candidate adversarial contexts in "
            f"{row['candidate_share_all_speeches_pct']:.2f}% of all speeches "
            f"({row['candidate_share_target_speeches_pct']:.1f}% of target speeches)."
        )

    lines.extend(["", "## By legislature", ""])
    for _, row in by_leg.iterrows():
        lines.append(
            f"- Leg {int(row['id_legislatura'])}, {row['target_label']}: "
            f"{row['candidate_share_all_speeches_pct']:.2f}% of all speeches; "
            f"{row['candidate_share_target_speeches_pct']:.1f}% of target speeches."
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.results_dir.mkdir(parents=True, exist_ok=True)

    speeches = read_speeches(args.preproc_dir)
    overall, by_leg, by_block, examples = analyze(speeches, args.window)

    outputs = {
        "overall": args.results_dir / "dissertacao_adversarial_group_discourse_overall.csv",
        "by_leg": args.results_dir / "dissertacao_adversarial_group_discourse_by_legislature.csv",
        "by_block": args.results_dir / "dissertacao_adversarial_group_discourse_by_block.csv",
        "examples": args.results_dir / "dissertacao_adversarial_group_discourse_examples.csv",
        "summary": args.results_dir / "dissertacao_adversarial_group_discourse_summary.md",
        "longitudinal_figure": Path("figuras/dissertacao") / "dissertacao_adversarial_longitudinal.png",
        "manifest": args.results_dir / "dissertacao_adversarial_group_discourse_manifest.json",
    }

    Path("figuras/dissertacao").mkdir(parents=True, exist_ok=True)

    overall.to_csv(outputs["overall"], index=False, encoding="utf-8")
    by_leg.to_csv(outputs["by_leg"], index=False, encoding="utf-8")
    by_block.to_csv(outputs["by_block"], index=False, encoding="utf-8")
    examples.to_csv(outputs["examples"], index=False, encoding="utf-8")
    write_summary(overall, by_leg, outputs["summary"])
    plot_adversarial_longitudinal(by_leg, outputs["longitudinal_figure"])

    manifest = {
        "purpose": "audit candidate adversarial contexts around workers and identity groups",
        "classification_warning": "candidate co-occurrence measure; requires qualitative validation",
        "window_tokens_each_side": args.window,
        "n_speeches": int(speeches.shape[0]),
        "target_groups": {
            group: {"label": spec["label"], "terms": sorted(spec["terms"])}
            for group, spec in TARGET_GROUPS.items()
        },
        "adversarial_lexicon": {k: sorted(v) for k, v in ADVERSARIAL_LEXICON.items()},
        "outputs": {k: str(v) for k, v in outputs.items()},
    }
    outputs["manifest"].write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Analyzed speeches: {speeches.shape[0]}")
    print(f"Saved overall table to: {outputs['overall']}")
    print(f"Saved by-legislature table to: {outputs['by_leg']}")
    print(f"Saved examples audit to: {outputs['examples']}")
    print(f"Saved longitudinal figure to: {outputs['longitudinal_figure']}")


if __name__ == "__main__":
    main()
