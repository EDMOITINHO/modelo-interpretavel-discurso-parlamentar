"""
Classify parliamentary speeches into broad national agendas and sectorized agendas.

National agendas refer to broad state/public-policy themes such as education,
health, economy, security, infrastructure, rights, and institutions. Sectorized
agendas refer to mentions of specific social groups, professional categories,
economic sectors, territories, or identity groups.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


DEFAULT_PREPROC_DIR = Path("data/dissertacao_preprocessado")
DEFAULT_RESULTS_DIR = Path("resultados")
DEFAULT_FIGURES_DIR = Path("figuras/dissertacao")
TARGET_LEGS = [52, 53, 54, 55, 56, 57]

NATIONAL_AGENDAS = {
    "education": {"educacao", "escola", "ensino", "professor", "professores", "universidade", "fundeb"},
    "health": {"saude", "sus", "hospital", "medico", "medicos", "vacina", "pandemia", "covid"},
    "economy": {"economia", "emprego", "trabalho", "renda", "salario", "inflacao", "juros", "credito"},
    "security": {"seguranca", "violencia", "crime", "policia", "armas", "trafico", "homicidio"},
    "infrastructure": {"infraestrutura", "transporte", "rodovia", "estrada", "energia", "saneamento", "internet"},
    "environment": {"ambiente", "ambiental", "amazonia", "floresta", "desmatamento", "clima", "agua"},
    "rights_social_policy": {"direitos", "cidadania", "pobreza", "fome", "desigualdade", "assistencia", "previdencia"},
    "institutions": {"democracia", "constituicao", "congresso", "camara", "senado", "governo", "voto", "judiciario"},
}

SECTORIZED_AGENDAS = {
    "agriculture_rural": {
        "agricultura", "agricola", "agropecuaria", "agronegocio", "produtor",
        "produtores", "rural", "campo", "pecuaria", "safra", "assentamento",
    },
    "business_industry": {
        "empresa", "empresas", "industria", "industrial", "comercio", "comerciante",
        "empreendedor", "empreendedores", "microempresa", "empresarios",
    },
    "workers_unions": {
        "trabalhador", "trabalhadores", "sindicato", "sindicatos", "servidor",
        "servidores", "categoria", "categorias", "funcionario", "funcionarios",
    },
    "public_servants_professions": {
        "professor", "professores", "medico", "medicos", "enfermeiro",
        "enfermeiros", "policial", "policiais", "militar", "militares",
    },
    "women_family_children": {
        "mulher", "mulheres", "familia", "familias", "crianca", "criancas",
        "adolescente", "adolescentes", "juventude", "idosos", "idoso",
    },
    "race_indigenous_minorities": {
        "negros", "negro", "racismo", "indigena", "indigenas", "quilombolas",
        "quilombola", "lgbt", "deficiencia", "pessoa", "pessoas",
    },
    "religious_moral": {
        "religiao", "religioso", "religiosos", "igreja", "igrejas", "evangelicos",
        "catolicos", "familia", "vida", "aborto",
    },
    "territorial_regional": {
        "municipio", "municipios", "prefeito", "prefeitos", "nordeste", "norte",
        "amazonia", "semiarido", "regional", "regiao", "comunidade", "comunidades",
    },
    "security_forces": {
        "policia", "policial", "policiais", "militar", "militares", "bombeiros",
        "guarda", "forcas", "armadas",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Classify national and sectorized agendas.")
    parser.add_argument("--preproc-dir", type=Path, default=DEFAULT_PREPROC_DIR)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--figures-dir", type=Path, default=DEFAULT_FIGURES_DIR)
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


def hits_by_dictionary(counts: Counter[str], dictionary: dict[str, set[str]]) -> dict[str, int]:
    return {name: sum(counts[t] for t in terms) for name, terms in dictionary.items()}


def make_summary(bucket: dict[tuple, dict[str, int]], group_cols: list[str]) -> pd.DataFrame:
    rows = []
    for key, stats in bucket.items():
        speeches = stats["speeches"]
        rows.append(
            {
                **dict(zip(group_cols, key)),
                "speeches": speeches,
                "national_speeches": stats["national_speeches"],
                "sectorized_speeches": stats["sectorized_speeches"],
                "mixed_speeches": stats["mixed_speeches"],
                "national_only_speeches": stats["national_only_speeches"],
                "sectorized_only_speeches": stats["sectorized_only_speeches"],
                "no_agenda_speeches": stats["no_agenda_speeches"],
                "national_share_pct": 100 * stats["national_speeches"] / speeches,
                "sectorized_share_pct": 100 * stats["sectorized_speeches"] / speeches,
                "mixed_share_pct": 100 * stats["mixed_speeches"] / speeches,
                "sectorized_among_national_pct": 100 * stats["mixed_speeches"] / stats["national_speeches"]
                if stats["national_speeches"]
                else 0,
                "sector_hits_per_10k_tokens": 10000 * stats["sector_token_hits"] / stats["tokens"]
                if stats["tokens"]
                else 0,
            }
        )
    return pd.DataFrame(rows).sort_values(group_cols)


def update_bucket(bucket: dict[tuple, dict[str, int]], key: tuple, tokens: int, has_national: bool, has_sector: bool, sector_hits: int) -> None:
    stats = bucket[key]
    stats["speeches"] += 1
    stats["tokens"] += tokens
    stats["national_speeches"] += int(has_national)
    stats["sectorized_speeches"] += int(has_sector)
    stats["mixed_speeches"] += int(has_national and has_sector)
    stats["national_only_speeches"] += int(has_national and not has_sector)
    stats["sectorized_only_speeches"] += int(has_sector and not has_national)
    stats["no_agenda_speeches"] += int(not has_national and not has_sector)
    stats["sector_token_hits"] += sector_hits


def default_stats() -> dict[str, int]:
    return {
        "speeches": 0,
        "tokens": 0,
        "national_speeches": 0,
        "sectorized_speeches": 0,
        "mixed_speeches": 0,
        "national_only_speeches": 0,
        "sectorized_only_speeches": 0,
        "no_agenda_speeches": 0,
        "sector_token_hits": 0,
    }


def analyze(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    overall = defaultdict(default_stats)
    by_leg = defaultdict(default_stats)
    by_block = defaultdict(default_stats)
    sector_type = defaultdict(lambda: {"speeches": 0, "speeches_with_type": 0, "token_hits": 0, "tokens": 0})

    for _, row in df.iterrows():
        tokens = str(row["clean_text"]).split()
        counts = Counter(tokens)
        national_hits = hits_by_dictionary(counts, NATIONAL_AGENDAS)
        sector_hits = hits_by_dictionary(counts, SECTORIZED_AGENDAS)
        has_national = any(v > 0 for v in national_hits.values())
        has_sector = any(v > 0 for v in sector_hits.values())
        total_sector_hits = sum(sector_hits.values())
        n_tokens = len(tokens)
        leg = int(row["id_legislatura"])
        block = str(row.get("bloco_ideologico", ""))

        update_bucket(overall, tuple(), n_tokens, has_national, has_sector, total_sector_hits)
        update_bucket(by_leg, (leg,), n_tokens, has_national, has_sector, total_sector_hits)
        update_bucket(by_block, (leg, block), n_tokens, has_national, has_sector, total_sector_hits)

        for sector, hits in sector_hits.items():
            key = (leg, sector)
            sector_type[key]["speeches"] += 1
            sector_type[key]["speeches_with_type"] += int(hits > 0)
            sector_type[key]["token_hits"] += hits
            sector_type[key]["tokens"] += n_tokens

    sector_rows = []
    for (leg, sector), stats in sector_type.items():
        speeches = stats["speeches"]
        sector_rows.append(
            {
                "id_legislatura": leg,
                "sector_agenda": sector,
                "speeches": speeches,
                "speeches_with_type": stats["speeches_with_type"],
                "speech_share_pct": 100 * stats["speeches_with_type"] / speeches,
                "token_hits_per_10k": 10000 * stats["token_hits"] / stats["tokens"] if stats["tokens"] else 0,
            }
        )

    return (
        make_summary(overall, []),
        make_summary(by_leg, ["id_legislatura"]),
        make_summary(by_block, ["id_legislatura", "bloco_ideologico"]),
        pd.DataFrame(sector_rows).sort_values(["id_legislatura", "speech_share_pct"], ascending=[True, False]),
    )


def plot_summary(by_leg: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.plot(by_leg["id_legislatura"], by_leg["national_share_pct"], marker="o", label="National agendas")
    ax.plot(by_leg["id_legislatura"], by_leg["sectorized_share_pct"], marker="o", label="Sectorized agendas")
    ax.plot(by_leg["id_legislatura"], by_leg["mixed_share_pct"], marker="o", label="Mixed")
    ax.set_xlabel("Legislature")
    ax.set_ylabel("% of speeches")
    ax.set_title("National and sectorized agendas in parliamentary speeches")
    ax.grid(axis="y", alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def write_summary(overall: pd.DataFrame, by_leg: pd.DataFrame, sector_types: pd.DataFrame, path: Path) -> None:
    row = overall.iloc[0]
    top_sector = sector_types.groupby("sector_agenda", as_index=False).agg(
        speech_share_pct=("speech_share_pct", "mean"),
        token_hits_per_10k=("token_hits_per_10k", "mean"),
    ).sort_values("speech_share_pct", ascending=False).head(8)

    lines = [
        "# National and sectorized agendas",
        "",
        "## Overall",
        "",
        f"- National agenda speeches: {row['national_share_pct']:.1f}%.",
        f"- Sectorized agenda speeches: {row['sectorized_share_pct']:.1f}%.",
        f"- Mixed national and sectorized speeches: {row['mixed_share_pct']:.1f}%.",
        f"- Sectorized among national agenda speeches: {row['sectorized_among_national_pct']:.1f}%.",
        "",
        "## By legislature",
        "",
    ]
    for _, leg in by_leg.iterrows():
        lines.append(
            f"- Leg {int(leg['id_legislatura'])}: national {leg['national_share_pct']:.1f}%, "
            f"sectorized {leg['sectorized_share_pct']:.1f}%, mixed {leg['mixed_share_pct']:.1f}%."
        )

    lines.extend(["", "## Most frequent sectorized agendas", ""])
    for _, sector in top_sector.iterrows():
        lines.append(
            f"- {sector['sector_agenda']}: mean {sector['speech_share_pct']:.1f}% of speeches by legislature; "
            f"{sector['token_hits_per_10k']:.1f} hits per 10k tokens."
        )

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.results_dir.mkdir(parents=True, exist_ok=True)
    args.figures_dir.mkdir(parents=True, exist_ok=True)

    speeches = read_speeches(args.preproc_dir)
    overall, by_leg, by_block, sector_types = analyze(speeches)

    outputs = {
        "overall": args.results_dir / "dissertacao_agenda_national_vs_sectorized_overall.csv",
        "by_leg": args.results_dir / "dissertacao_agenda_national_vs_sectorized_by_legislature.csv",
        "by_block": args.results_dir / "dissertacao_agenda_national_vs_sectorized_by_block.csv",
        "sector_types": args.results_dir / "dissertacao_sectorized_agenda_types_by_legislature.csv",
        "summary": args.results_dir / "dissertacao_agenda_national_vs_sectorized_summary.md",
        "manifest": args.results_dir / "dissertacao_agenda_national_vs_sectorized_manifest.json",
        "figure": args.figures_dir / "dissertacao_agenda_national_vs_sectorized.png",
    }

    overall.to_csv(outputs["overall"], index=False, encoding="utf-8")
    by_leg.to_csv(outputs["by_leg"], index=False, encoding="utf-8")
    by_block.to_csv(outputs["by_block"], index=False, encoding="utf-8")
    sector_types.to_csv(outputs["sector_types"], index=False, encoding="utf-8")
    write_summary(overall, by_leg, sector_types, outputs["summary"])
    plot_summary(by_leg, outputs["figure"])

    manifest = {
        "purpose": "distinguish broad national agendas from sectorized group agendas",
        "n_speeches": int(speeches.shape[0]),
        "national_agendas": {k: sorted(v) for k, v in NATIONAL_AGENDAS.items()},
        "sectorized_agendas": {k: sorted(v) for k, v in SECTORIZED_AGENDAS.items()},
        "outputs": {k: str(v) for k, v in outputs.items()},
    }
    outputs["manifest"].write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Analyzed speeches: {speeches.shape[0]}")
    print(f"Saved overall table to: {outputs['overall']}")
    print(f"Saved by-legislature table to: {outputs['by_leg']}")
    print(f"Saved summary to: {outputs['summary']}")


if __name__ == "__main__":
    main()
