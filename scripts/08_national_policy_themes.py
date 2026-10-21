"""
Measure how parliamentary speeches connect to central national-policy themes.

This analysis is intentionally substantive rather than methodological. It maps
cleaned speeches to broad themes of an organized state: education, health,
economy, security, environment, infrastructure, rights, social protection, and
institutional governance.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


DEFAULT_PREPROC_DIR = Path("data/dissertacao_preprocessado")
DEFAULT_RESULTS_DIR = Path("resultados")
DEFAULT_FIGURES_DIR = Path("figuras/dissertacao")
TARGET_LEGS = [52, 53, 54, 55, 56, 57]


THEMES = {
    "education": {
        "label": "Education",
        "terms": {
            "educacao", "escola", "escolas", "ensino", "professor", "professores",
            "aluno", "alunos", "universidade", "universidades", "creche",
            "alfabetizacao", "fundeb", "enem", "estudante", "estudantes",
            "pesquisa", "ciencia", "cientifico", "tecnologia",
        },
    },
    "health": {
        "label": "Health",
        "terms": {
            "saude", "sus", "hospital", "hospitais", "medico", "medicos",
            "enfermeiro", "enfermeiros", "enfermeiras", "vacina", "vacinas",
            "pandemia", "covid", "doenca", "doencas", "tratamento",
            "medicamento", "medicamentos", "pacientes", "atendimento",
        },
    },
    "economy_work_income": {
        "label": "Economy, work, and income",
        "terms": {
            "economia", "economico", "economica", "emprego", "empregos",
            "trabalho", "trabalhador", "trabalhadores", "renda", "salario",
            "salarios", "inflacao", "juros", "credito", "industria",
            "empresas", "empresa", "mercado", "crescimento", "desenvolvimento",
            "investimento", "investimentos", "tributaria", "imposto", "impostos",
        },
    },
    "security_violence": {
        "label": "Public security and violence",
        "terms": {
            "seguranca", "violencia", "crime", "crimes", "criminalidade",
            "policia", "policial", "policiais", "prisao", "presidio",
            "armas", "arma", "trafico", "homicidio", "homicidios",
            "feminicidio", "militar", "militares",
        },
    },
    "environment_climate": {
        "label": "Environment and climate",
        "terms": {
            "meio", "ambiente", "ambiental", "amazonia", "floresta",
            "florestas", "desmatamento", "clima", "climatica", "climatico",
            "sustentavel", "sustentabilidade", "biodiversidade", "indigena",
            "indigenas", "garimpo", "queimadas", "agua", "saneamento",
        },
    },
    "infrastructure_transport_housing": {
        "label": "Infrastructure, transport, and housing",
        "terms": {
            "infraestrutura", "transporte", "transportes", "rodovia",
            "rodovias", "estrada", "estradas", "ferrovia", "ferrovias",
            "porto", "portos", "aeroporto", "aeroportos", "energia",
            "eletrica", "habitacao", "moradia", "saneamento", "internet",
            "telecomunicacoes", "obra", "obras",
        },
    },
    "social_protection_inequality": {
        "label": "Social protection and inequality",
        "terms": {
            "pobreza", "fome", "desigualdade", "desigualdades",
            "assistencia", "social", "bolsa", "familia", "beneficio",
            "beneficios", "previdencia", "aposentadoria", "aposentados",
            "vulnerabilidade", "moradia", "direitos", "mulheres", "criancas",
            "juventude", "idosos", "deficiencia",
        },
    },
    "federalism_municipalities": {
        "label": "Federalism and municipalities",
        "terms": {
            "municipio", "municipios", "prefeito", "prefeitos", "prefeitura",
            "prefeituras", "estado", "estados", "federacao", "federal",
            "governador", "governadores", "regional", "regionais", "regiao",
            "regioes", "nordeste", "norte", "sul", "sudeste", "centro",
        },
    },
    "democracy_institutions": {
        "label": "Democracy and institutions",
        "terms": {
            "democracia", "democratico", "democratica", "constituicao",
            "constitucional", "congresso", "camara", "senado", "parlamento",
            "governo", "presidente", "presidencia", "republica", "eleicao",
            "eleicoes", "voto", "impeachment", "stf", "supremo",
            "judiciario", "justica",
        },
    },
    "rights_citizenship": {
        "label": "Rights and citizenship",
        "terms": {
            "direito", "direitos", "cidadania", "cidadao", "cidadaos",
            "igualdade", "liberdade", "justica", "mulher", "mulheres",
            "negros", "racismo", "discriminacao", "lgbt", "indigena",
            "indigenas", "quilombolas", "criancas", "adolescentes",
        },
    },
    "agriculture_food": {
        "label": "Agriculture and food",
        "terms": {
            "agricultura", "agricola", "agropecuaria", "agronegocio",
            "produtor", "produtores", "rural", "campo", "alimentos",
            "alimento", "comida", "fome", "pecuaria", "safra",
            "assentamento", "reforma", "agraria",
        },
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Quantify national-policy themes in parliamentary speeches."
    )
    parser.add_argument("--preproc-dir", type=Path, default=DEFAULT_PREPROC_DIR)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--figures-dir", type=Path, default=DEFAULT_FIGURES_DIR)
    return parser.parse_args()


def read_speeches(preproc_dir: Path) -> pd.DataFrame:
    metadata_path = preproc_dir / "corpus_dissertacao_metadata.csv"
    metadata = pd.read_csv(metadata_path)
    frames = []
    for leg in TARGET_LEGS:
        path = preproc_dir / "by_legislature" / f"leg{leg}.txt"
        lines = path.read_text(encoding="utf-8").splitlines()
        sub = metadata[metadata["id_legislatura"] == leg].copy().reset_index(drop=True)
        if len(sub) != len(lines):
            raise ValueError(
                f"Line count mismatch for leg{leg}: metadata={len(sub)} speeches={len(lines)}"
            )
        sub["clean_text"] = lines
        frames.append(sub)
    return pd.concat(frames, ignore_index=True)


def score_speech(text: str) -> tuple[int, dict[str, int], dict[str, int]]:
    tokens = text.split()
    counts = Counter(tokens)
    hits = {}
    token_hits = {}
    for theme, spec in THEMES.items():
        n = sum(counts[term] for term in spec["terms"])
        token_hits[theme] = n
        hits[theme] = int(n > 0)
    return len(tokens), hits, token_hits


def build_long_scores(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for idx, row in df.iterrows():
        token_count, hits, token_hits = score_speech(str(row["clean_text"]))
        for theme, spec in THEMES.items():
            rows.append(
                {
                    "speech_id": idx,
                    "id_legislatura": row["id_legislatura"],
                    "governo": row.get("governo", ""),
                    "sigla_partido": row.get("sigla_partido", ""),
                    "bloco_ideologico": row.get("bloco_ideologico", ""),
                    "theme": theme,
                    "theme_label": spec["label"],
                    "speech_tokens": token_count,
                    "theme_present": hits[theme],
                    "theme_token_hits": token_hits[theme],
                    "theme_token_share": token_hits[theme] / token_count if token_count else 0.0,
                }
            )
    return pd.DataFrame(rows)


def summarize(scores: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    return (
        scores.groupby(group_cols + ["theme", "theme_label"], as_index=False)
        .agg(
            speeches=("speech_id", "nunique"),
            speeches_with_theme=("theme_present", "sum"),
            total_tokens=("speech_tokens", "sum"),
            theme_token_hits=("theme_token_hits", "sum"),
            mean_speech_theme_share=("theme_token_share", "mean"),
        )
        .assign(
            speech_share_pct=lambda d: 100 * d["speeches_with_theme"] / d["speeches"],
            token_share_per_10k=lambda d: 10000 * d["theme_token_hits"] / d["total_tokens"],
        )
        .sort_values(group_cols + ["speech_share_pct"], ascending=[True] * len(group_cols) + [False])
    )


def plot_legislature_heatmap(summary_leg: pd.DataFrame, out_path: Path) -> None:
    pivot = summary_leg.pivot(
        index="theme_label",
        columns="id_legislatura",
        values="speech_share_pct",
    ).fillna(0)
    pivot = pivot.loc[pivot.mean(axis=1).sort_values(ascending=False).index]

    fig, ax = plt.subplots(figsize=(9, 6.2))
    im = ax.imshow(pivot.values, aspect="auto", cmap="YlGnBu")
    ax.set_xticks(range(len(pivot.columns)), labels=pivot.columns)
    ax.set_yticks(range(len(pivot.index)), labels=pivot.index)
    ax.set_xlabel("Legislature")
    ax.set_title("Share of speeches linked to central national-policy themes")

    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            ax.text(j, i, f"{pivot.iloc[i, j]:.1f}", ha="center", va="center", fontsize=7)

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("% of speeches")
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def build_summary(summary_leg: pd.DataFrame, summary_all: pd.DataFrame) -> str:
    top_all = summary_all.sort_values("speech_share_pct", ascending=False).head(8)
    lines = [
        "# National-policy themes in parliamentary discourse",
        "",
        "## Interpretation",
        "",
        (
            "This analysis measures how strongly parliamentary speeches are "
            "connected to broad themes of state organization and public policy. "
            "A speech is counted as linked to a theme when it contains at least "
            "one term from the theme dictionary; token density is also reported "
            "to distinguish broad presence from intensity."
        ),
        "",
        "## Most recurrent themes in the full corpus",
        "",
    ]
    for _, row in top_all.iterrows():
        lines.append(
            f"- {row['theme_label']}: {row['speech_share_pct']:.1f}% of speeches; "
            f"{row['token_share_per_10k']:.1f} theme hits per 10k tokens."
        )

    lines.extend(["", "## Legislature-level shifts", ""])
    for theme_label, group in summary_leg.groupby("theme_label"):
        ordered = group.sort_values("id_legislatura")
        first = ordered.iloc[0]
        last = ordered.iloc[-1]
        delta = last["speech_share_pct"] - first["speech_share_pct"]
        lines.append(
            f"- {theme_label}: {first['speech_share_pct']:.1f}% in leg {int(first['id_legislatura'])} "
            f"to {last['speech_share_pct']:.1f}% in leg {int(last['id_legislatura'])} "
            f"(delta {delta:+.1f} pp)."
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    args.results_dir.mkdir(parents=True, exist_ok=True)
    args.figures_dir.mkdir(parents=True, exist_ok=True)

    speeches = read_speeches(args.preproc_dir)
    scores = build_long_scores(speeches)

    summary_leg = summarize(scores, ["id_legislatura"])
    summary_block = summarize(scores, ["id_legislatura", "bloco_ideologico"])
    summary_all = summarize(scores, [])

    outputs = {
        "speech_theme_scores": args.results_dir / "dissertacao_national_theme_speech_scores.csv",
        "theme_by_legislature": args.results_dir / "dissertacao_national_theme_by_legislature.csv",
        "theme_by_block": args.results_dir / "dissertacao_national_theme_by_block.csv",
        "theme_overall": args.results_dir / "dissertacao_national_theme_overall.csv",
        "summary": args.results_dir / "dissertacao_national_theme_summary.md",
        "manifest": args.results_dir / "dissertacao_national_theme_manifest.json",
        "figure": args.figures_dir / "dissertacao_national_theme_heatmap.png",
    }

    scores.drop(columns=["speech_tokens"]).to_csv(
        outputs["speech_theme_scores"], index=False, encoding="utf-8"
    )
    summary_leg.to_csv(outputs["theme_by_legislature"], index=False, encoding="utf-8")
    summary_block.to_csv(outputs["theme_by_block"], index=False, encoding="utf-8")
    summary_all.to_csv(outputs["theme_overall"], index=False, encoding="utf-8")
    outputs["summary"].write_text(build_summary(summary_leg, summary_all), encoding="utf-8")
    plot_legislature_heatmap(summary_leg, outputs["figure"])

    manifest = {
        "purpose": "measure links between parliamentary speeches and central national-policy themes",
        "unit": "speech",
        "method": "dictionary presence and token density over cleaned corpus",
        "n_speeches": int(speeches.shape[0]),
        "themes": {
            theme: {"label": spec["label"], "terms": sorted(spec["terms"])}
            for theme, spec in THEMES.items()
        },
        "outputs": {key: str(path) for key, path in outputs.items()},
    }
    outputs["manifest"].write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"Analyzed speeches: {speeches.shape[0]}")
    print(f"Saved theme-by-legislature table to: {outputs['theme_by_legislature']}")
    print(f"Saved theme-by-block table to: {outputs['theme_by_block']}")
    print(f"Saved summary to: {outputs['summary']}")
    print(f"Saved heatmap to: {outputs['figure']}")


if __name__ == "__main__":
    main()
