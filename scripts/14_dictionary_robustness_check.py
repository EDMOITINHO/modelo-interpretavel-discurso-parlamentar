# ============================================================
# 14_dictionary_robustness_check.py
# Dictionary-threshold robustness check (Workstream B)
# Robustness check for the dictionary-based agenda layer in the dissertation.
#
# Recomputes national-policy-theme flags (same THEMES dictionary as
# 08_national_policy_themes.py) under three rules:
#   R1 (baseline): speech flagged if it contains >=1 dictionary term
#   R2 (stricter): flagged if it contains >=2 DISTINCT dictionary terms
#                  OR >=3 total term occurrences
#   R3 (density):  flagged if term-hit density exceeds the median density
#                  observed among R1-positive speeches for that theme
#                  (a data-driven, theme-specific intensity threshold)
#
# R1 must reproduce the existing baseline numbers exactly (sanity check).
# Reports whether the core descriptive claims (overall theme prevalence,
# and specifically "Democracy and institutions") hold directionally
# across all three rules.
# ============================================================

from __future__ import annotations

from collections import Counter
from pathlib import Path

import pandas as pd

TARGET_LEGS = [52, 53, 54, 55, 56, 57]
PREPROC_DIR = Path("data/dissertacao_preprocessado")
RESULTS_DIR = Path("resultados")

# Identical to 08_national_policy_themes.py THEMES dict.
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


def score_speech(text: str) -> tuple[int, dict[str, int], dict[str, int]]:
    """Returns (token_count, distinct_term_hits_per_theme, total_occurrence_hits_per_theme)."""
    tokens = text.split()
    counts = Counter(tokens)
    distinct = {}
    total = {}
    for theme, spec in THEMES.items():
        matched_terms = [t for t in spec["terms"] if counts[t] > 0]
        distinct[theme] = len(matched_terms)
        total[theme] = sum(counts[t] for t in matched_terms)
    return len(tokens), distinct, total


def main() -> None:
    print("Reading corpus...")
    df = read_speeches()
    n_speeches = len(df)
    print(f"Speeches: {n_speeches:,}")

    rows = []
    for idx, row in df.iterrows():
        token_count, distinct, total = score_speech(str(row["clean_text"]))
        for theme in THEMES:
            density = 10_000 * total[theme] / token_count if token_count else 0.0
            rows.append(
                {
                    "speech_id": idx,
                    "theme": theme,
                    "distinct_terms": distinct[theme],
                    "total_hits": total[theme],
                    "density_per_10k": density,
                }
            )
        if idx % 50_000 == 0:
            print(f"  processed {idx:,}/{n_speeches:,}")

    scores = pd.DataFrame(rows)
    print("Scoring complete. Computing rules...")

    # R1: baseline presence rule
    scores["flag_r1_presence"] = (scores["total_hits"] > 0).astype(int)

    # R2: stricter rule
    scores["flag_r2_strict"] = (
        (scores["distinct_terms"] >= 2) | (scores["total_hits"] >= 3)
    ).astype(int)

    # R3: density rule -- theme-specific median density among R1-positive speeches
    medians = (
        scores[scores["flag_r1_presence"] == 1]
        .groupby("theme")["density_per_10k"]
        .median()
        .rename("median_density_r1_positive")
    )
    scores = scores.merge(medians, on="theme", how="left")
    scores["flag_r3_density"] = (
        scores["density_per_10k"] >= scores["median_density_r1_positive"]
    ).astype(int)
    # A speech that never mentions the theme at all (density 0) should never
    # pass a density threshold that is itself derived from positive speeches.
    scores.loc[scores["total_hits"] == 0, "flag_r3_density"] = 0

    summary = (
        scores.groupby("theme")
        .agg(
            speeches_r1=("flag_r1_presence", "sum"),
            speeches_r2=("flag_r2_strict", "sum"),
            speeches_r3=("flag_r3_density", "sum"),
            median_density_threshold=("median_density_r1_positive", "first"),
        )
        .reset_index()
    )
    summary["pct_r1"] = 100 * summary["speeches_r1"] / n_speeches
    summary["pct_r2"] = 100 * summary["speeches_r2"] / n_speeches
    summary["pct_r3"] = 100 * summary["speeches_r3"] / n_speeches
    summary["theme_label"] = summary["theme"].map(lambda t: THEMES[t]["label"])
    summary = summary.sort_values("pct_r1", ascending=False)

    out_path = RESULTS_DIR / "dissertacao_dictionary_robustness_check.csv"
    summary.to_csv(out_path, index=False, encoding="utf-8-sig")

    print("\n=== Dictionary robustness summary (% of all speeches flagged) ===")
    print(
        summary[["theme_label", "pct_r1", "pct_r2", "pct_r3"]]
        .round(1)
        .to_string(index=False)
    )
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
