# ============================================================
# 00_check_estrutura_corpus.py
# Corpus structure diagnostics — dados_camara/
# Dissertation — Semantic drift in Brazilian parliamentary speeches
# ============================================================

from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# ------------------------------------------------------------
# 1. Configuration
# ------------------------------------------------------------

DATA_DIR    = Path("dados_camara")
FULL_CORPUS = DATA_DIR / "corpus_completo.parquet"
RESULTS_DIR = Path("resultados")
DPI = 300

# Word2Vec requires sufficient tokens per sub-corpus; below this embeddings are unreliable
MIN_TOKENS_W2V = 100_000

# Speeches shorter than this are noise for distributional models
MIN_WORDS_TEXT = 50

# Procrustes alignment degrades when the largest corpus is this many times the smallest
MAX_BALANCE_RATIO = 5.0

# Target legislatures for the dissertation
TARGET_LEGS = {52, 53, 54, 55, 56, 57}

# Ideological block mapping — Brazilian parties (standard political science classification)
IDEOLOGICAL_BLOCK = {
    # Left / Centre-Left
    "PT": "Left",         "PDT": "Left",        "PCdoB": "Left",
    "PC do B": "Left",    "PSOL": "Left",        "PSB": "Left",
    "REDE": "Left",       "PV": "Left",          "SOLIDARIEDADE": "Left",
    "Solidariedade": "Left", "PROS": "Left",
    # Centre
    "MDB": "Centre",      "PMDB": "Centre",      "PSDB": "Centre",
    "PSD": "Centre",      "CIDADANIA": "Centre", "Cidadania": "Centre",
    "PPS": "Centre",      "PP": "Centre",        "PP**": "Centre",
    "PROGRESSISTAS": "Centre", "PPB": "Centre",
    # Right / Centre-Right
    "DEM": "Right",       "DEMOCRATAS": "Right", "PFL": "Right",
    "PL": "Right",        "PL*": "Right",        "PRB": "Right",
    "REPUBLICANOS": "Right", "Republicanos": "Right",
    "PSL": "Right",       "PODEMOS": "Right",    "PODE": "Right",
    "PTB": "Right",       "AVANTE": "Right",     "PATRIOTA": "Right",
    "PSC": "Right",       "NOVO": "Right",       "PR": "Right",
    "PTdoB": "Right",     "PRONA": "Right",      "PTC": "Right",
    "PMN": "Right",       "PHS": "Right",
    "UNIAO": "Right",     "UNIAO BRASIL": "Right",
    # accented variants (stored in data with diacritics)
    "UNIÃO": "Right", "UNIÃO BRASIL": "Right",
}

# Canonical column name candidates (ordered by likelihood from the Câmara API)
TEXT_CANDIDATES  = ["texto", "discurso", "texto_discurso", "transcricao", "conteudo", "body", "speech"]
LEG_CANDIDATES   = ["id_legislatura", "legislatura", "idLegislatura", "leg", "numero_legislatura"]
YEAR_CANDIDATES  = ["ano", "year"]
DATE_CANDIDATES  = ["data", "data_hora_inicio", "dataHoraInicio", "dataHoraFim", "data_discurso", "date"]
PARTY_CANDIDATES = ["sigla_partido", "siglaPartido", "partido", "partido_sigla", "party"]


# ------------------------------------------------------------
# 2. Helpers
# ------------------------------------------------------------

def load_parquet(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    return pd.read_parquet(path)


def detect_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def fmt_int(x, _=None):
    if x >= 1_000_000:
        return f"{x/1e6:.1f}M"
    if x >= 1_000:
        return f"{x/1e3:.0f}K"
    return str(int(x))


# ------------------------------------------------------------
# 3. List folder structure
# ------------------------------------------------------------

def list_structure() -> list[Path]:
    print("\n==============================")
    print("FOLDER STRUCTURE: dados_camara")
    print("==============================\n")

    if not DATA_DIR.exists():
        raise FileNotFoundError("Directory dados_camara not found.")

    print("Subdirectories:")
    for item in sorted(DATA_DIR.iterdir()):
        if item.is_dir():
            print(f"  - {item.name}")

    parquet_files = sorted(DATA_DIR.glob("*.parquet"))
    print("\nParquet files:")
    for f in parquet_files:
        print(f"  - {f.name}")

    return parquet_files


# ------------------------------------------------------------
# 4. Diagnose per-legislature Parquet files
# ------------------------------------------------------------

def diagnose_per_legislature_files():
    print("\n==============================")
    print("PER-LEGISLATURE FILE DIAGNOSTICS")
    print("==============================\n")

    files = sorted(DATA_DIR.glob("discursos_leg*.parquet"))
    if not files:
        print("No discursos_leg*.parquet files found.")
        return

    summary = []
    for f in files:
        print(f"--- {f.name} ---")
        df = load_parquet(f)
        print(f"  Rows   : {len(df):,}")
        print(f"  Columns: {len(df.columns)}")
        print("  Column names:")
        for col in df.columns:
            print(f"    - {col}")
        print()
        leg_id = f.stem.replace("discursos_leg", "")
        summary.append({"file": f.name, "legislature": leg_id, "rows": len(df), "columns": len(df.columns)})

    summary_df = pd.DataFrame(summary)
    print("Overall summary:")
    print(summary_df.to_string(index=False))

    RESULTS_DIR.mkdir(exist_ok=True)
    out = RESULTS_DIR / "summary_legislature_files.csv"
    summary_df.to_csv(out, index=False, encoding="utf-8-sig")
    print(f"\nSummary saved to: {out}")


# ------------------------------------------------------------
# 5. Diagnose corpus_completo.parquet
# ------------------------------------------------------------

def diagnose_full_corpus() -> pd.DataFrame | None:
    print("\n==============================")
    print("FULL CORPUS DIAGNOSTICS: corpus_completo.parquet")
    print("==============================\n")

    if not FULL_CORPUS.exists():
        print("corpus_completo.parquet not found.")
        return None

    df = load_parquet(FULL_CORPUS)

    print(f"Rows   : {len(df):,}")
    print(f"Columns: {len(df.columns)}")

    print("\nColumn names:")
    for col in df.columns:
        print(f"  - {col}")

    print("\nColumn dtypes:")
    print(df.dtypes.to_string())

    print("\nMissing values (top 30):")
    print(df.isna().sum().sort_values(ascending=False).head(30).to_string())

    col_text  = detect_col(df, TEXT_CANDIDATES)
    col_leg   = detect_col(df, LEG_CANDIDATES)
    col_party = detect_col(df, PARTY_CANDIDATES)

    print(f"\nDetected text column      : {col_text  or 'NONE'}")
    print(f"Detected legislature column: {col_leg   or 'NONE'}")
    print(f"Detected party column      : {col_party or 'NONE'}")

    if col_leg:
        print(f"\nSpeeches per legislature ('{col_leg}'):")
        print(df[col_leg].value_counts(dropna=False).sort_index().to_string())

    col_year = detect_col(df, YEAR_CANDIDATES)
    col_date = detect_col(df, DATE_CANDIDATES)

    if col_year:
        print(f"\nSpeeches by year ('{col_year}'):")
        print(df[col_year].value_counts(dropna=False).sort_index().to_string())
    elif col_date:
        anos = pd.to_datetime(df[col_date], errors="coerce").dt.year
        print(f"\nSpeeches by year (inferred from '{col_date}'):")
        print(anos.value_counts(dropna=False).sort_index().to_string())

    if col_party:
        print(f"\nTop 30 parties ('{col_party}'):")
        print(df[col_party].value_counts(dropna=False).head(30).to_string())

    if col_text:
        texts = df[col_text].fillna("").astype(str)
        stats = pd.DataFrame({
            "n_chars": texts.str.len(),
            "n_words_approx": texts.str.split().str.len()
        })
        print(f"\nText statistics ('{col_text}'):")
        print(stats.describe().to_string())

        print("\nFirst non-empty text sample (1,200 chars):")
        sample = texts[texts.str.len() > 0].iloc[0]
        print(sample[:1200])

    RESULTS_DIR.mkdir(exist_ok=True)
    out = RESULTS_DIR / "preview_corpus_completo.csv"
    df.head(1000).to_csv(out, index=False, encoding="utf-8-sig")
    print(f"\nPreview (1,000 rows) saved to: {out}")

    return df


# ------------------------------------------------------------
# 6. Word2Vec readiness validation
# ------------------------------------------------------------

def validate_word2vec_readiness(df: pd.DataFrame):
    print("\n==============================")
    print("WORD2VEC READINESS VALIDATION")
    print("==============================\n")

    col_text = detect_col(df, TEXT_CANDIDATES)
    col_leg  = detect_col(df, LEG_CANDIDATES)

    if not col_text:
        print("[ERROR] No text column detected — cannot validate Word2Vec readiness.")
        return
    if not col_leg:
        print("[ERROR] No legislature column detected — cannot split corpus by legislature.")
        return

    df = df.copy()
    df["_n_words"] = df[col_text].fillna("").astype(str).str.split().str.len().fillna(0).astype(int)

    # --- short-speech proportion
    below = (df["_n_words"] < MIN_WORDS_TEXT).sum()
    pct   = 100.0 * below / len(df)
    tag   = "[WARNING]" if pct > 20 else "[OK]"
    print(f"Speeches with fewer than {MIN_WORDS_TEXT} words: {below:,} ({pct:.1f}%)  {tag}")
    if pct > 20:
        print("  More than 20% of speeches are very short — consider filtering before Word2Vec training.")

    # --- token counts per legislature
    print(f"\nToken counts per legislature (column: '{col_leg}'):")
    tokens_per_leg = df.groupby(col_leg)["_n_words"].sum().sort_index()
    docs_per_leg   = df.groupby(col_leg)["_n_words"].count().sort_index()

    for leg in tokens_per_leg.index:
        tok = int(tokens_per_leg[leg])
        doc = int(docs_per_leg[leg])
        tag = "[OK]" if tok >= MIN_TOKENS_W2V else f"[WARNING] below {MIN_TOKENS_W2V:,}"
        print(f"  Legislature {leg:>3}: {doc:>8,} speeches | {tok:>12,} tokens  {tag}")

    # --- balance check for Procrustes
    if len(tokens_per_leg) >= 2:
        ratio = tokens_per_leg.max() / tokens_per_leg.min()
        tag   = "[OK]" if ratio <= MAX_BALANCE_RATIO else "[WARNING]"
        print(f"\nLargest/smallest token ratio: {ratio:.1f}x  {tag}")
        if ratio > MAX_BALANCE_RATIO:
            print(f"  Ratio exceeds {MAX_BALANCE_RATIO}x — consider downsampling before Procrustes alignment.")

    # --- figure: token counts + speech-length boxplots
    RESULTS_DIR.mkdir(exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # left panel — token totals
    legs_labels = [str(l) for l in tokens_per_leg.index]
    bars = axes[0].bar(legs_labels, tokens_per_leg.values, color="steelblue", edgecolor="white", width=0.6)
    axes[0].axhline(MIN_TOKENS_W2V, color="crimson", linestyle="--", linewidth=1.2,
                    label=f"Min threshold ({MIN_TOKENS_W2V:,})")
    axes[0].set_title("Total Tokens per Legislature", fontsize=13, fontweight="bold")
    axes[0].set_xlabel("Legislature")
    axes[0].set_ylabel("Token count")
    axes[0].yaxis.set_major_formatter(mticker.FuncFormatter(fmt_int))
    axes[0].legend(fontsize=9)
    axes[0].tick_params(axis="x", rotation=0)
    for bar, val in zip(bars, tokens_per_leg.values):
        axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.01,
                     fmt_int(val), ha="center", va="bottom", fontsize=9)

    # right panel — speech length boxplots
    legs_sorted = sorted(df[col_leg].dropna().unique())
    data_boxes  = [df[df[col_leg] == l]["_n_words"].clip(upper=2000).values for l in legs_sorted]
    bp = axes[1].boxplot(data_boxes, tick_labels=[str(l) for l in legs_sorted],
                         patch_artist=True, medianprops={"color": "white", "linewidth": 2})
    for patch in bp["boxes"]:
        patch.set_facecolor("steelblue")
    axes[1].axhline(MIN_WORDS_TEXT, color="crimson", linestyle="--", linewidth=1.2,
                    label=f"Min words/speech ({MIN_WORDS_TEXT})")
    axes[1].set_title("Speech Length Distribution per Legislature\n(clipped at 2,000 words)",
                       fontsize=13, fontweight="bold")
    axes[1].set_xlabel("Legislature")
    axes[1].set_ylabel("Words per speech")
    axes[1].legend(fontsize=9)

    plt.tight_layout()
    out = RESULTS_DIR / "w2v_readiness.png"
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"\nFigure saved to: {out}")


# ------------------------------------------------------------
# 7. Ideological block coverage validation
# ------------------------------------------------------------

def validate_ideological_blocks(df: pd.DataFrame):
    print("\n==============================")
    print("IDEOLOGICAL BLOCK COVERAGE VALIDATION")
    print("==============================\n")

    col_party = detect_col(df, PARTY_CANDIDATES)
    col_leg   = detect_col(df, LEG_CANDIDATES)

    if not col_party:
        print("[ERROR] No party column detected — cannot validate ideological block mapping.")
        return
    if not col_leg:
        print("[ERROR] No legislature column detected.")
        return

    df = df.copy()
    df["_block"] = df[col_party].map(IDEOLOGICAL_BLOCK)

    total   = len(df)
    mapped  = int(df["_block"].notna().sum())
    pct_map = 100.0 * mapped / total

    tag = "[OK]" if pct_map >= 80 else "[WARNING]"
    print(f"Party mapping coverage: {mapped:,} / {total:,} speeches ({pct_map:.1f}%)  {tag}")

    if pct_map < 80:
        print("  Less than 80% mapped — unmapped parties (top 20):")
        unmapped = df[df["_block"].isna()][col_party].value_counts().head(20)
        for party, cnt in unmapped.items():
            print(f"    {party}: {cnt:,} speeches")
    else:
        print("  Coverage is sufficient for Left vs. Right semantic divergence analysis.")

    print("\nSpeeches per ideological block:")
    print(df["_block"].value_counts(dropna=False).to_string())

    print(f"\nSpeeches per legislature x block ('{col_leg}' x '_block'):")
    pivot = df.groupby([col_leg, "_block"]).size().unstack(fill_value=0)
    print(pivot.to_string())

    # --- figure
    RESULTS_DIR.mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 5))
    block_colors = {"Left": "#3366CC", "Centre": "#33AA55", "Right": "#CC3333"}
    cols_ordered = [c for c in ["Left", "Centre", "Right"] if c in pivot.columns]
    colors = [block_colors.get(c, "gray") for c in cols_ordered]
    pivot[cols_ordered].plot(kind="bar", ax=ax, color=colors, edgecolor="white", width=0.7)
    ax.set_title("Speeches per Ideological Block by Legislature", fontsize=13, fontweight="bold")
    ax.set_xlabel("Legislature")
    ax.set_ylabel("Number of speeches")
    ax.legend(title="Block", fontsize=10)
    ax.tick_params(axis="x", rotation=0)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    plt.tight_layout()

    out_fig = RESULTS_DIR / "ideological_blocks.png"
    fig.savefig(out_fig, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"\nFigure saved to: {out_fig}")

    out_csv = RESULTS_DIR / "ideological_block_coverage.csv"
    pivot.to_csv(out_csv, encoding="utf-8-sig")
    print(f"Coverage table saved to: {out_csv}")


# ------------------------------------------------------------
# 8. Dissertation readiness summary
# ------------------------------------------------------------

def print_readiness_summary(df: pd.DataFrame):
    print("\n==============================")
    print("DISSERTATION — CORPUS READINESS SUMMARY")
    print("==============================\n")

    col_text  = detect_col(df, TEXT_CANDIDATES)
    col_leg   = detect_col(df, LEG_CANDIDATES)
    col_party = detect_col(df, PARTY_CANDIDATES)

    checks: dict[str, bool] = {
        "Text column present"        : col_text  is not None,
        "Legislature column present" : col_leg   is not None,
        "Party column present"       : col_party is not None,
    }

    if col_text and col_leg:
        df = df.copy()
        df["_n_words"] = (df[col_text].fillna("").astype(str)
                          .str.split().str.len().fillna(0).astype(int))
        tokens_per_leg = df.groupby(col_leg)["_n_words"].sum()

        checks["All legislatures meet min-token threshold"] = bool(
            (tokens_per_leg >= MIN_TOKENS_W2V).all()
        )
        ratio = (tokens_per_leg.max() / tokens_per_leg.min()
                 if len(tokens_per_leg) >= 2 else 1.0)
        checks["Corpus balance OK for Procrustes (ratio <= 5x)"] = bool(ratio <= MAX_BALANCE_RATIO)

        short_pct = (df["_n_words"] < MIN_WORDS_TEXT).mean()
        checks["Short-speech proportion < 20%"] = bool(short_pct < 0.20)

        try:
            present_legs = {int(v) for v in df[col_leg].dropna().unique()}
        except (ValueError, TypeError):
            present_legs = set()
        checks["Target legislatures 52-57 all present"] = TARGET_LEGS.issubset(present_legs)

    if col_party:
        mapped_pct = df[col_party].map(IDEOLOGICAL_BLOCK).notna().mean()
        checks["Ideological block coverage >= 80%"] = bool(mapped_pct >= 0.80)

    print(f"  {'CHECK':<48} STATUS")
    print("  " + "-" * 58)
    for check, passed in checks.items():
        mark   = "[+]" if passed else "[-]"
        status = "PASS" if passed else "FAIL"
        print(f"  {mark}  {check:<47} {status}")

    n_pass  = sum(checks.values())
    n_total = len(checks)
    print(f"\n  Result: {n_pass}/{n_total} checks passed.")

    if n_pass == n_total:
        print("\n  [READY] Corpus is ready for the dissertation pipeline (Word2Vec + Procrustes).")
    else:
        print("\n  [NOT READY] Resolve the FAIL items above before starting the Word2Vec pipeline.")


# ------------------------------------------------------------
# 9. Entry point
# ------------------------------------------------------------

if __name__ == "__main__":
    list_structure()
    diagnose_per_legislature_files()
    df = diagnose_full_corpus()
    if df is not None:
        validate_word2vec_readiness(df)
        validate_ideological_blocks(df)
        print_readiness_summary(df)
