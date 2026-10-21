# ============================================================
# 01_preprocess_corpus.py
# Preprocess corpus for Dissertation
# Semantic drift in Brazilian parliamentary speeches, legs 52-57
# ============================================================

from __future__ import annotations

import argparse
import json
import re
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Iterable

import pandas as pd


# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------

DATA_DIR = Path("dados_camara")
FULL_CORPUS = DATA_DIR / "corpus_completo.parquet"
OUT_DIR = Path("data") / "dissertacao_preprocessado"
RESULTS_DIR = Path("resultados")

TARGET_LEGS = {52, 53, 54, 55, 56, 57}
MIN_WORDS_RAW = 50
MIN_TOKENS_CLEAN = 20
MIN_TOKENS_W2V = 100_000

TEXT_CANDIDATES = [
    "texto",
    "discurso",
    "texto_discurso",
    "transcricao",
    "conteudo",
    "body",
    "speech",
]
LEG_CANDIDATES = [
    "id_legislatura",
    "legislatura",
    "idLegislatura",
    "leg",
    "numero_legislatura",
]
PARTY_CANDIDATES = [
    "sigla_partido",
    "siglaPartido",
    "partido",
    "partido_sigla",
    "party",
]
DATE_CANDIDATES = [
    "data",
    "data_hora_inicio",
    "dataHoraInicio",
    "dataHoraFim",
    "data_discurso",
    "date",
]
HASH_CANDIDATES = ["hash_texto", "hash", "text_hash"]
DUP_CANDIDATES = ["is_duplicata", "duplicata", "is_duplicate"]


# Conservative party block map for the period covered by the corpus.
# The keys are normalized by strip_accents().upper().
IDEOLOGICAL_BLOCK = {
    # Left / centre-left
    "PT": "Left",
    "PDT": "Left",
    "PCDOB": "Left",
    "PC DO B": "Left",
    "PSOL": "Left",
    "PSB": "Left",
    "REDE": "Left",
    "PV": "Left",
    "SOLIDARIEDADE": "Left",
    "PROS": "Left",
    # Centre
    "MDB": "Centre",
    "PMDB": "Centre",
    "PSDB": "Centre",
    "PSD": "Centre",
    "CIDADANIA": "Centre",
    "PPS": "Centre",
    "PP": "Centre",
    "PPB": "Centre",
    "PROGRESSISTAS": "Centre",
    # Right / centre-right
    "DEM": "Right",
    "DEMOCRATAS": "Right",
    "PFL": "Right",
    "PL": "Right",
    "PRB": "Right",
    "REPUBLICANOS": "Right",
    "PSL": "Right",
    "PODEMOS": "Right",
    "PODE": "Right",
    "PTB": "Right",
    "AVANTE": "Right",
    "PATRIOTA": "Right",
    "PSC": "Right",
    "NOVO": "Right",
    "PR": "Right",
    "PTDOB": "Right",
    "PRONA": "Right",
    "PTC": "Right",
    "PMN": "Right",
    "PHS": "Right",
    "UNIAO": "Right",
    "UNIAO BRASIL": "Right",
}


PT_STOPWORDS = {
    "a",
    "ao",
    "aos",
    "aquela",
    "aquelas",
    "aquele",
    "aqueles",
    "aquilo",
    "as",
    "ate",
    "com",
    "como",
    "da",
    "das",
    "de",
    "dela",
    "delas",
    "dele",
    "deles",
    "depois",
    "do",
    "dos",
    "e",
    "ela",
    "elas",
    "ele",
    "eles",
    "em",
    "entre",
    "era",
    "eram",
    "essa",
    "essas",
    "esse",
    "esses",
    "esta",
    "estao",
    "estar",
    "estas",
    "este",
    "estes",
    "eu",
    "foi",
    "foram",
    "ha",
    "isso",
    "isto",
    "ja",
    "lhe",
    "lhes",
    "mais",
    "mas",
    "me",
    "mesmo",
    "meu",
    "meus",
    "minha",
    "minhas",
    "muito",
    "na",
    "nas",
    "nem",
    "no",
    "nos",
    "nossa",
    "nossas",
    "nosso",
    "nossos",
    "num",
    "numa",
    "o",
    "os",
    "ou",
    "para",
    "pela",
    "pelas",
    "pelo",
    "pelos",
    "por",
    "porque",
    "qual",
    "quando",
    "que",
    "quem",
    "sao",
    "se",
    "sem",
    "ser",
    "seu",
    "seus",
    "sua",
    "suas",
    "tambem",
    "te",
    "tem",
    "ter",
    "teu",
    "teus",
    "toda",
    "todas",
    "todo",
    "todos",
    "tu",
    "um",
    "uma",
    "umas",
    "uns",
    "voce",
    "voces",
}


# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------

def detect_col(df: pd.DataFrame, candidates: Iterable[str]) -> str | None:
    for col in candidates:
        if col in df.columns:
            return col
    return None


def strip_accents(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", str(text))
    return "".join(ch for ch in normalized if not unicodedata.combining(ch))


def normalize_party(value: object) -> str:
    text = strip_accents(str(value or "")).upper().strip()
    text = re.sub(r"[^A-Z0-9 ]+", "", text)
    text = re.sub(r"\s+", " ", text)
    return text


def normalize_for_tokens(text: object) -> str:
    text = strip_accents(str(text or "")).lower()
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"\S+@\S+", " ", text)
    text = re.sub(r"[^a-z_ ]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def regex_tokenize(text: object, min_len: int = 3) -> list[str]:
    clean = normalize_for_tokens(text)
    return [
        token
        for token in clean.split()
        if len(token) >= min_len and token not in PT_STOPWORDS
    ]


def get_spacy_pipeline(model: str):
    try:
        import spacy
    except ImportError as exc:
        raise RuntimeError(
            "spaCy is not installed. Run without --use-spacy or install spaCy first."
        ) from exc

    try:
        return spacy.load(model, disable=["ner", "parser"])
    except OSError as exc:
        raise RuntimeError(
            f"spaCy model '{model}' not found. Install it or choose another model."
        ) from exc


def spacy_tokenize_texts(texts: list[str], model: str, batch_size: int) -> list[list[str]]:
    nlp = get_spacy_pipeline(model)
    tokenized: list[list[str]] = []
    for doc in nlp.pipe(texts, batch_size=batch_size):
        tokens = []
        for token in doc:
            if not token.is_alpha or token.is_stop:
                continue
            lemma = normalize_for_tokens(token.lemma_)
            if len(lemma) >= 3 and lemma not in PT_STOPWORDS:
                tokens.append(lemma)
        tokenized.append(tokens)
    return tokenized


def write_lines(path: Path, lines: Iterable[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for line in lines:
            f.write(line)
            f.write("\n")


def add_step(report_rows: list[dict], step: str, before: int, after: int) -> None:
    removed = before - after
    pct_removed = 100.0 * removed / before if before else 0.0
    report_rows.append(
        {
            "step": step,
            "before_rows": before,
            "after_rows": after,
            "removed_rows": removed,
            "removed_pct": round(pct_removed, 3),
        }
    )


# ------------------------------------------------------------
# Main preprocessing
# ------------------------------------------------------------

def preprocess(args: argparse.Namespace) -> None:
    if not args.input.exists():
        raise FileNotFoundError(f"Input file not found: {args.input}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(args.input)

    col_text = detect_col(df, TEXT_CANDIDATES)
    col_leg = detect_col(df, LEG_CANDIDATES)
    col_party = detect_col(df, PARTY_CANDIDATES)
    col_date = detect_col(df, DATE_CANDIDATES)
    col_hash = detect_col(df, HASH_CANDIDATES)
    col_dup = detect_col(df, DUP_CANDIDATES)

    missing = []
    if col_text is None:
        missing.append("text")
    if col_leg is None:
        missing.append("legislature")
    if col_party is None:
        missing.append("party")
    if missing:
        raise ValueError(f"Could not detect required columns: {', '.join(missing)}")

    report_rows: list[dict] = []
    before = len(df)

    df = df.copy()
    df["_legislatura"] = pd.to_numeric(df[col_leg], errors="coerce").astype("Int64")
    df = df[df["_legislatura"].isin(TARGET_LEGS)].copy()
    add_step(report_rows, "keep_legislatures_52_57", before, len(df))

    before = len(df)
    df["_texto_raw"] = df[col_text].fillna("").astype(str)
    df = df[df["_texto_raw"].str.strip().ne("")].copy()
    add_step(report_rows, "remove_empty_text", before, len(df))

    if args.drop_duplicates:
        before = len(df)
        if col_dup:
            df = df[~df[col_dup].fillna(False).astype(bool)].copy()
            add_step(report_rows, f"drop_duplicates_flag_{col_dup}", before, len(df))
        elif col_hash:
            has_hash = df[col_hash].fillna("").astype(str).str.strip().ne("")
            deduped = df[has_hash].drop_duplicates(subset=[col_hash], keep="first")
            no_hash = df[~has_hash]
            df = pd.concat([deduped, no_hash], ignore_index=True)
            add_step(report_rows, f"drop_duplicates_hash_{col_hash}", before, len(df))
        else:
            df = df.drop_duplicates(subset=["_texto_raw"], keep="first").copy()
            add_step(report_rows, "drop_duplicates_text", before, len(df))

    before = len(df)
    df["_n_words_raw"] = df["_texto_raw"].str.split().str.len().fillna(0).astype(int)
    df = df[df["_n_words_raw"] >= args.min_words_raw].copy()
    add_step(report_rows, f"filter_min_words_raw_{args.min_words_raw}", before, len(df))

    df["_partido_norm"] = df[col_party].map(normalize_party)
    df["_bloco_ideologico"] = df["_partido_norm"].map(IDEOLOGICAL_BLOCK).fillna("Unknown")

    if args.use_spacy:
        texts = df["_texto_raw"].tolist()
        tokenized = spacy_tokenize_texts(texts, args.spacy_model, args.spacy_batch_size)
        df["_tokens"] = tokenized
        tokenizer = f"spacy:{args.spacy_model}"
    else:
        df["_tokens"] = df["_texto_raw"].map(regex_tokenize)
        tokenizer = "regex"

    df["_texto_limpo"] = df["_tokens"].map(" ".join)
    df["_n_tokens"] = df["_tokens"].map(len).astype(int)

    before = len(df)
    df = df[df["_n_tokens"] >= args.min_tokens_clean].copy()
    add_step(report_rows, f"filter_min_tokens_clean_{args.min_tokens_clean}", before, len(df))

    keep_cols = [
        col for col in [
            col_date,
            col_party,
            col_hash,
            col_dup,
            "governo" if "governo" in df.columns else None,
            "id_deputado" if "id_deputado" in df.columns else None,
            "nome_deputado" if "nome_deputado" in df.columns else None,
            "uf" if "uf" in df.columns else None,
        ]
        if col is not None
    ]

    out_df = df[
        keep_cols
        + [
            "_legislatura",
            "_partido_norm",
            "_bloco_ideologico",
            "_n_words_raw",
            "_n_tokens",
            "_texto_raw",
            "_texto_limpo",
        ]
    ].rename(
        columns={
            "_legislatura": "id_legislatura",
            "_partido_norm": "partido_norm",
            "_bloco_ideologico": "bloco_ideologico",
            "_n_words_raw": "n_words_raw",
            "_n_tokens": "n_tokens",
            "_texto_raw": "texto_raw",
            "_texto_limpo": "texto_limpo",
        }
    )

    corpus_path = OUT_DIR / "corpus_dissertacao_filtrado.parquet"
    out_df.to_parquet(corpus_path, index=False)

    metadata_path = OUT_DIR / "corpus_dissertacao_metadata.csv"
    out_df.drop(columns=["texto_raw", "texto_limpo"]).to_csv(
        metadata_path, index=False, encoding="utf-8-sig"
    )

    by_leg_dir = OUT_DIR / "by_legislature"
    by_block_dir = OUT_DIR / "by_legislature_block"

    for leg, sub in out_df.groupby("id_legislatura", sort=True):
        write_lines(
            by_leg_dir / f"leg{int(leg)}.txt",
            sub["texto_limpo"].tolist(),
        )

    for (leg, block), sub in out_df.groupby(["id_legislatura", "bloco_ideologico"], sort=True):
        if block == "Unknown":
            continue
        write_lines(
            by_block_dir / f"leg{int(leg)}_{block.lower()}.txt",
            sub["texto_limpo"].tolist(),
        )

    readiness_rows = []
    for leg, sub in out_df.groupby("id_legislatura", sort=True):
        token_counter = Counter()
        for text in sub["texto_limpo"]:
            token_counter.update(text.split())
        total_tokens = int(sub["n_tokens"].sum())
        readiness_rows.append(
            {
                "id_legislatura": int(leg),
                "speeches": int(len(sub)),
                "tokens": total_tokens,
                "types": int(len(token_counter)),
                "mean_tokens_per_speech": round(float(sub["n_tokens"].mean()), 2),
                "w2v_min_tokens": MIN_TOKENS_W2V,
                "w2v_ready": total_tokens >= MIN_TOKENS_W2V,
            }
        )

    readiness = pd.DataFrame(readiness_rows)
    readiness_path = RESULTS_DIR / "dissertacao_preprocess_readiness.csv"
    readiness.to_csv(readiness_path, index=False, encoding="utf-8-sig")

    block_coverage = (
        out_df.groupby(["id_legislatura", "bloco_ideologico"])
        .size()
        .reset_index(name="speeches")
        .sort_values(["id_legislatura", "bloco_ideologico"])
    )
    block_path = RESULTS_DIR / "dissertacao_block_coverage_preprocess.csv"
    block_coverage.to_csv(block_path, index=False, encoding="utf-8-sig")

    report = pd.DataFrame(report_rows)
    report_path = RESULTS_DIR / "dissertacao_preprocess_report.csv"
    report.to_csv(report_path, index=False, encoding="utf-8-sig")

    manifest = {
        "input": str(args.input),
        "output_dir": str(OUT_DIR),
        "tokenizer": tokenizer,
        "target_legislatures": sorted(TARGET_LEGS),
        "min_words_raw": args.min_words_raw,
        "min_tokens_clean": args.min_tokens_clean,
        "drop_duplicates": args.drop_duplicates,
        "detected_columns": {
            "text": col_text,
            "legislature": col_leg,
            "party": col_party,
            "date": col_date,
            "hash": col_hash,
            "duplicate_flag": col_dup,
        },
        "outputs": {
            "corpus": str(corpus_path),
            "metadata": str(metadata_path),
            "by_legislature": str(by_leg_dir),
            "by_legislature_block": str(by_block_dir),
            "readiness": str(readiness_path),
            "block_coverage": str(block_path),
            "report": str(report_path),
        },
    }
    manifest_path = RESULTS_DIR / "dissertacao_preprocess_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("\nPreprocessing finished.")
    print(f"Filtered corpus: {corpus_path}")
    print(f"Legislature text files: {by_leg_dir}")
    print(f"Block text files: {by_block_dir}")
    print(f"Report: {report_path}")
    print(f"Readiness: {readiness_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Preprocess corpus for Dissertation semantic drift analysis."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=FULL_CORPUS,
        help="Input Parquet file. Default: dados_camara/corpus_completo.parquet",
    )
    parser.add_argument(
        "--min-words-raw",
        type=int,
        default=MIN_WORDS_RAW,
        help=f"Minimum raw word count per speech. Default: {MIN_WORDS_RAW}",
    )
    parser.add_argument(
        "--min-tokens-clean",
        type=int,
        default=MIN_TOKENS_CLEAN,
        help=f"Minimum clean token count per speech. Default: {MIN_TOKENS_CLEAN}",
    )
    parser.add_argument(
        "--keep-duplicates",
        action="store_true",
        help="Keep duplicate speeches. Default behavior drops duplicates when possible.",
    )
    parser.add_argument(
        "--use-spacy",
        action="store_true",
        help="Use spaCy lemmatization instead of regex tokenization.",
    )
    parser.add_argument(
        "--spacy-model",
        default="pt_core_news_lg",
        help="spaCy Portuguese model used with --use-spacy. Default: pt_core_news_lg",
    )
    parser.add_argument(
        "--spacy-batch-size",
        type=int,
        default=256,
        help="spaCy pipe batch size. Default: 256",
    )
    args = parser.parse_args()
    args.drop_duplicates = not args.keep_duplicates
    return args


if __name__ == "__main__":
    preprocess(parse_args())
