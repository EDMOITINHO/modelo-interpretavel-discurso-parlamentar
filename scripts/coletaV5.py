"""
coletaV5.py — Coleta paralela de discursos — Câmara Federal API v2
===================================================================
Uso:
    python coletaV5.py                        # todas as legislaturas
    python coletaV5.py --leg 55 56 57         # legislaturas específicas
    python coletaV5.py --retomar              # retoma coleta interrompida
    python coletaV5.py --so-consolidar        # só regenera corpus_completo.parquet
    python coletaV5.py --workers 8            # ajusta workers (padrão 5)

Melhorias sobre v4
==================
[BUG] consolidar(): deduplicação ignorava hash_texto="" — colapsava todos os
      discursos sem transcrição em UM único registro, jogando fora ~40K linhas
      da leg 55. Corrigido: só deduplica onde hash_texto != "".
[BUG] id_discurso usava dataHoraInicio+id_dep — colide quando o mesmo deputado
      profere 2 discursos no mesmo segundo. Corrigido: acrescenta índice sequencial.
[BUG] 500 em alguns deputies era retentado 10× (até 50 min por deputado).
      Corrigido: 500 num recurso específico é permanente — nenhum retry.
[PERF] Estratégia coarse-to-fine: tenta o período completo primeiro (1 req/dep),
       recua para anual só se vier 504/timeout, e para mensal só se o ano
       também falhar. Reduz chamadas de 48/dep para ~1–3/dep na média.
[PERF] ThreadPoolExecutor com N_WORKERS workers: processa N deputados em paralelo.
       Com 5 workers e ~2 req/dep: ~3–10 min por legislatura vs. 24h no v4.
[PERF] Backoff reduzido: max 60 s (era 300 s) e 5 tentativas (era 10).
       Evita travar 50 min num único 504; falhas persistentes são registradas
       e a coleta segue para os próximos deputados.
[PERF] Delay por thread reduzido para 0.4 s (era 1.2 s). Com 5 workers a taxa
       efetiva é ≈5 req/s — conservadora para uma API pública sem SLA publicado.
"""

import argparse
import hashlib
import json
import logging
import random
import threading
import time
from calendar import monthrange
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import pandas as pd
import requests
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)
from tqdm import tqdm

# ── constants ────────────────────────────────────────────────────────────────
BASE_URL  = "https://dadosabertos.camara.leg.br/api/v2"
ITENS_PAG = 100          # max allowed by the API
DELAY_REQ = 0.4          # base delay per thread between requests
JITTER    = 0.3          # random jitter added to each delay
TIMEOUT   = 90           # request timeout in seconds
DIR_SAIDA = Path("dados_camara")
DIR_LOG   = Path("logs")
CHECKPOINT_EVERY = 100   # checkpoint every N completed deputies

# Parallel workers — conservative for a public API without documented rate limits.
# Increase to 8–10 if you observe no 429 errors after a test run with 5.
N_WORKERS = 5

LEGISLATURAS = {
    52: {"dataInicio": "2003-02-01", "dataFim": "2007-01-31", "governo": "Lula I"},
    53: {"dataInicio": "2007-02-01", "dataFim": "2011-01-31", "governo": "Lula II"},
    54: {"dataInicio": "2011-02-01", "dataFim": "2015-01-31", "governo": "Dilma I"},
    55: {"dataInicio": "2015-02-01", "dataFim": "2019-01-31", "governo": "Dilma II / Temer"},
    56: {"dataInicio": "2019-02-01", "dataFim": "2023-01-31", "governo": "Bolsonaro"},
    57: {"dataInicio": "2023-02-01", "dataFim": "2027-01-31", "governo": "Lula III"},
}

GOVERNOS = [
    ("FHC_II",         "1999-01-01", "2002-12-31"),
    ("Lula_I",         "2003-01-01", "2006-12-31"),
    ("Lula_II",        "2007-01-01", "2010-12-31"),
    ("Dilma_I",        "2011-01-01", "2014-12-31"),
    ("Dilma_II",       "2015-01-01", "2016-05-11"),
    ("Dilma_afastada", "2016-05-12", "2016-08-31"),
    ("Temer",          "2016-09-01", "2018-12-31"),
    ("Bolsonaro",      "2019-01-01", "2022-12-31"),
    ("Lula_III",       "2023-01-01", "2027-12-31"),
]


# ── logging ───────────────────────────────────────────────────────────────────
DIR_LOG.mkdir(exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  [%(threadName)s]  %(message)s",
    handlers=[
        logging.FileHandler(
            DIR_LOG / f"coleta_{datetime.now():%Y%m%d_%H%M}.log",
            encoding="utf-8",
        ),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger(__name__)


# ── per-thread HTTP session ───────────────────────────────────────────────────
_tls = threading.local()


def _session() -> requests.Session:
    """Returns a per-thread Session, creating it on first call in each thread."""
    if not hasattr(_tls, "session"):
        s = requests.Session()
        s.headers.update({
            "Accept": "application/json",
            "User-Agent": "pesquisa-nlp-ppgmsb-uneb/1.0",
        })
        _tls.session = s
    return _tls.session


# ── custom exceptions ─────────────────────────────────────────────────────────
class RateLimitError(Exception):
    """HTTP 429 — retriable after Retry-After header."""


class GatewayError(Exception):
    """HTTP 502/503/504 — infrastructure overload, retriable."""


class DataError(Exception):
    """HTTP 500 on a specific resource — data unavailable, NOT retriable.
    Retrying a 500 on /deputados/{id}/discursos wastes 10+ minutes (v4 bug)."""


# ── HTTP layer ────────────────────────────────────────────────────────────────
@retry(
    retry=retry_if_exception_type((
        requests.ConnectionError,
        requests.Timeout,
        RateLimitError,
        GatewayError,
    )),
    # [PERF v5] max 60 s (was 300 s) and 5 attempts (was 10).
    # 300 s max meant a single 504 could stall a thread for 50 minutes.
    wait=wait_exponential(multiplier=2, min=5, max=60),
    stop=stop_after_attempt(5),
)
def _get_json(url: str, params: dict = None) -> dict:
    """GET with exponential retry. DataError is never retried."""
    resp = _session().get(url, params=params, timeout=TIMEOUT)

    if resp.status_code == 429:
        wait_s = int(resp.headers.get("Retry-After", 60))
        log.warning(f"Rate limit (429) — waiting {wait_s}s")
        time.sleep(wait_s)
        raise RateLimitError(f"HTTP 429 at {url}")

    if resp.status_code == 500:
        # [BUG FIX v5] 500 on a specific resource is permanent — skip, do not retry.
        # v3/early-v4 would retry this 10 times (up to 50 min per deputy).
        raise DataError(f"HTTP 500 at {url}")

    if resp.status_code in (502, 503, 504):
        log.debug(f"HTTP {resp.status_code} at {url} — will retry")
        raise GatewayError(f"HTTP {resp.status_code} at {url}")

    resp.raise_for_status()
    return resp.json()


def _jitter_sleep(base: float = DELAY_REQ):
    time.sleep(base + random.uniform(0, JITTER))


# ── pagination ────────────────────────────────────────────────────────────────
def _paginate(url: str, params: dict) -> tuple[list[dict], bool]:
    """
    Collect all pages for url+params.
    Returns (records, needs_fallback).
    needs_fallback=True when a GatewayError survives all retries,
    signalling the caller to retry with a smaller time window.
    DataError (500) returns partial data with needs_fallback=False —
    smaller windows won't fix a missing resource.
    """
    results = []
    p = dict(params)
    page = 1
    while True:
        p["pagina"] = page
        try:
            data = _get_json(url, p)
        except DataError:
            return results, False   # permanent — no fallback
        except GatewayError:
            return results, True    # transient on this window — caller should narrow
        except Exception as e:
            log.warning(f"Unhandled error at {url} p={page}: {e}")
            return results, True

        batch = data.get("dados", [])
        if not batch:
            break
        results.extend(batch)

        links = {lk["rel"]: lk["href"] for lk in data.get("links", [])}
        if "next" not in links:
            break
        page += 1
        _jitter_sleep()

    return results, False


# ── date range generators ─────────────────────────────────────────────────────
def _years(start: str, end: str):
    ini = datetime.strptime(start, "%Y-%m-%d")
    fim = datetime.strptime(end,   "%Y-%m-%d")
    for y in range(ini.year, fim.year + 1):
        yield (
            start if y == ini.year else f"{y}-01-01",
            end   if y == fim.year else f"{y}-12-31",
        )


def _months(start: str, end: str):
    ini = datetime.strptime(start, "%Y-%m-%d")
    fim = datetime.strptime(end,   "%Y-%m-%d")
    cur = ini.replace(day=1)
    while cur <= fim:
        last_day = cur.replace(day=monthrange(cur.year, cur.month)[1])
        yield (cur.strftime("%Y-%m-%d"), min(last_day, fim).strftime("%Y-%m-%d"))
        cur = (
            cur.replace(year=cur.year + 1, month=1, day=1)
            if cur.month == 12
            else cur.replace(month=cur.month + 1, day=1)
        )


# ── coarse-to-fine speech collection ─────────────────────────────────────────
def _fetch_speeches(id_dep: int, start: str, end: str) -> list[dict]:
    """
    Coarse-to-fine: try the full period (1 request) → yearly → monthly.
    Most deputies need only 1–2 requests total.
    In v4 every deputy always made 48 requests (one per month).
    """
    url = f"{BASE_URL}/deputados/{id_dep}/discursos"
    base_params = {"ordenarPor": "dataHoraInicio", "ordem": "ASC", "itens": ITENS_PAG}

    # Level 1: full period
    results, fallback = _paginate(url, {**base_params, "dataInicio": start, "dataFim": end})
    if not fallback:
        return results

    log.info(f"dep {id_dep}: full period timed out — falling back to yearly")
    all_results = []

    # Level 2: year by year
    for y_start, y_end in _years(start, end):
        batch, fallback = _paginate(url, {**base_params, "dataInicio": y_start, "dataFim": y_end})
        if not fallback:
            all_results.extend(batch)
            _jitter_sleep(0.3)
            continue

        log.info(f"dep {id_dep}: year {y_start[:4]} timed out — falling back to monthly")

        # Level 3: month by month (only for this year)
        for m_start, m_end in _months(y_start, y_end):
            m_batch, _ = _paginate(url, {**base_params, "dataInicio": m_start, "dataFim": m_end})
            all_results.extend(m_batch)
            _jitter_sleep(0.2)

    return all_results


# ── record builder ────────────────────────────────────────────────────────────
def _make_hash(text: str) -> str:
    if not text:
        return ""
    return hashlib.md5(text.strip().lower().encode("utf-8")).hexdigest()


def _assign_governo(date_str: str) -> str:
    if not date_str:
        return "unknown"
    d = date_str[:10]
    for name, ini, fim in GOVERNOS:
        if ini <= d <= fim:
            return name
    return "out_of_range"


def _build_records(speeches: list[dict], dep: dict, id_leg: int) -> list[dict]:
    id_dep = dep["id"]
    nome   = dep.get("nome", "")
    rows   = []
    for idx, s in enumerate(speeches):
        text = s.get("transcricao") or s.get("sumario") or ""
        ts   = s.get("dataHoraInicio") or ""
        rows.append({
            # [BUG FIX v5] add idx — avoids collision when same deputy speaks twice
            # in the same second (previously: dataHoraInicio + id_dep only)
            "id_discurso":      f"{id_dep}_{ts}_{idx}",
            "id_parlamentar":   id_dep,
            "nome_parlamentar": nome,
            "sigla_partido":    dep.get("siglaPartido", ""),
            "uri_partido":      dep.get("uriPartido", ""),
            "uf":               dep.get("siglaUf", ""),
            "id_legislatura":   id_leg,
            "data_hora_inicio": ts,
            "data_hora_fim":    s.get("dataHoraFim", ""),
            "data":             ts[:10],
            "governo":          _assign_governo(ts),
            "tipo_discurso":    s.get("tipoDiscurso", ""),
            "fase_evento":      (s.get("faseEvento") or {}).get("titulo", "")
                                if isinstance(s.get("faseEvento"), dict) else "",
            "tipo_evento":      s.get("tipoEvento", ""),
            "orgao_evento":     (s.get("orgaos") or [{}])[0].get("nome", ""),
            "sumario":          s.get("sumario", ""),
            "keywords":         s.get("keywords", ""),
            "transcricao":      text,
            "uri_evento":       s.get("uriEvento", ""),
            "hash_texto":       _make_hash(text),
            "tem_transcricao":  bool(s.get("transcricao")),
            "n_chars":          len(text),
        })
    return rows


# ── per-deputy worker (runs in thread pool) ───────────────────────────────────
def _worker(dep: dict, id_leg: int, cfg: dict) -> tuple[list[dict], dict | None]:
    """Collect all speeches for one deputy. Returns (records, error_or_None)."""
    id_dep = dep["id"]
    try:
        speeches = _fetch_speeches(id_dep, cfg["dataInicio"], cfg["dataFim"])
        return _build_records(speeches, dep, id_leg), None
    except Exception as exc:
        log.warning(f"dep {id_dep} ({dep.get('nome','?')}): {exc}")
        return [], {"id_dep": id_dep, "nome": dep.get("nome", "?"), "erro": str(exc)}


# ── deputy list ───────────────────────────────────────────────────────────────
def _get_deputies(id_leg: int) -> list[dict]:
    log.info(f"Fetching deputy list — legislature {id_leg}")
    deputies, page = [], 1
    while True:
        data = _get_json(f"{BASE_URL}/deputados", params={
            "idLegislatura": id_leg, "itens": ITENS_PAG,
            "pagina": page, "ordem": "ASC", "ordenarPor": "nome",
        })
        batch = data.get("dados", [])
        if not batch:
            break
        deputies.extend(batch)
        links = {lk["rel"]: lk["href"] for lk in data.get("links", [])}
        if "next" not in links:
            break
        page += 1
        _jitter_sleep(0.5)
    log.info(f"  {len(deputies)} deputies found")
    return deputies


# ── checkpoint ────────────────────────────────────────────────────────────────
def _save_checkpoint(
    records: list, path: Path, processed: set, prog_path: Path
) -> pd.DataFrame:
    if not records:
        return pd.DataFrame()
    df = pd.DataFrame(records)
    df.to_parquet(path, index=False, compression="snappy")
    prog_path.write_text(json.dumps({
        "ids": list(processed),
        "total": len(records),
        "updated": datetime.now().isoformat(),
    }))
    return df


# ── legislature collection ────────────────────────────────────────────────────
def coletar_legislatura(id_leg: int, retomar: bool = False) -> pd.DataFrame:
    cfg     = LEGISLATURAS[id_leg]
    dir_leg = DIR_SAIDA / f"leg_{id_leg}"
    dir_leg.mkdir(parents=True, exist_ok=True)

    arq_saida = DIR_SAIDA / f"discursos_leg{id_leg}.parquet"
    arq_prog  = dir_leg / "progresso.json"

    processed: set[int] = set()
    if retomar and arq_prog.exists():
        processed = set(json.loads(arq_prog.read_text())["ids"])
        log.info(f"Resuming: {len(processed)} deputies already done")

    deputies = _get_deputies(id_leg)
    pending  = [d for d in deputies if d["id"] not in processed]
    log.info(f"Legislature {id_leg}: {len(pending)} to process, {len(processed)} already done")

    all_records: list[dict] = []
    if retomar and arq_saida.exists():
        all_records = pd.read_parquet(arq_saida).to_dict("records")
        log.info(f"  Loaded {len(all_records):,} existing records from checkpoint")

    errors: list[dict] = []
    lock   = threading.Lock()

    with tqdm(total=len(pending), desc=f"Leg {id_leg}", unit="dep", ncols=90) as pbar:
        with ThreadPoolExecutor(
            max_workers=N_WORKERS,
            thread_name_prefix=f"L{id_leg}",
        ) as ex:
            futures = {ex.submit(_worker, dep, id_leg, cfg): dep for dep in pending}
            n_done  = 0

            for fut in as_completed(futures):
                dep     = futures[fut]
                records, err = fut.result()

                with lock:
                    all_records.extend(records)
                    processed.add(dep["id"])
                    if err:
                        errors.append(err)
                    n_done += 1
                    pbar.update(1)
                    pbar.set_postfix_str(dep.get("nome", "")[:28])

                    if n_done % CHECKPOINT_EVERY == 0:
                        snapshot_r = list(all_records)
                        snapshot_p = set(processed)

                # checkpoint outside lock to avoid blocking other threads
                if n_done % CHECKPOINT_EVERY == 0:
                    _save_checkpoint(snapshot_r, arq_saida, snapshot_p, arq_prog)
                    log.info(f"  Checkpoint: {n_done}/{len(pending)} deputies, "
                             f"{len(snapshot_r):,} records")

    df = _save_checkpoint(all_records, arq_saida, processed, arq_prog)

    if errors:
        pd.DataFrame(errors).to_csv(dir_leg / "erros_coleta.csv", index=False)
        log.warning(f"  {len(errors)} deputies with errors — see erros_coleta.csv")

    log.info(f"Legislature {id_leg}: {len(df):,} speeches from {len(processed)} deputies")
    return df


# ── audit ─────────────────────────────────────────────────────────────────────
def auditar(df: pd.DataFrame, id_leg: int) -> pd.DataFrame:
    dir_leg = DIR_SAIDA / f"leg_{id_leg}"
    df      = df.copy()
    df["is_duplicata"] = (
        df.duplicated("hash_texto", keep="first") & (df["hash_texto"] != "")
    )
    stats = {
        "total_speeches":     len(df),
        "with_transcript":    int(df["tem_transcricao"].sum()),
        "without_transcript": int((~df["tem_transcricao"]).sum()),
        "hash_duplicates":    int(df["is_duplicata"].sum()),
        "unique_deputies":    df["id_parlamentar"].nunique(),
        "unique_parties":     df["sigla_partido"].nunique(),
        "by_governo":         df["governo"].value_counts().to_dict(),
        "top10_parties":      df["sigla_partido"].value_counts().head(10).to_dict(),
        "chars_median":       float(df["n_chars"].median()),
        "chars_p5":           float(df["n_chars"].quantile(0.05)),
        "chars_p95":          float(df["n_chars"].quantile(0.95)),
    }
    (dir_leg / "auditoria.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2)
    )
    log.info(f"\n{'='*55}")
    log.info(f"AUDIT — Legislature {id_leg}")
    log.info(f"  Total speeches:      {stats['total_speeches']:>8,}")
    log.info(f"  With transcript:     {stats['with_transcript']:>8,}")
    log.info(f"  Without transcript:  {stats['without_transcript']:>8,}")
    log.info(f"  Hash duplicates:     {stats['hash_duplicates']:>8,}")
    log.info(f"  Unique deputies:     {stats['unique_deputies']:>8,}")
    log.info(f"  Median chars:        {stats['chars_median']:>8,.0f}")
    log.info(f"{'='*55}\n")
    return df


# ── consolidation ─────────────────────────────────────────────────────────────
def consolidar():
    """
    Merge all per-legislature Parquets into corpus_completo.parquet.

    [BUG FIX v5] v4 called drop_duplicates(subset=["hash_texto"]) on the
    entire corpus. Since hash_texto="" for all speeches without transcription,
    this collapsed every empty-text speech across all legislatures into ONE row,
    discarding ~40K speeches from leg 55 (whose data happened to come after
    legs 52-54 in the concat order).

    Fix: only deduplicate rows that have actual content (hash_texto != "").
    Empty-text rows are kept as-is, deduplicated only within the same hash
    (i.e., truly identical content if that content is non-empty).
    """
    files = sorted(DIR_SAIDA.glob("discursos_leg*.parquet"))
    if not files:
        log.warning("No legislature Parquets found — nothing to consolidate.")
        return

    dfs = []
    for f in files:
        df = pd.read_parquet(f)
        dfs.append(df)
        log.info(f"  {f.name}: {len(df):,} records")

    corpus = pd.concat(dfs, ignore_index=True)
    before = len(corpus)

    # Deduplicate only where there is actual text content
    has_text   = corpus["hash_texto"] != ""
    deduped    = corpus[has_text].drop_duplicates(subset=["hash_texto"], keep="first")
    no_text    = corpus[~has_text]
    corpus     = pd.concat([deduped, no_text], ignore_index=True)

    removed = before - len(corpus)
    log.info(f"Content deduplication: {before:,} -> {len(corpus):,} ({removed:,} removed)")

    corpus["is_duplicata"] = False

    out = DIR_SAIDA / "corpus_completo.parquet"
    corpus.to_parquet(out, index=False, compression="snappy")
    log.info(f"Corpus saved: {out}  ({len(corpus):,} speeches)")

    log.info("\nSpeeches per legislature:")
    for leg, n in corpus["id_legislatura"].value_counts().sort_index().items():
        log.info(f"  Legislature {leg}: {n:>8,}")

    log.info("\nSpeeches per government:")
    for gov, n in corpus["governo"].value_counts().items():
        log.info(f"  {gov:<22} {n:>8,}")


# ── CLI ───────────────────────────────────────────────────────────────────────
def main():
    global N_WORKERS
    parser = argparse.ArgumentParser(
        description="Collect Camara Federal speeches — API v2 (v5)"
    )
    parser.add_argument(
        "--leg", type=int, nargs="+",
        default=list(LEGISLATURAS.keys()),
        help="Legislatures to collect (default: all)",
    )
    parser.add_argument(
        "--retomar", action="store_true",
        help="Resume interrupted collection (skip already-processed deputies)",
    )
    parser.add_argument(
        "--so-consolidar", action="store_true",
        help="Only regenerate corpus_completo.parquet from existing per-leg files",
    )
    parser.add_argument(
        "--workers", type=int, default=N_WORKERS,
        help=f"Parallel worker threads (default {N_WORKERS})",
    )
    args = parser.parse_args()

    N_WORKERS = args.workers
    log.info(f"Using {N_WORKERS} parallel workers")

    DIR_SAIDA.mkdir(exist_ok=True)

    if args.so_consolidar:
        consolidar()
        return

    for id_leg in args.leg:
        if id_leg not in LEGISLATURAS:
            log.warning(f"Legislature {id_leg} not configured — skipping")
            continue

        log.info(f"\n{'#'*60}")
        log.info(f"  LEGISLATURE {id_leg} — {LEGISLATURAS[id_leg]['governo']}")
        log.info(f"{'#'*60}")

        df = coletar_legislatura(id_leg, retomar=args.retomar)
        if not df.empty:
            df = auditar(df, id_leg)
            arq = DIR_SAIDA / f"discursos_leg{id_leg}.parquet"
            arq.unlink(missing_ok=True)
            df.to_parquet(arq, index=False, compression="snappy")

    consolidar()
    log.info("\nCollection complete.")


if __name__ == "__main__":
    main()
