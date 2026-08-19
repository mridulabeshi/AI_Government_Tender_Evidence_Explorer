"""
Stage 1 — AI Tender Discovery.

Two passes over the raw procurement corpus:

  Pass A (candidate filter): fast, cheap keyword/regex scan over
  `title` + `tender_description` using DuckDB, run directly against the
  Parquet file so it scales to millions of rows without loading
  everything into memory.

  Pass B (verification layer): a keyword hit is NOT treated as a
  confirmed AI tender on its own ("Smart City Platform" may have no AI;
  "Intelligent Surveillance System" may hint at computer vision without
  saying "AI"). This pass re-scores each candidate using a slightly
  wider signal set and assigns a confidence tier + AI-type bucket.
  In production Pass B would also read the tender document itself; here
  it works off title+description, and Stage 3 (document_extractor +
  governance_extractor) does the deeper, document-level confirmation.
"""

import os
import re

import duckdb
import pandas as pd

import config
import data_loader


def _build_keyword_regex(keywords):
    escaped = [re.escape(k) for k in keywords]
    return re.compile(r"\b(" + "|".join(escaped) + r")\b", re.IGNORECASE)


KEYWORD_REGEX = _build_keyword_regex(config.AI_KEYWORDS)


def run_candidate_filter(parquet_path: str = config.RAW_PARQUET,
                          source: str = "sample",
                          limit_files: int = None) -> pd.DataFrame:
    """
    Pass A. Uses DuckDB SQL (regexp_matches) so this scales to the full
    multi-million row dataset without ever pulling the whole corpus into
    memory — only rows matching the WHERE clause are materialized into
    the returned DataFrame.

    source="sample"  -> query the local offline synthetic Parquet file
                         (from `python data_loader.py --sample N`)
    source="real"    -> query the real rumourscape/tenders dataset
                         directly over HTTPS from Hugging Face (7 remote
                         Parquet files, ~4.92M rows). Needs network
                         access to huggingface.co. `limit_files` can cap
                         how many of the 7 remote files are scanned, for
                         a quicker partial test run.
    """
    con = duckdb.connect()
    keyword_pattern = "|".join(re.escape(k) for k in config.AI_KEYWORDS)

    if source == "real":
        con.execute("INSTALL httpfs; LOAD httpfs;")
        files = data_loader.HF_PARQUET_FILES
        if limit_files:
            files = files[:limit_files]
        from_clause = data_loader._REAL_DATASET_SELECT.format(
            files=data_loader._files_sql_list(files)
        )
        query = f"""
            WITH tenders AS ({from_clause})
            SELECT * FROM tenders
            WHERE regexp_matches(lower(title), '{keyword_pattern}')
               OR regexp_matches(lower(tender_description), '{keyword_pattern}')
        """
    else:
        query = f"""
            SELECT *
            FROM read_parquet('{parquet_path}')
            WHERE regexp_matches(lower(title), '{keyword_pattern}')
               OR regexp_matches(lower(tender_description), '{keyword_pattern}')
        """

    df = con.execute(query).df()
    return df


def _extract_matched_keywords(text: str) -> list:
    if not isinstance(text, str):
        return []
    return sorted(set(m.lower() for m in KEYWORD_REGEX.findall(text)))


def _keywords_to_ai_types(keywords: list) -> list:
    types = {config.KEYWORD_TO_AI_TYPE.get(k, "Other AI") for k in keywords}
    return sorted(types)


def run_verification_layer(candidates: pd.DataFrame) -> pd.DataFrame:
    """
    Pass B. Adds:
      - matched_keywords: which vocabulary terms actually fired
      - ai_types: coarse capability buckets (Computer Vision, GenAI/LLM, ...)
      - confidence: High / Medium / Low, based on where + how many hits
      - verified: bool — whether this clears the bar to be treated as a
        genuine AI procurement candidate for Stage 2 deep-dive selection

    Confidence heuristic:
      High   -> keyword hit in the title itself (title is a strong signal
                and is usually deliberately descriptive of the system)
      Medium -> 2+ distinct keyword families hit only in the description
      Low    -> a single, possibly generic keyword hit only in the
                description (e.g. lone "biometric" mention)
    """
    rows = []
    for _, row in candidates.iterrows():
        title_kw = _extract_matched_keywords(row.get("title", ""))
        desc_kw = _extract_matched_keywords(row.get("tender_description", ""))
        all_kw = sorted(set(title_kw) | set(desc_kw))

        if title_kw:
            confidence = "High"
        elif len(set(desc_kw)) >= 2:
            confidence = "Medium"
        else:
            confidence = "Low"

        ai_types = _keywords_to_ai_types(all_kw)

        rows.append({
            **row.to_dict(),
            "matched_keywords": ", ".join(all_kw),
            "ai_types": ", ".join(ai_types),
            "confidence": confidence,
            "verified": confidence in ("High", "Medium"),
        })

    return pd.DataFrame(rows)


def discover(parquet_path: str = config.RAW_PARQUET,
             source: str = "sample",
             limit_files: int = None) -> pd.DataFrame:
    """Run Pass A then Pass B and persist the result."""
    candidates = run_candidate_filter(parquet_path, source=source, limit_files=limit_files)
    os.makedirs(config.DATA_DIR, exist_ok=True)
    candidates.to_parquet(config.CANDIDATES_PARQUET, index=False)

    verified = run_verification_layer(candidates)
    verified.to_parquet(config.CONFIRMED_PARQUET, index=False)
    return verified


def capability_map(verified: pd.DataFrame) -> pd.Series:
    """Counts per AI type, for the Stage-1 dashboard view."""
    exploded = verified.assign(
        ai_types=verified["ai_types"].str.split(", ")
    ).explode("ai_types")
    exploded = exploded[exploded["verified"]]
    return exploded["ai_types"].value_counts()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", choices=["sample", "real"], default="sample",
                         help="'sample' = local offline synthetic data, "
                              "'real' = query rumourscape/tenders directly from Hugging Face")
    parser.add_argument("--limit-files", type=int, default=None,
                         help="With --source real: only scan the first N of the "
                              "7 remote Parquet files (faster partial test run)")
    args = parser.parse_args()

    verified_df = discover(source=args.source, limit_files=args.limit_files)
    print(f"Candidates: {len(verified_df)} | Verified: {int(verified_df['verified'].sum())}")
    print(capability_map(verified_df))
