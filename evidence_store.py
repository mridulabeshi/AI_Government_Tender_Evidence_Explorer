"""
Evidence Database — persists per-tender governance findings so the
Streamlit dashboard can query them without re-running LLM extraction on
every page load. Backed by SQLite for MVP simplicity (swap for Postgres
later without changing the calling code much).
"""

import json
import os
import sqlite3

import config


SCHEMA = """
CREATE TABLE IF NOT EXISTS tender_profiles (
    tender_id TEXT PRIMARY KEY,
    organisation TEXT,
    state TEXT,
    title TEXT,
    ai_types TEXT,
    impact TEXT,
    impact_matched_phrase TEXT,
    data_categories_json TEXT,
    human_oversight_json TEXT,
    bias_fairness_json TEXT,
    failure_fallback_json TEXT,
    source_document_url TEXT,
    num_pages INTEGER
);
"""


def get_connection():
    os.makedirs(config.DATA_DIR, exist_ok=True)
    conn = sqlite3.connect(config.EVIDENCE_DB)
    conn.execute(SCHEMA)
    return conn


def save_tender_profile(conn, tender_row: dict, governance: dict, impact: dict):
    """
    tender_row: dict with tender_id, organisation, state, title, ai_types,
                tender_document_url
    governance: output of governance_extractor.extract_governance_evidence
    impact:     output of impact_classifier.classify_impact
    """
    conn.execute(
        """
        INSERT INTO tender_profiles (
            tender_id, organisation, state, title, ai_types,
            impact, impact_matched_phrase,
            data_categories_json, human_oversight_json,
            bias_fairness_json, failure_fallback_json,
            source_document_url, num_pages
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(tender_id) DO UPDATE SET
            organisation=excluded.organisation,
            state=excluded.state,
            title=excluded.title,
            ai_types=excluded.ai_types,
            impact=excluded.impact,
            impact_matched_phrase=excluded.impact_matched_phrase,
            data_categories_json=excluded.data_categories_json,
            human_oversight_json=excluded.human_oversight_json,
            bias_fairness_json=excluded.bias_fairness_json,
            failure_fallback_json=excluded.failure_fallback_json,
            source_document_url=excluded.source_document_url,
            num_pages=excluded.num_pages
        """,
        (
            tender_row["tender_id"], tender_row.get("organisation"),
            tender_row.get("state"), tender_row.get("title"),
            tender_row.get("ai_types"),
            impact["impact"], impact["matched_phrase"],
            json.dumps(governance["data_categories"]),
            json.dumps(governance["human_oversight"]),
            json.dumps(governance["bias_fairness_testing"]),
            json.dumps(governance["failure_fallback"]),
            tender_row.get("tender_document_url"),
            governance.get("num_pages"),
        ),
    )
    conn.commit()


def load_all_profiles(conn) -> list:
    rows = conn.execute("SELECT * FROM tender_profiles").fetchall()
    cols = [d[0] for d in conn.execute("SELECT * FROM tender_profiles LIMIT 0").description]
    profiles = []
    for row in rows:
        d = dict(zip(cols, row))
        d["data_categories"] = json.loads(d.pop("data_categories_json"))
        d["human_oversight"] = json.loads(d.pop("human_oversight_json"))
        d["bias_fairness_testing"] = json.loads(d.pop("bias_fairness_json"))
        d["failure_fallback"] = json.loads(d.pop("failure_fallback_json"))
        profiles.append(d)
    return profiles


def load_profile(conn, tender_id: str) -> dict:
    for p in load_all_profiles(conn):
        if p["tender_id"] == tender_id:
            return p
    return None
