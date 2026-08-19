"""
Stage 2 + orchestration — Curated Deep-Dive.

Selects the top N verified AI candidates (prioritising high-impact
categories: facial recognition, surveillance, healthcare AI, welfare
systems, etc.) and runs each through Stage 3 (document extraction +
governance evidence) and Stage 4 (impact classification), persisting
results to the evidence database.

Usage:
    python pipeline.py --top-n 15 --mock
        --mock uses a built-in synthetic document generator instead of
        downloading + calling Claude on real PDFs, so the full pipeline
        can be demoed/tested offline. Drop --mock (and set
        ANTHROPIC_API_KEY) to run against real tender documents.
"""

import argparse
import random

import pandas as pd

import config
import evidence_store
from impact_classifier import classify_impact

PRIORITY_KEYWORDS = [
    "facial recognition", "surveillance", "biometric", "computer vision",
    "chatbot", "generative ai", "healthcare", "welfare", "eligibility",
    "predictive",
]


def select_deep_dive_candidates(verified_df: pd.DataFrame, top_n: int = 15) -> pd.DataFrame:
    verified_df = verified_df[verified_df["verified"]].copy()

    def priority_score(row):
        text = f"{row['title']} {row['tender_description']}".lower()
        return sum(1 for kw in PRIORITY_KEYWORDS if kw in text)

    verified_df["priority_score"] = verified_df.apply(priority_score, axis=1)
    verified_df = verified_df.sort_values(
        ["priority_score", "confidence"], ascending=[False, True]
    )
    return verified_df.head(top_n)


def _mock_governance_result(title: str, description: str) -> dict:
    """
    Stand-in for governance_extractor.extract_governance_evidence(), used
    when --mock is passed (no ANTHROPIC_API_KEY / no real PDFs available).
    Produces plausible, varied evidence so the dashboard/demo has
    something realistic to show, INCLUDING tenders where dimensions are
    genuinely "Not Found" — that's the point of the project.
    """
    text = f"{title} {description}".lower()
    random.seed(hash(title) % (2**32))

    data_categories = []
    if "facial" in text or "biometric" in text:
        data_categories.append({"category": "Biometric / facial images", "evidence": "biometric matching against existing databases", "page": 18})
    if "cctv" in text or "surveillance" in text or "video" in text:
        data_categories.append({"category": "CCTV / video footage", "evidence": "CCTV integration across public spaces", "page": 12})
    if "health" in text or "diagnos" in text:
        data_categories.append({"category": "Health data", "evidence": "radiology image analysis for diagnostic prioritization", "page": 9})
    if "welfare" in text or "eligibility" in text:
        data_categories.append({"category": "Financial / demographic records", "evidence": "beneficiary eligibility based on demographic and financial records", "page": 6})
    if "chatbot" in text or "grievance" in text:
        data_categories.append({"category": "Personal information", "evidence": "citizen grievance queries and contact details", "page": 4})

    if random.random() < 0.55:
        human_oversight = {"verdict": "Required", "evidence": "Human review shall be conducted before any automated action is finalized.", "page": 47}
    elif random.random() < 0.5:
        human_oversight = {"verdict": "Unclear", "evidence": "The system shall support human intervention where necessary.", "page": 22}
    else:
        human_oversight = {"verdict": "Not Found", "evidence": None, "page": None}

    if "welfare" in text or "recruit" in text:
        bias = {"verdict": "Not Found", "evidence": None, "page": None}
    else:
        bias = random.choice([
            {"verdict": "Not Found", "evidence": None, "page": None},
            {"verdict": "Not Found", "evidence": None, "page": None},
            {"verdict": "Found", "evidence": "The vendor shall demonstrate testing for demographic performance parity prior to go-live.", "page": 31},
        ])

    failure = random.choice([
        {"verdict": "Found", "evidence": "Manual fallback procedures shall be documented and activated on system failure.", "page": 52},
        {"verdict": "Unclear", "evidence": "The system shall include appropriate safeguards.", "page": 40},
        {"verdict": "Not Found", "evidence": None, "page": None},
    ])

    return {
        "data_categories": data_categories,
        "human_oversight": human_oversight,
        "bias_fairness_testing": bias,
        "failure_fallback": failure,
        "num_pages": random.randint(20, 60),
        "num_chunks": random.randint(3, 8),
    }


def run_pipeline(top_n: int = 15, mock: bool = True):
    verified_df = pd.read_parquet(config.CONFIRMED_PARQUET)
    selected = select_deep_dive_candidates(verified_df, top_n=top_n)

    conn = evidence_store.get_connection()
    processed = []

    for _, row in selected.iterrows():
        if mock:
            governance = _mock_governance_result(row["title"], row["tender_description"])
        else:
            from governance_extractor import extract_governance_evidence
            # In a real run you'd first download row["tender_document_url"]
            # to a local PDF path and pass pdf_path=... here.
            governance = extract_governance_evidence(raw_text=row["tender_description"])

        impact = classify_impact(row["title"], row["tender_description"], row.get("ai_types", ""))

        evidence_store.save_tender_profile(conn, row.to_dict(), governance, impact)
        processed.append(row["tender_id"])

    conn.close()
    print(f"Processed {len(processed)} tenders into evidence DB -> {config.EVIDENCE_DB}")
    return processed


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--top-n", type=int, default=15)
    parser.add_argument("--mock", action="store_true", default=True)
    parser.add_argument("--live", dest="mock", action="store_false",
                         help="Use real Claude extraction instead of mock data")
    args = parser.parse_args()
    run_pipeline(top_n=args.top_n, mock=args.mock)
