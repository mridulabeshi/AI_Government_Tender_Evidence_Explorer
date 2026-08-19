"""
Stage 0 — Data loading.

Loads the `rumourscape/tenders` Indian government procurement dataset
(~4.92M rows, CC BY 4.0, hosted on Hugging Face) into a local Parquet
file so every later stage can query it cheaply with DuckDB.

Usage:
    python -m src.data_loader                 # download real dataset
    python -m src.data_loader --sample 500     # generate a synthetic
                                                # sample instead (useful
                                                # for offline dev/demo)
"""

import argparse
import os
import random

import duckdb
import pandas as pd

from src import config

HF_PARQUET_URL = (
    "hf://datasets/rumourscape/tenders/data/train-00000-of-*.parquet"
)

REQUIRED_COLUMNS = [
    "tender_id", "organisation", "state", "title", "tender_description",
    "tender_type", "date", "contract_value", "selected_bidder",
    "detail_url", "tender_document_url",
]


def load_real_dataset() -> pd.DataFrame:
    """
    Pull the dataset straight from Hugging Face using DuckDB's hf:// reader
    and materialize it to local Parquet. Requires network access to
    huggingface.co and the `duckdb` httpfs/hf extensions.
    """
    con = duckdb.connect()
    con.execute("INSTALL httpfs; LOAD httpfs;")
    query = f"SELECT * FROM read_parquet('{HF_PARQUET_URL}')"
    df = con.execute(query).df()
    os.makedirs(config.DATA_DIR, exist_ok=True)
    df.to_parquet(config.RAW_PARQUET, index=False)
    return df


def generate_sample_dataset(n: int = 500, seed: int = 42) -> pd.DataFrame:
    """
    Build a synthetic-but-realistic stand-in dataset so the pipeline can be
    developed, tested, and demoed without network access. A fixed fraction
    of rows are deliberately AI-flavoured (some obviously, some only in the
    description, mirroring the "Smart City" vs "Intelligent Surveillance"
    distinction the project is built around).
    """
    random.seed(seed)

    states = ["Tamil Nadu", "Karnataka", "Maharashtra", "Delhi", "Uttar Pradesh",
              "Kerala", "West Bengal", "Gujarat", "Rajasthan", "Telangana"]
    orgs = ["Municipal Corporation", "Police Department", "Health Department",
            "Transport Department", "Revenue Department", "Smart City Mission",
            "Education Department", "Welfare Board"]

    ai_titles = [
        ("AI-Based Facial Recognition Surveillance System",
         "Supply, installation and commissioning of an AI-enabled facial "
         "recognition and video analytics system across public spaces, "
         "including CCTV integration and biometric matching against "
         "existing databases."),
        ("Deployment of Citizen Grievance Chatbot",
         "Development of a conversational AI / NLP based chatbot for "
         "citizen grievance redressal, integrated with the department "
         "portal and capable of natural language query resolution."),
        ("Predictive Traffic Management Analytics Platform",
         "Design and deployment of a machine learning based predictive "
         "analytics platform for real-time traffic flow prediction and "
         "signal optimization using historical and live sensor data."),
        ("Smart City Integrated Command and Control Platform",
         "Establishment of an integrated command and control centre "
         "including video wall, GIS mapping and dashboard software for "
         "municipal operations."),  # no real AI capability (decoy)
        ("Intelligent Surveillance System for Public Safety",
         "Procurement of an intelligent surveillance system including "
         "object detection, automated anomaly alerts and computer vision "
         "based crowd counting for public safety monitoring."),
        ("AI-Enabled Welfare Eligibility Screening System",
         "Automated decision support system using predictive analytics "
         "to assess beneficiary eligibility for welfare schemes based on "
         "demographic and financial records."),
        ("Generative AI Document Summarization Tool",
         "Procurement of a generative AI / LLM based tool for internal "
         "document summarization and drafting assistance for department "
         "staff."),
        ("Healthcare Diagnostic Support System",
         "AI-based computer vision system for radiology image analysis "
         "to assist doctors in diagnostic prioritization."),
        ("Automated Recruitment Screening Portal",
         "Development of a machine learning based resume screening and "
         "candidate ranking system for recruitment drives."),
        ("Speech Recognition Based IVR Upgrade",
         "Upgradation of existing IVR system with speech recognition and "
         "natural language processing capability for citizen helplines."),
    ]

    non_ai_titles = [
        ("Construction of Municipal Office Building",
         "Civil construction works for a new municipal administrative "
         "office building including electrical and plumbing works."),
        ("Supply of Office Stationery",
         "Annual rate contract for supply of stationery items to "
         "department offices."),
        ("Road Resurfacing Works",
         "Resurfacing and maintenance of arterial roads within municipal "
         "limits."),
        ("Procurement of School Furniture",
         "Supply and installation of desks and benches for government "
         "schools."),
        ("Drinking Water Pipeline Extension",
         "Laying of new drinking water supply pipelines in urban wards."),
        ("Solid Waste Collection Services",
         "Engagement of an agency for door-to-door solid waste collection "
         "and transportation."),
        ("Annual Maintenance Contract for Generators",
         "AMC for diesel generator sets installed at government "
         "buildings."),
    ]

    rows = []
    for i in range(n):
        is_ai = random.random() < 0.18  # ~18% genuinely AI-flavoured
        title, desc = random.choice(ai_titles if is_ai else non_ai_titles)
        state = random.choice(states)
        org = f"{state} {random.choice(orgs)}"
        rows.append({
            "tender_id": f"TND{100000 + i}",
            "organisation": org,
            "state": state,
            "title": title,
            "tender_description": desc,
            "tender_type": random.choice(["Open", "Limited", "Single"]),
            "date": f"2025-{random.randint(1,12):02d}-{random.randint(1,28):02d}",
            "contract_value": random.choice([500000, 1200000, 4500000, 9800000, 22000000]),
            "selected_bidder": f"Vendor Pvt Ltd {random.randint(1,50)}",
            "detail_url": f"https://example-tender-portal.gov.in/tender/{100000+i}",
            "tender_document_url": f"https://example-tender-portal.gov.in/docs/{100000+i}.pdf",
        })

    df = pd.DataFrame(rows)[REQUIRED_COLUMNS]
    os.makedirs(config.DATA_DIR, exist_ok=True)
    df.to_parquet(config.RAW_PARQUET, index=False)
    return df


def main():
    parser = argparse.ArgumentParser(description="Load the tender dataset.")
    parser.add_argument("--sample", type=int, default=0,
                         help="Generate N synthetic rows instead of "
                              "downloading the real dataset (offline dev).")
    args = parser.parse_args()

    if args.sample:
        df = generate_sample_dataset(args.sample)
        print(f"Generated synthetic dataset: {len(df)} rows -> {config.RAW_PARQUET}")
    else:
        df = load_real_dataset()
        print(f"Loaded real dataset: {len(df)} rows -> {config.RAW_PARQUET}")

    print(df.head(3).to_string())


if __name__ == "__main__":
    main()
