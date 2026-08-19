"""
Shared configuration: paths, keyword vocabulary, governance dimension
labels, and impact-classification rules.
"""

import os

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
DOCS_DIR = os.path.join(BASE_DIR, "sample_docs")

RAW_PARQUET = os.path.join(DATA_DIR, "tenders_raw.parquet")
CANDIDATES_PARQUET = os.path.join(DATA_DIR, "ai_candidates.parquet")
CONFIRMED_PARQUET = os.path.join(DATA_DIR, "ai_confirmed.parquet")
EVIDENCE_DB = os.path.join(DATA_DIR, "evidence.sqlite")

# ---------------------------------------------------------------------------
# Stage 1 — keyword vocabulary used for the first-pass candidate filter.
# Matching is case-insensitive and word-boundary aware (see ai_discovery.py).
# ---------------------------------------------------------------------------
AI_KEYWORDS = [
    "artificial intelligence", "machine learning", "deep learning",
    "ai/ml", "ai-ml", "generative ai", "genai", "llm",
    "large language model", "computer vision", "facial recognition",
    "facial identification", "nlp", "natural language processing",
    "predictive analytics", "video analytics", "object detection",
    "speech recognition", "conversational ai", "chatbot",
    "automated decision", "biometric identification", "biometric",
    "neural network", "image recognition", "pattern recognition",
    "intelligent surveillance", "smart surveillance",
]

# Maps a keyword hit -> a coarse AI-type bucket used for the capability map.
KEYWORD_TO_AI_TYPE = {
    "artificial intelligence": "AI/ML", "machine learning": "AI/ML",
    "deep learning": "AI/ML", "ai/ml": "AI/ML", "ai-ml": "AI/ML",
    "neural network": "AI/ML",
    "generative ai": "GenAI / LLM", "genai": "GenAI / LLM",
    "llm": "GenAI / LLM", "large language model": "GenAI / LLM",
    "computer vision": "Computer Vision", "object detection": "Computer Vision",
    "image recognition": "Computer Vision", "pattern recognition": "Computer Vision",
    "facial recognition": "Facial Recognition",
    "facial identification": "Facial Recognition",
    "biometric identification": "Facial Recognition", "biometric": "Facial Recognition",
    "intelligent surveillance": "Facial Recognition", "smart surveillance": "Facial Recognition",
    "nlp": "NLP", "natural language processing": "NLP",
    "predictive analytics": "Predictive Analytics",
    "video analytics": "Computer Vision",
    "speech recognition": "Conversational AI",
    "conversational ai": "Conversational AI", "chatbot": "Conversational AI",
    "automated decision": "Automated Decision Systems",
}

# ---------------------------------------------------------------------------
# Stage 3 — governance dimension search terms (used as a lexical backstop /
# sanity check alongside the LLM extraction).
# ---------------------------------------------------------------------------
DATA_CATEGORY_TERMS = [
    "personal information", "personal data", "biometric data",
    "facial image", "cctv", "video footage", "location data", "gps",
    "voice data", "health data", "medical record", "financial information",
    "government records", "aadhaar", "demographic data",
]

HUMAN_OVERSIGHT_TERMS = [
    "human review", "human oversight", "human in the loop",
    "manual review", "human override", "shall be reviewed by",
    "approval of the officer", "final decision shall rest with",
    "human intervention",
]

BIAS_FAIRNESS_TERMS = [
    "bias testing", "bias test", "fairness", "demographic performance",
    "disparate impact", "representative dataset", "equal treatment",
    "performance across population", "gender bias", "discriminat",
]

FAILURE_FALLBACK_TERMS = [
    "human override", "manual fallback", "emergency shutdown",
    "model rollback", "incident report", "error handling",
    "disaster recovery", "fallback mechanism", "kill switch",
    "business continuity",
]

# ---------------------------------------------------------------------------
# Stage 4 — deterministic impact classification.
# Checked top-down; first matching bucket wins.
# ---------------------------------------------------------------------------
IMPACT_RULES = [
    ("Very High", [
        "facial recognition", "biometric identification", "biometric",
        "welfare eligibility", "automated welfare", "law enforcement",
        "predictive policing", "criminal", "surveillance",
    ]),
    ("High", [
        "recruitment", "screening", "eligibility assessment",
        "healthcare prioriti", "credit scoring", "loan approval",
        "diagnosis",
    ]),
    ("Moderate", [
        "chatbot", "citizen service", "traffic prediction",
        "decision support", "predictive analytics", "video analytics",
    ]),
    ("Low", [
        "summarization", "internal search", "administrative automation",
        "document classification", "translation",
    ]),
]
IMPACT_COLORS = {
    "Very High": "🔴", "High": "🟠", "Moderate": "🟡",
    "Low": "🟢", "Unclassified": "⚪",
}

VERDICT_ICONS = {
    "Required": "🟢", "Override Available": "🟢", "Found": "🟢",
    "Unclear": "🟡", "Not Found": "⚪",
}
