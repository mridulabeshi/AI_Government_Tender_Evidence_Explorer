"""
Stage 4 — Impact Classification.

Deterministic, explainable, rule-based classification of a confirmed AI
tender into Low / Moderate / High / Very High impact, based on
`IMPACT_RULES` in config.py. No opaque model score — every classification
can point to exactly which phrase in the title/description/ai_types
triggered it, which is shown to the user alongside the verdict.
"""

from src import config


def classify_impact(title: str, description: str, ai_types: str = "") -> dict:
    """
    Returns {"impact": "Very High"/"High"/"Moderate"/"Low"/"Unclassified",
             "matched_phrase": str or None,
             "icon": str}
    Checks IMPACT_RULES top-down (Very High first); first match wins, since
    a tender that touches facial recognition should never be down-graded
    just because it also mentions a "chatbot" component.
    """
    haystack = " ".join([
        title or "", description or "", ai_types or ""
    ]).lower()

    for impact_level, phrases in config.IMPACT_RULES:
        for phrase in phrases:
            if phrase in haystack:
                return {
                    "impact": impact_level,
                    "matched_phrase": phrase,
                    "icon": config.IMPACT_COLORS[impact_level],
                }

    return {
        "impact": "Unclassified",
        "matched_phrase": None,
        "icon": config.IMPACT_COLORS["Unclassified"],
    }


if __name__ == "__main__":
    tests = [
        ("AI-Based Facial Recognition Surveillance System", "biometric matching"),
        ("Automated Recruitment Screening Portal", "resume screening"),
        ("Deployment of Citizen Grievance Chatbot", "NLP based chatbot"),
        ("Generative AI Document Summarization Tool", "internal drafting"),
        ("Road Resurfacing Works", "civil construction"),
    ]
    for title, desc in tests:
        print(title, "->", classify_impact(title, desc))
