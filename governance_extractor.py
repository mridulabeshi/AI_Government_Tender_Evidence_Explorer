"""
Stage 3b — Governance Evidence Extraction.

Feeds each document chunk (see document_extractor.chunk_pages) to Claude
with a strict JSON-only extraction prompt covering the four governance
dimensions:

    A. Data categories
    B. Human oversight
    C. Bias / fairness testing
    D. Failure & fallback mechanisms

The model is explicitly instructed to only report a finding when the
clause is actually present, and to always return the exact supporting
text + page number. Chunk-level results are then merged per tender:
the first "Found"/"Required"/etc. verdict wins, but every chunk is
still consulted so nothing is missed just because it fell in a later page.

This module talks to the Anthropic API directly (not the in-browser
artifact fetch pattern) since it runs as a standalone backend/ETL step.
Set ANTHROPIC_API_KEY in the environment before running.
"""

import json
import os

import config
from document_extractor import extract_pages, chunk_pages, extract_pages_from_text

EXTRACTION_SYSTEM_PROMPT = """You are a careful legal/technical analyst reading Indian government tender documents to extract evidence about AI governance requirements.

You will be given one chunk of a tender document, tagged with [Page N] markers.

For EACH of the four dimensions below, decide a verdict using ONLY what is explicitly present in this chunk. Do not infer, assume, or use outside knowledge about how such systems "usually" work.

1. data_categories: what categories of data the AI system is specified to use (e.g. biometric, facial images, CCTV/video, location, voice, health, financial, government records, personal information). If none mentioned in this chunk, return an empty list.

2. human_oversight: verdict must be one of "Required", "Override Available", "Unclear", or "Not Found".
   - "Required": the document explicitly requires human review/approval of AI outputs or decisions.
   - "Override Available": a human can override/reverse the AI's output, but review isn't mandatory.
   - "Unclear": human involvement is mentioned but its role/authority is ambiguous.
   - "Not Found": no relevant text in this chunk.

3. bias_fairness_testing: verdict must be "Found" or "Not Found". "Found" only if the chunk explicitly requires bias, fairness, or demographic-performance testing.

4. failure_fallback: verdict must be "Found", "Unclear", or "Not Found", covering human override, manual fallback, emergency shutdown, model rollback, or incident reporting requirements.

For every verdict that is NOT "Not Found", you MUST include:
   - "evidence": the exact supporting quote from the chunk (keep it short, under 40 words)
   - "page": the page number from the nearest preceding [Page N] marker

Respond with ONLY valid JSON, no markdown fences, no preamble, in exactly this shape:
{
  "data_categories": [{"category": "...", "evidence": "...", "page": N}, ...],
  "human_oversight": {"verdict": "...", "evidence": "...", "page": N},
  "bias_fairness_testing": {"verdict": "...", "evidence": "...", "page": N},
  "failure_fallback": {"verdict": "...", "evidence": "...", "page": N}
}
If a dimension has verdict "Not Found", omit "evidence" and "page" for it (set them to null).
"""

VERDICT_RANK = {
    # higher rank = stronger evidence, used when merging chunks
    "human_oversight": {"Not Found": 0, "Unclear": 1, "Override Available": 2, "Required": 3},
    "bias_fairness_testing": {"Not Found": 0, "Found": 1},
    "failure_fallback": {"Not Found": 0, "Unclear": 1, "Found": 2},
}


def _call_claude(chunk_text: str) -> dict:
    """
    Calls the Anthropic Messages API with the extraction prompt. Isolated
    into its own function so it's easy to swap models or mock in tests.
    """
    import anthropic
    client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from env

    resp = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1500,
        system=EXTRACTION_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": chunk_text}],
    )
    raw = "".join(b.text for b in resp.content if getattr(b, "type", "") == "text")
    raw = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        # Fail safe: treat an unparsable response as "nothing found" rather
        # than crashing the whole batch job.
        return {
            "data_categories": [],
            "human_oversight": {"verdict": "Not Found", "evidence": None, "page": None},
            "bias_fairness_testing": {"verdict": "Not Found", "evidence": None, "page": None},
            "failure_fallback": {"verdict": "Not Found", "evidence": None, "page": None},
        }


def _merge_scalar(dimension: str, current: dict, new: dict) -> dict:
    """Keeps whichever of current/new has the stronger verdict."""
    if current is None:
        return new
    rank = VERDICT_RANK[dimension]
    if rank.get(new.get("verdict", "Not Found"), 0) > rank.get(current.get("verdict", "Not Found"), 0):
        return new
    return current


def extract_governance_evidence(pdf_path: str = None, raw_text: str = None) -> dict:
    """
    Full Stage-3 run for a single tender document. Provide either a PDF
    path or raw text (for portals that expose the notice as HTML/plaintext
    rather than a downloadable PDF).

    Returns a dict with the four merged dimension results plus the list of
    chunk-level results for audit/debugging.
    """
    if pdf_path:
        pages = extract_pages(pdf_path)
    elif raw_text:
        pages = extract_pages_from_text(raw_text)
    else:
        raise ValueError("Provide either pdf_path or raw_text")

    chunks = chunk_pages(pages)

    merged_data_categories = []
    merged_human_oversight = None
    merged_bias = None
    merged_failure = None
    chunk_results = []

    for chunk in chunks:
        result = _call_claude(chunk["text"])
        chunk_results.append({"page_start": chunk["page_start"], "page_end": chunk["page_end"], "result": result})

        merged_data_categories.extend(result.get("data_categories", []))
        merged_human_oversight = _merge_scalar(
            "human_oversight", merged_human_oversight,
            result.get("human_oversight", {"verdict": "Not Found"}))
        merged_bias = _merge_scalar(
            "bias_fairness_testing", merged_bias,
            result.get("bias_fairness_testing", {"verdict": "Not Found"}))
        merged_failure = _merge_scalar(
            "failure_fallback", merged_failure,
            result.get("failure_fallback", {"verdict": "Not Found"}))

    # de-dupe data categories by category name, keep first evidence seen
    seen, deduped = set(), []
    for item in merged_data_categories:
        key = item.get("category", "").lower()
        if key and key not in seen:
            seen.add(key)
            deduped.append(item)

    return {
        "data_categories": deduped,
        "human_oversight": merged_human_oversight or {"verdict": "Not Found", "evidence": None, "page": None},
        "bias_fairness_testing": merged_bias or {"verdict": "Not Found", "evidence": None, "page": None},
        "failure_fallback": merged_failure or {"verdict": "Not Found", "evidence": None, "page": None},
        "chunk_results": chunk_results,
        "num_pages": len(pages),
        "num_chunks": len(chunks),
    }


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python governance_extractor.py <path_to_pdf>")
        sys.exit(0)
    result = extract_governance_evidence(pdf_path=sys.argv[1])
    print(json.dumps(result, indent=2))
