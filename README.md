# AI Government Tender Evidence Explorer

Mines Indian government tender records to find AI-related procurements,
then extracts evidence-linked governance findings (data use, human
oversight, bias testing, failure/fallback) from the underlying tender
documents.

## Project layout

```
ai_tender_explorer/
├── requirements.txt
├── dashboard.py                 # Streamlit app (3 views)
├── data/                        # generated Parquet/SQLite artifacts
└── src/
    ├── config.py                # keyword vocab, governance terms, impact rules
    ├── data_loader.py           # Stage 0 — load the tender corpus
    ├── ai_discovery.py          # Stage 1 — candidate filter + verification
    ├── pipeline.py              # Stage 2 — curated selection + orchestration
    ├── document_extractor.py    # Stage 3a — PDF/OCR extraction + chunking
    ├── governance_extractor.py  # Stage 3b — LLM evidence extraction (Claude)
    ├── impact_classifier.py     # Stage 4 — rule-based impact tier
    └── evidence_store.py        # SQLite persistence for tender profiles
```

## 1. Install

```bash
cd ai_tender_explorer
python -m venv venv && source venv/bin/activate     # optional but recommended
pip install -r requirements.txt
```

If you want the LLM-based governance extraction to run for real (not the
offline mock), also set:

```bash
export ANTHROPIC_API_KEY=sk-ant-...
```

## 2. Load the dataset

**Option A — real dataset** (needs network access to Hugging Face; this
pulls `rumourscape/tenders`, ~4.92M rows, via DuckDB's `hf://` reader):

```bash
python -m src.data_loader
```

**Option B — offline sample** (500 synthetic-but-realistic rows, useful
for development, testing, or a demo without network access):

```bash
python -m src.data_loader --sample 500
```

Either way this writes `data/tenders_raw.parquet`.

## 3. Run Stage 1 — AI Tender Discovery

```bash
python -m src.ai_discovery
```

This scans `title` + `tender_description` for the AI keyword vocabulary
(Pass A, run as a single DuckDB SQL query directly against the Parquet
file so it scales to millions of rows), then runs a verification pass
(Pass B) that assigns a `confidence` tier (High/Medium/Low) and an
`ai_types` bucket (Computer Vision, GenAI/LLM, Facial Recognition, ...) to
each candidate. Output: `data/ai_candidates.parquet` and
`data/ai_confirmed.parquet`.

## 4. Run Stage 2–4 — Curated Deep-Dive + Governance Extraction

```bash
# Offline / demo mode — uses built-in synthetic governance findings,
# no API key or real PDFs required:
python -m src.pipeline --top-n 15 --mock

# Live mode — downloads each selected tender's document and runs the
# real Claude-based extraction (needs ANTHROPIC_API_KEY):
python -m src.pipeline --top-n 15 --live
```

This selects the top N verified candidates (prioritising facial
recognition, surveillance, healthcare AI, welfare/eligibility systems,
etc.), runs each through document extraction + governance evidence
extraction, classifies impact, and writes everything to
`data/evidence.sqlite`.

To process one document directly (e.g. to inspect the raw extraction
JSON), you can also call the extractor stages individually:

```bash
python -m src.document_extractor path/to/tender.pdf
python -m src.governance_extractor path/to/tender.pdf
python -m src.impact_classifier
```

## 5. Launch the dashboard

```bash
streamlit run dashboard.py
```

Three tabs:

* **AI Capability Map** — counts of verified AI tenders by type and
  state, plus the full candidate table.
* **Tender Profile** — a governance summary card for one selected
  tender (data, human oversight, bias testing, failure/fallback, impact
  tier).
* **Evidence Explorer** — click any verdict to see the exact clause
  text and page number it was extracted from. A "Not Found" verdict is
  always shown with the explicit caveat that this means *no requirement
  was identified in the analyzed document*, not that the safeguard is
  absent.

## How the four governance dimensions are extracted

`governance_extractor.py` chunks each tender document into page-tagged
segments (`document_extractor.chunk_pages`) and sends each chunk to
Claude with a strict JSON-only prompt (see `EXTRACTION_SYSTEM_PROMPT`)
that:

1. Forbids inference — a dimension is only marked "Found"/"Required" if
   the chunk explicitly says so.
2. Requires an exact evidence quote + page number for every non-"Not
   Found" verdict.
3. Returns structured JSON only, which is merged across chunks per
   tender (the strongest verdict wins per `VERDICT_RANK`, so a "Required"
   found on page 47 isn't discarded just because an earlier chunk said
   "Not Found").

`impact_classifier.py` is intentionally *not* an LLM call — it's a
transparent, ordered keyword-rule table (`config.IMPACT_RULES`) so every
classification can be explained by pointing at the exact phrase that
triggered it.

## Extending beyond the MVP

* Swap `data_loader.generate_sample_dataset` for the real
  `fireboy21/tenders_aoc` corpus (mentioned in the pitch as the
  natural next step, since it links tender + Award-of-Contract records).
* Add a real document downloader in `pipeline.run_pipeline` (currently a
  placeholder comment) to fetch `tender_document_url` before calling
  `governance_extractor.extract_governance_evidence(pdf_path=...)`.
* Swap `evidence_store`'s SQLite backend for Postgres for concurrent
  writes at scale.
* Enable OCR (`pytesseract` + `tesseract-ocr`) for scanned tender PDFs —
  the hook is already in `document_extractor._ocr_page`.
