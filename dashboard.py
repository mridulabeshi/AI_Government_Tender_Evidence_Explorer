"""
AI Government Tender Evidence Explorer — Streamlit dashboard.

Three views, matching the pitch deck:
  1. AI Capability Map   — counts of discovered AI tenders by type/state
  2. Tender Profile       — governance summary card for one tender
  3. Evidence Explorer    — click a verdict, see the exact clause/page

Run with:
    streamlit run dashboard.py
"""

import pandas as pd
import plotly.express as px
import streamlit as st

from src import config, evidence_store

st.set_page_config(page_title="AI Government Tender Evidence Explorer", layout="wide")

VERDICT_ICON = {
    "Required": "🟢", "Override Available": "🟢", "Found": "🟢",
    "Unclear": "🟡", "Not Found": "⚪",
}


@st.cache_data
def load_capability_data():
    try:
        return pd.read_parquet(config.CONFIRMED_PARQUET)
    except FileNotFoundError:
        return pd.DataFrame()


def load_profiles():
    conn = evidence_store.get_connection()
    profiles = evidence_store.load_all_profiles(conn)
    conn.close()
    return profiles


def render_finding(label: str, finding: dict, key_prefix: str):
    verdict = finding.get("verdict", "Not Found")
    icon = VERDICT_ICON.get(verdict, "⚪")
    with st.container(border=True):
        st.markdown(f"**{label}**")
        st.markdown(f"### {icon} {verdict}")
        if finding.get("evidence"):
            if st.button("View Evidence →", key=f"{key_prefix}_{label}"):
                st.session_state[f"open_{key_prefix}_{label}"] = True
        if st.session_state.get(f"open_{key_prefix}_{label}"):
            st.markdown(f"**Clause / Page {finding.get('page', '—')}**")
            st.info(f'"{finding["evidence"]}"')
        elif verdict == "Not Found":
            st.caption("No explicit requirement was identified in the analyzed document.")


# ---------------------------------------------------------------------------
st.title("🔍 AI Government Tender Evidence Explorer")
st.caption(
    "Millions of real Indian government tender records → verified AI "
    "procurements → evidence-linked governance findings."
)

tab1, tab2, tab3 = st.tabs([
    "📊 AI Capability Map", "📄 Tender Profile", "🔬 Evidence Explorer"
])

# --- View 1: Capability Map -------------------------------------------------
with tab1:
    df = load_capability_data()
    if df.empty:
        st.warning(
            "No discovery data found. Run:\n\n"
            "`python -m src.data_loader --sample 500` then "
            "`python -m src.ai_discovery`"
        )
    else:
        verified = df[df["verified"]]
        col1, col2, col3 = st.columns(3)
        col1.metric("Rows scanned", f"{len(df):,}")
        col2.metric("Keyword candidates", f"{len(df):,}")
        col3.metric("Verified AI tenders", f"{len(verified):,}")

        exploded = verified.assign(
            ai_types=verified["ai_types"].str.split(", ")
        ).explode("ai_types")

        c1, c2 = st.columns(2)
        with c1:
            counts = exploded["ai_types"].value_counts().reset_index()
            counts.columns = ["AI Type", "Count"]
            fig = px.bar(counts, x="Count", y="AI Type", orientation="h",
                         title="AI Procurement Discovery by Type")
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            state_counts = verified["state"].value_counts().reset_index()
            state_counts.columns = ["State", "Count"]
            fig2 = px.bar(state_counts, x="State", y="Count",
                          title="Verified AI Tenders by State")
            st.plotly_chart(fig2, use_container_width=True)

        st.markdown("#### Verified candidates")
        st.dataframe(
            verified[["tender_id", "organisation", "state", "title",
                      "ai_types", "confidence"]],
            use_container_width=True, hide_index=True,
        )

# --- View 2 + 3: Tender Profile / Evidence Explorer -------------------------
profiles = load_profiles()

if not profiles:
    with tab2:
        st.warning(
            "No governance profiles yet. Run:\n\n"
            "`python -m src.pipeline --top-n 15 --mock`"
        )
    with tab3:
        st.info("Select a tender in the Tender Profile tab first.")
else:
    profile_labels = {
        p["tender_id"]: f"{p['tender_id']} — {p['title']}" for p in profiles
    }
    selected_id = st.sidebar.selectbox(
        "Select a tender", options=list(profile_labels.keys()),
        format_func=lambda x: profile_labels[x],
    )
    profile = next(p for p in profiles if p["tender_id"] == selected_id)

    with tab2:
        st.subheader(profile["title"])
        c1, c2, c3, c4 = st.columns(4)
        c1.markdown(f"**Organisation**\n\n{profile['organisation']}")
        c2.markdown(f"**State**\n\n{profile['state']}")
        c3.markdown(f"**AI Type**\n\n{profile['ai_types']}")
        icon = config.IMPACT_COLORS.get(profile["impact"], "⚪")
        c4.markdown(f"**Impact**\n\n{icon} {profile['impact']}")
        if profile.get("impact_matched_phrase"):
            st.caption(f"Impact classification triggered by phrase: \"{profile['impact_matched_phrase']}\"")

        st.markdown("#### Governance Summary")
        g1, g2, g3, g4 = st.columns(4)
        with g1:
            cats = profile["data_categories"]
            label = ", ".join(c["category"] for c in cats) if cats else "Not identified"
            st.metric("Data", "")
            st.caption(label)
        with g2:
            ho = profile["human_oversight"]
            st.metric("Human Oversight", f"{VERDICT_ICON.get(ho['verdict'],'⚪')} {ho['verdict']}")
        with g3:
            bf = profile["bias_fairness_testing"]
            st.metric("Bias Testing", f"{VERDICT_ICON.get(bf['verdict'],'⚪')} {bf['verdict']}")
        with g4:
            ff = profile["failure_fallback"]
            st.metric("Failure/Fallback", f"{VERDICT_ICON.get(ff['verdict'],'⚪')} {ff['verdict']}")

        st.caption("Open the Evidence Explorer tab to inspect the clause behind any of these verdicts.")

    with tab3:
        st.subheader(f"Evidence Explorer — {profile['title']}")
        st.caption(f"Source document analyzed: {profile.get('source_document_url', '—')} "
                   f"({profile.get('num_pages', '—')} pages)")

        st.markdown("##### A. Data Categories")
        if profile["data_categories"]:
            for cat in profile["data_categories"]:
                with st.container(border=True):
                    st.markdown(f"**{cat['category']}**")
                    st.info(f'"{cat["evidence"]}" — Page {cat["page"]}')
        else:
            st.caption("⚪ No explicit data category was identified in the analyzed document.")

        col_a, col_b, col_c = st.columns(3)
        with col_a:
            render_finding("Human Oversight", profile["human_oversight"], key_prefix=profile["tender_id"])
        with col_b:
            render_finding("Bias Testing", profile["bias_fairness_testing"], key_prefix=profile["tender_id"])
        with col_c:
            render_finding("Failure / Fallback", profile["failure_fallback"], key_prefix=profile["tender_id"])

        st.markdown("---")
        st.markdown(
            "> **Not Found ≠ Doesn't Exist.** A 'Not Found' verdict means "
            "no explicit requirement was identified in the analyzed "
            "document — not that the safeguard is absent from the system."
        )

st.sidebar.markdown("---")
st.sidebar.markdown(
    "**Pipeline stages**\n\n"
    "1. `data_loader.py` — load corpus\n"
    "2. `ai_discovery.py` — keyword + verification filter\n"
    "3. `pipeline.py` — curated selection + orchestration\n"
    "4. `governance_extractor.py` — LLM evidence extraction\n"
    "5. `impact_classifier.py` — rule-based impact tier\n"
    "6. `evidence_store.py` — SQLite persistence"
)
