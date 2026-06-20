import streamlit as st

st.set_page_config(
    page_title="Research Gap Finder",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Global Styles ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1a1a2e;
        margin-bottom: 0.2rem;
    }
    .subtitle {
        font-size: 1.05rem;
        color: #555;
        margin-bottom: 1.5rem;
    }
    .section-header {
        font-size: 1.3rem;
        font-weight: 600;
        color: #1a1a2e;
        border-left: 4px solid #4a90d9;
        padding-left: 0.6rem;
        margin-top: 1.2rem;
        margin-bottom: 0.5rem;
    }
    .info-box {
        background: #f0f4ff;
        border: 1px solid #c5d5f0;
        border-radius: 6px;
        padding: 0.9rem 1.1rem;
        margin-bottom: 1rem;
        font-size: 0.92rem;
        color: #333;
    }
    .warning-box {
        background: #fff8e1;
        border: 1px solid #f0c040;
        border-radius: 6px;
        padding: 0.9rem 1.1rem;
        margin-bottom: 1rem;
        font-size: 0.92rem;
        color: #555;
    }
    div[data-testid="stExpander"] {
        border: 1px solid #e0e0e0;
        border-radius: 6px;
        margin-bottom: 0.5rem;
    }
    .stTabs [data-baseweb="tab"] {
        font-size: 0.9rem;
        font-weight: 500;
    }
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### Research Gap Finder")
    st.markdown("---")

    st.markdown("**How to use this tool**")
    st.markdown("""
1. Enter your research topic in plain English
2. Choose how many papers to retrieve per source
3. Optionally enable full text retrieval
4. Click **Run Analysis**
5. Navigate the tabs to explore results
    """)

    st.markdown("---")
    st.markdown("**Analysis Settings**")

    max_per_source = st.slider(
        "Papers per database",
        min_value=10,
        max_value=100,
        value=30,
        step=10,
        help="More papers = more accurate results but longer wait time. 30 is recommended for a first run."
    )

    st.markdown(f"""
<div class="info-box">
Estimated retrieval time at {max_per_source} papers per source:<br>
<b>{max_per_source * 3 // 60 + 1}–{max_per_source * 3 // 30 + 2} minutes</b>
</div>
""", unsafe_allow_html=True)

    st.markdown("---")

    fetch_fulltext = st.toggle(
        "Fetch full text where available",
        value=False,
        help=(
            "Attempts to retrieve full paper text from PubMed Central, "
            "arXiv, and Unpaywall open access PDFs. "
            "Increases analysis time by 2-5 minutes but improves accuracy. "
            "Falls back to abstract if full text is unavailable."
        )
    )

    if fetch_fulltext:
        st.markdown("""
<div class="warning-box">
Full text mode is enabled. Expect an additional 2-5 minutes.
Approximately 40-60% of papers will have full text available.
</div>
""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("**Active Databases**")
    st.markdown("""
| Database | Coverage |
|---|---|
| PubMed | Biomedical |
| OpenAlex | All fields |
| Semantic Scholar | All fields |
| arXiv | CS / Physics |
    """)

    # ── NEW: Source Selection ───────────────────────────────────────────────
    st.markdown("---")
    st.markdown("**Choose Data Sources**")
    st.markdown("Select one or more databases to search. Unchecking all will query all available sources.")

    all_sources = ["PubMed", "OpenAlex", "Semantic Scholar", "arXiv"]
    selected_sources = st.multiselect(
        "Sources",
        options=all_sources,
        default=all_sources,
        help="Limit retrieval to specific databases for faster and more focused results."
    )
    st.caption("Tip: Uncheck sources to reduce retrieval time and avoid cross‑field noise.")

    st.markdown("---")
    st.markdown("**Field Detection**")
    st.markdown("""
The system automatically detects whether your topic is medical,
computer science, physics, economics, or another field and
queries the most relevant databases accordingly.
    """)

    st.markdown("---")
    st.markdown("**Cache Management**")

    try:
        from src.retrieval.cache import list_cached_queries, clear_cache
        cached_queries = list_cached_queries()
        if cached_queries:
            st.caption(f"{len(cached_queries)} topic(s) cached locally")
            for cq in cached_queries[:3]:
                st.caption(
                    f"- {cq['query'][:35]} "
                    f"({cq['papers']} papers, {cq['age_days']}d ago)"
                )
            if st.button("Clear Cache", key="clear_cache_btn"):
                deleted = clear_cache()
                st.success(f"Cleared {deleted} cached result(s)")
        else:
            st.caption("No cached results yet")
            st.caption("Results are cached for 7 days after first search")
    except Exception:
        st.caption("Cache unavailable")

    st.markdown("---")
    st.caption(
        "Built with Streamlit, Sentence Transformers, "
        "Scikit-learn, Plotly, NetworkX"
    )


# ── Main Header ───────────────────────────────────────────────────────────────
st.markdown(
    '<div class="main-title">Research Gap Finder</div>',
    unsafe_allow_html=True
)
st.markdown(
    '<div class="subtitle">Automated literature analysis — retrieve papers, '
    'map the knowledge landscape, and identify what has not been studied yet.</div>',
    unsafe_allow_html=True
)

st.markdown("---")

# ── What This Tool Does ───────────────────────────────────────────────────────
with st.expander("What does this tool do and how does it work?", expanded=False):
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
**What it does**

Most researchers spend weeks reading through hundreds of papers just to figure
out what has not been studied yet. This tool automates that process.

You enter any research topic. The system retrieves papers from academic
databases, analyses them using NLP, and produces a structured report
identifying:

- Which subtopics are well-studied
- Which subtopics have almost no papers (research gaps)
- Which papers reach contradictory conclusions
- Which population groups are underrepresented in the literature
- How papers relate to each other in a visual network
        """)
    with col2:
        st.markdown("""
**How it works — step by step**

1. **Retrieval** — Papers are fetched from PubMed, OpenAlex, Semantic
   Scholar, and arXiv using their free APIs

2. **Full text** — Where available, full paper text is retrieved from
   PubMed Central, arXiv PDFs, and Unpaywall open access links

3. **Embeddings** — Each paper is converted into a 384-dimensional vector
   using a Sentence Transformer model. Papers about similar topics get
   similar vectors.

4. **Clustering** — Papers are grouped into subtopic clusters automatically.
   The system determines the number of clusters without being told.

5. **Gap detection** — Clusters with very few papers, or clusters adjacent
   to well-studied areas but themselves thin, are flagged as gaps.

6. **Contradiction detection** — Pairs of papers on the same topic but with
   opposing conclusions are identified.

7. **Report** — All findings are compiled into a structured plain-English
   report you can download.
        """)

st.markdown("---")

# ── Input Section ─────────────────────────────────────────────────────────────
st.markdown(
    '<div class="section-header">Enter Your Research Topic</div>',
    unsafe_allow_html=True
)

st.markdown("""
<div class="info-box">
<b>Tips for a good query:</b> Be specific but not too narrow.
"Depression treatment in adolescents" works better than "depression" (too broad)
or "CBT outcomes in 14-year-old females in Scotland" (too narrow).
The system works for any academic field — medicine, computer science,
economics, physics, social science, law, and more.
</div>
""", unsafe_allow_html=True)

col_input, col_button = st.columns([4, 1])
with col_input:
    topic = st.text_input(
        "Research topic",
        placeholder=(
            "e.g.  depression treatment in adolescents  |  "
            "transformer models for code generation  |  "
            "renewable energy storage"
        ),
        label_visibility="collapsed"
    )
with col_button:
    run_button = st.button(
        "Run Analysis",
        type="primary",
        use_container_width=True
    )

st.markdown("""
<div class="warning-box">
<b>Note on wait time:</b> The first run takes 3-8 minutes depending on
settings. Most of this time is spent waiting for academic databases to
respond. Do not close the browser tab while it runs. Subsequent searches
on the same topic load from cache in under 15 seconds.
</div>
""", unsafe_allow_html=True)

if run_button and not topic.strip():
    st.error("Please enter a research topic before running the analysis.")
    st.stop()

# ── Pipeline ──────────────────────────────────────────────────────────────────
if run_button and topic.strip():

    with st.spinner("Loading analysis modules..."):
        import pandas as pd
        import plotly.express as px
        import plotly.graph_objects as go
        import numpy as np
        from dotenv import load_dotenv
        from src.retrieval.retriever import retrieve_all_papers
        from src.processing.embeddings import generate_embeddings
        from src.processing.clustering import (
            cluster_papers, describe_clusters, get_cluster_keywords
        )
        from src.analysis.gap_detector import (
            detect_research_gaps, format_gaps_as_text
        )
        from src.processing.contradiction import detect_contradictions
        from src.analysis.demographics import (
            analyse_demographics,
            identify_demographic_gaps,
            format_demographic_report
        )
        from src.report.generator import generate_report
        from src.visualisation.citation_graph import (
            build_similarity_network,
            compute_layout,
            render_citation_network,
            get_network_stats
        )
        load_dotenv()

    st.markdown("---")
    st.markdown(
        '<div class="section-header">Analysis Progress</div>',
        unsafe_allow_html=True
    )

    progress_bar = st.progress(0)
    status_text = st.empty()

    def update_progress(step, total, message):
        progress_bar.progress(step / total)
        status_text.markdown(f"**Step {step} of {total}** — {message}")

    try:
        update_progress(1, 7,
                        "Retrieving papers" +
                        (" and full texts" if fetch_fulltext else "") +
                        " from academic databases..."
                        )
        df = retrieve_all_papers(
            topic,
            max_per_source=max_per_source,
            fetch_fulltext=fetch_fulltext,
            sources=selected_sources   # <-- NEW: pass selected sources
        )

        if df.empty or len(df) < 3:
            st.error(
                "Not enough papers were retrieved for this topic. "
                "Try broadening your query or increasing papers per source."
            )
            st.stop()

        update_progress(2, 7,
                        f"Generating semantic embeddings for {len(df)} papers..."
                        )
        df, embeddings = generate_embeddings(df)

        update_progress(3, 7, "Clustering papers into subtopic groups...")
        df, reduced = cluster_papers(embeddings, df, min_cluster_size=2)
        summary = describe_clusters(df)
        keywords = get_cluster_keywords(df)

        update_progress(4, 7, "Identifying research gaps...")
        gaps = detect_research_gaps(df, embeddings, summary, keywords)
        gap_report = format_gaps_as_text(gaps, topic)

        update_progress(5, 7, "Detecting contradictions between papers...")
        contradictions = detect_contradictions(
            df, embeddings, similarity_threshold=0.55
        )

        update_progress(6, 7, "Analysing demographic representation...")
        demo_results = analyse_demographics(df)
        demo_gaps = identify_demographic_gaps(demo_results, df)
        demo_report = format_demographic_report(demo_gaps, topic)

        update_progress(7, 7, "Generating structured report...")
        ai_report = generate_report(
            topic, df, summary, keywords, gaps, contradictions, demo_gaps
        )

        progress_bar.progress(1.0)
        status_text.markdown("**Analysis complete.**")

        G = build_similarity_network(df, embeddings, similarity_threshold=0.50)
        pos = compute_layout(G)
        fig_network = render_citation_network(G, pos, df, keywords)
        network_stats = get_network_stats(G)

    except Exception as e:
        st.error(f"An error occurred during analysis: {str(e)}")
        st.caption(
            "If this persists, try a different topic or "
            "reduce papers per source."
        )
        st.stop()

    # ── Summary Metrics ───────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown(
        '<div class="section-header">Summary of Results</div>',
        unsafe_allow_html=True
    )

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Papers Retrieved", len(df))
    m2.metric(
        "Subtopic Clusters",
        len(summary[summary["cluster_id"] != -1])
    )
    m3.metric("Research Gaps", len(gaps))
    m4.metric("Contradictions", len(contradictions))
    m5.metric("Demographic Gaps", len(demo_gaps))

    sources = df["source"].value_counts().to_dict()
    source_str = "  |  ".join(f"{k}: {v}" for k, v in sources.items())
    st.caption(f"Sources: {source_str}")

    if fetch_fulltext and "fulltext_available" in df.columns:
        ft_count = int(df["fulltext_available"].sum())
        st.caption(
            f"Full text retrieved for {ft_count} of {len(df)} papers "
            f"({round(ft_count / len(df) * 100)}%)"
        )

    st.markdown("---")

    # ── Tabs ──────────────────────────────────────────────────────────────────
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "Knowledge Landscape",
        "Research Gaps",
        "Contradictions",
        "Demographics",
        "Citation Network",
        "Full Report",
        "All Papers"
    ])

    # ── TAB 1: Knowledge Landscape ────────────────────────────────────────────
    with tab1:
        st.markdown(
            '<div class="section-header">Knowledge Landscape</div>',
            unsafe_allow_html=True
        )
        st.markdown("""
<div class="info-box">
<b>What you are looking at:</b> The chart below shows how papers on your
topic are distributed across distinct subtopics. Each bar is a cluster of
papers discussing a similar angle. A tall bar means that subtopic is
well-studied. A very short bar means very few researchers have worked on
that angle. Green = well-studied. Yellow = moderate coverage.
Red = underexplored.
</div>
""", unsafe_allow_html=True)

        plot_df = summary[summary["cluster_id"] != -1].copy()
        plot_df["keywords"] = plot_df["cluster_id"].map(
            lambda cid: ", ".join(keywords.get(cid, ["unknown"])[:3])
        )

        fig = px.bar(
            plot_df,
            x="label",
            y="paper_count",
            color="density",
            color_discrete_map={
                "Dense": "#2ecc71",
                "Moderate": "#f39c12",
                "Sparse": "#e74c3c"
            },
            hover_data={"keywords": True, "paper_count": True},
            labels={
                "paper_count": "Number of Papers",
                "label": "Subtopic Cluster",
                "density": "Coverage Level"
            },
            title=f"Distribution of Research Effort — {topic.title()}"
        )
        fig.update_layout(
            height=420,
            plot_bgcolor="#fafafa",
            paper_bgcolor="#ffffff",
            font=dict(size=12),
            showlegend=True
        )
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("**Cluster Details**")
        st.markdown(
            "Each cluster below represents a group of papers on a similar "
            "subtopic. Keywords are extracted automatically from paper abstracts."
        )

        for _, row in summary.sort_values(
            "paper_count", ascending=False
        ).iterrows():
            if row["cluster_id"] == -1:
                continue
            cid = row["cluster_id"]
            kws = ", ".join(keywords.get(cid, [])[:4])
            years = row.get("years", [])
            year_str = (
                f"{min(years)} to {max(years)}"
                if len(years) > 1
                else (years[0] if years else "unknown")
            )
            coverage = {
                "Dense": "Well-studied",
                "Moderate": "Moderately studied",
                "Sparse": "Underexplored"
            }.get(row["density"], "Unknown")

            with st.expander(
                f"{row['label']} — {row['paper_count']} papers — {coverage}"
            ):
                st.markdown(f"**Keywords:** {kws}")
                st.markdown(f"**Year range:** {year_str}")
                st.markdown(f"**Coverage:** {coverage}")
                if row.get("sample_titles"):
                    st.markdown("**Sample papers in this cluster:**")
                    for t in row["sample_titles"][:3]:
                        st.markdown(f"- {t}")

    # ── TAB 2: Research Gaps ──────────────────────────────────────────────────
    with tab2:
        st.markdown(
            '<div class="section-header">Research Gaps</div>',
            unsafe_allow_html=True
        )
        st.markdown("""
<div class="info-box">
<b>What is a research gap?</b> A research gap is an angle of a topic that
has not been studied, or has been studied very little. Three types are detected:
<ul>
<li><b>Underexplored Subtopic</b> — a cluster exists but has only 1-2 papers.
The topic exists in the literature but almost nobody has worked on it.</li>
<li><b>Adjacent Gap</b> — a thin cluster sits right next to a well-studied
cluster. Researchers are publishing nearby but have not crossed into this
angle. Especially valuable because existing methodology can be reused.</li>
<li><b>Temporal Gap</b> — a subtopic had papers years ago but nothing recent.
Modern methods have not been applied to it. Lower risk to pursue because
prior work exists as a foundation.</li>
</ul>
Each gap includes what it means in plain English and a concrete recommendation.
</div>
""", unsafe_allow_html=True)

        if not gaps:
            st.success(
                "No significant research gaps were identified. "
                "The literature on this topic appears relatively complete "
                "based on the retrieved papers."
            )
        else:
            c1, c2, c3 = st.columns(3)
            high_gaps = [g for g in gaps if g["severity"] == "High"]
            medium_gaps = [g for g in gaps if g["severity"] == "Medium"]
            low_gaps = [g for g in gaps if g["severity"] == "Low"]
            c1.metric("High Priority Gaps", len(high_gaps))
            c2.metric("Medium Priority Gaps", len(medium_gaps))
            c3.metric("Low Priority Gaps", len(low_gaps))

            st.markdown("---")
            st.markdown(
                "Gaps are ordered from highest to lowest priority. "
                "Each gap shows what was found, what it means, "
                "and what you can do about it."
            )

            for i, gap in enumerate(gaps, 1):
                priority_label = {
                    "High": "HIGH PRIORITY",
                    "Medium": "MEDIUM PRIORITY",
                    "Low": "LOW PRIORITY"
                }.get(gap["severity"], "")

                with st.expander(
                    f"Gap {i}  |  {gap['gap_type']}  |  {priority_label}"
                ):
                    st.markdown("**What the analysis found:**")
                    st.write(gap["description"])

                    st.markdown("---")
                    col1, col2 = st.columns(2)

                    with col1:
                        st.markdown("**What this means for researchers:**")
                        st.info(
                            gap.get("what_it_means", "See description above.")
                        )

                    with col2:
                        st.markdown("**Recommended research action:**")
                        st.success(
                            gap.get("what_to_do", "See description above.")
                        )

                    if gap.get("keywords") and \
                            gap["keywords"] != ["single paper cluster"]:
                        st.markdown(
                            f"**Keywords:** "
                            f"{', '.join(gap['keywords'][:5])}"
                        )

                    if gap.get("sample_titles"):
                        st.markdown(
                            "**Existing papers in this cluster:**"
                        )
                        for t in gap["sample_titles"][:3]:
                            st.markdown(f"- {t}")

                    if gap["gap_type"] == "Adjacent Gap":
                        st.markdown(
                            f"**Why this is especially valuable:** "
                            f"Similarity score of "
                            f"{gap.get('similarity_to_dense', 'N/A')} "
                            f"means this gap is very close to an active "
                            f"research area. You can directly borrow "
                            f"methods from that neighbouring cluster."
                        )
                    elif gap["gap_type"] == "Temporal Gap":
                        st.markdown(
                            f"**Why this is especially valuable:** "
                            f"The last paper on this subtopic was from "
                            f"{gap.get('last_paper_year', 'unknown')} — "
                            f"{gap.get('years_since', '?')} years ago. "
                            f"Replication studies with modern methods are "
                            f"among the most reliably publishable research."
                        )

            st.markdown("---")
            with st.expander("Download plain-text gap summary"):
                st.text(gap_report)
                st.download_button(
                    "Download Gap Summary (.txt)",
                    gap_report,
                    file_name="gap_summary.txt",
                    mime="text/plain"
                )

    # ── TAB 3: Contradictions ─────────────────────────────────────────────────
    with tab3:
        st.markdown(
            '<div class="section-header">Contradictions in the Literature</div>',
            unsafe_allow_html=True
        )
        st.markdown("""
<div class="info-box">
<b>What is a contradiction?</b> Two papers are flagged as contradictory
when they address the same research question (measured by semantic
similarity of abstracts) but reach opposing conclusions — one reports
a positive or significant finding, the other reports a null or negative
finding. These represent unresolved debates where more research is needed.
<br><br>
<b>Strength levels:</b> Strong = high semantic similarity and clear
sentiment opposition. Moderate = one or both signals is partial.
Weak = flagged but requires manual review to confirm.
</div>
""", unsafe_allow_html=True)

        if not contradictions:
            st.info(
                "No contradictions were detected in the retrieved literature. "
                "The papers appear broadly consistent in their conclusions."
            )
        else:
            st.markdown(
                f"**{len(contradictions)} potential contradiction(s) detected** "
                f"across {len(df)} papers."
            )

            strong = [
                c for c in contradictions
                if c["contradiction_strength"] == "Strong"
            ]
            moderate = [
                c for c in contradictions
                if c["contradiction_strength"] == "Moderate"
            ]
            weak = [
                c for c in contradictions
                if c["contradiction_strength"] == "Weak"
            ]

            c1, c2, c3 = st.columns(3)
            c1.metric("Strong Contradictions", len(strong))
            c2.metric("Moderate Contradictions", len(moderate))
            c3.metric("Weak / Possible", len(weak))

            st.markdown("---")

            for i, c in enumerate(contradictions[:10], 1):
                with st.expander(
                    f"Contradiction {i}  |  {c['contradiction_strength']}  "
                    f"|  Semantic similarity: {c['similarity']}"
                ):
                    st.markdown(
                        "These two papers address the same research question "
                        "but reach different conclusions:"
                    )
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown("**Paper A**")
                        st.markdown(f"Year: {c['paper_a_year']}")
                        st.markdown(f"Source: {c['paper_a_source']}")
                        st.markdown(
                            f"Conclusion signal: **{c['sentiment_a']}**"
                        )
                        st.markdown(f"Title: {c['paper_a_title']}")
                        if c.get("paper_a_url"):
                            st.markdown(
                                f"[View paper]({c['paper_a_url']})"
                            )
                    with col2:
                        st.markdown("**Paper B**")
                        st.markdown(f"Year: {c['paper_b_year']}")
                        st.markdown(f"Source: {c['paper_b_source']}")
                        st.markdown(
                            f"Conclusion signal: **{c['sentiment_b']}**"
                        )
                        st.markdown(f"Title: {c['paper_b_title']}")
                        if c.get("paper_b_url"):
                            st.markdown(
                                f"[View paper]({c['paper_b_url']})"
                            )
                    st.caption(
                        f"Sentiment divergence: {c['sentiment_divergence']} "
                        f"| Sources: {c['paper_a_source']} "
                        f"vs {c['paper_b_source']}"
                    )

    # ── TAB 4: Demographics ───────────────────────────────────────────────────
    with tab4:
        st.markdown(
            '<div class="section-header">Demographic Representation Analysis</div>',
            unsafe_allow_html=True
        )
        st.markdown("""
<div class="info-box">
<b>What this analyses:</b> Research is not equally distributed across all
populations. Some groups — by geography, age, gender, or socioeconomic
status — are systematically underrepresented in the academic literature.
This section scans all retrieved abstracts for mentions of specific
population groups and flags where the literature has blind spots.
<br><br>
A Critical gap means zero papers mention that population.
A High gap means fewer than 5% of papers address that group.
</div>
""", unsafe_allow_html=True)

        geo_data = demo_results.get("geography", {})
        age_data = demo_results.get("age", {})

        col1, col2 = st.columns(2)

        with col1:
            if geo_data:
                geo_df = pd.DataFrame([
                    {
                        "Region": k,
                        "Papers Mentioning": v["count"],
                        "Percentage of Total": v["percentage"]
                    }
                    for k, v in geo_data.items()
                ])
                fig_geo = px.bar(
                    geo_df,
                    x="Region",
                    y="Percentage of Total",
                    title="Geographic Representation of Studies (%)",
                    color="Percentage of Total",
                    color_continuous_scale=[
                        "#e74c3c", "#f39c12", "#2ecc71"
                    ],
                    labels={"Percentage of Total": "% of papers"}
                )
                fig_geo.update_layout(
                    height=380,
                    xaxis_tickangle=-35,
                    plot_bgcolor="#fafafa"
                )
                st.plotly_chart(fig_geo, use_container_width=True)

        with col2:
            if age_data:
                age_df = pd.DataFrame([
                    {"Age Group": k, "Papers": v["count"]}
                    for k, v in age_data.items()
                    if v["count"] > 0
                ])
                if not age_df.empty:
                    fig_age = px.pie(
                        age_df,
                        names="Age Group",
                        values="Papers",
                        title="Age Group Representation",
                        color_discrete_sequence=px.colors.qualitative.Set2
                    )
                    fig_age.update_layout(height=380)
                    st.plotly_chart(fig_age, use_container_width=True)

        st.markdown("---")
        st.markdown("**Identified Representation Gaps**")

        if not demo_gaps:
            st.success("No major demographic representation gaps identified.")
        else:
            by_category = {}
            for gap in demo_gaps:
                by_category.setdefault(gap["category"], []).append(gap)

            for category, cat_gaps in by_category.items():
                st.markdown(f"**{category.title()}**")
                for gap in cat_gaps:
                    severity_prefix = {
                        "Critical": "CRITICAL",
                        "High": "HIGH",
                        "Medium": "MEDIUM"
                    }.get(gap["severity"], "")
                    st.markdown(
                        f"- [{severity_prefix}] {gap['description']}"
                    )
                st.markdown("")

        with st.expander("View full demographic report"):
            st.text(demo_report)

    # ── TAB 5: Citation Network ───────────────────────────────────────────────
    with tab5:
        st.markdown(
            '<div class="section-header">Semantic Citation Network</div>',
            unsafe_allow_html=True
        )
        st.markdown("""
<div class="info-box">
<b>What you are looking at:</b> Each circle is a paper. Two papers are
connected by a line if their abstracts are semantically similar — meaning
they likely address the same or adjacent research questions.
<br><br>
<b>Node size</b> reflects how many connections a paper has. Larger nodes
are central to the literature and represent the core of the field.
Smaller isolated nodes represent papers on underexplored angles.
<br><br>
<b>Colour</b> represents which subtopic cluster each paper belongs to.
Use the slider to adjust how strict the similarity threshold is.
</div>
""", unsafe_allow_html=True)

        threshold = st.slider(
            "Similarity threshold — lower shows more connections, "
            "higher shows only the strongest",
            min_value=0.3,
            max_value=0.9,
            value=0.5,
            step=0.05
        )

        if st.button("Rebuild Network with New Threshold"):
            G = build_similarity_network(
                df, embeddings, similarity_threshold=threshold
            )
            pos = compute_layout(G)
            fig_network = render_citation_network(G, pos, df, keywords)
            network_stats = get_network_stats(G)

        st.plotly_chart(fig_network, use_container_width=True)

        if network_stats:
            st.markdown("**Network Statistics**")
            n1, n2, n3, n4 = st.columns(4)
            n1.metric("Total Papers", network_stats["total_nodes"])
            n2.metric("Connections", network_stats["total_edges"])
            n3.metric(
                "Avg Connections per Paper",
                network_stats["avg_connections"]
            )
            n4.metric("Isolated Papers", network_stats["isolated_papers"])

            if network_stats["isolated_papers"] > 0:
                st.markdown("""
<div class="warning-box">
<b>Isolated papers detected:</b> Papers with no connections to others are
on angles of the topic that no other retrieved paper addresses. These are
strong candidates for research gap investigation.
</div>
""", unsafe_allow_html=True)

            if network_stats.get("top_connected"):
                st.markdown(
                    "**Most connected papers — core of the field:**"
                )
                for node_id, degree in network_stats["top_connected"]:
                    title = G.nodes[node_id]["title"]
                    st.markdown(f"- {title} ({degree} connections)")

    # ── TAB 6: Full Report ────────────────────────────────────────────────────
    with tab6:
        st.markdown(
            '<div class="section-header">Full Research Gap Report</div>',
            unsafe_allow_html=True
        )
        st.markdown("""
<div class="info-box">
This report synthesises all analysis outputs — gaps, contradictions,
demographic representation, and cluster landscape — into a single
structured document written in plain academic English. It is suitable
for including in a dissertation, grant application, or systematic review
as a starting point. Download it using the button below.
</div>
""", unsafe_allow_html=True)

        st.markdown(
            f"**Topic:** {topic.title()}  |  "
            f"**Papers analysed:** {len(df)}  |  "
            f"**Gaps identified:** {len(gaps)}  |  "
            f"**Contradictions:** {len(contradictions)}"
        )
        st.markdown("---")
        st.text(ai_report)
        st.markdown("---")
        st.download_button(
            "Download Full Report (.txt)",
            ai_report,
            file_name=(
                f"research_gap_report_"
                f"{topic[:40].replace(' ', '_')}.txt"
            ),
            mime="text/plain",
            use_container_width=True
        )

    # ── TAB 7: All Papers ─────────────────────────────────────────────────────
    with tab7:
        st.markdown(
            '<div class="section-header">All Retrieved Papers</div>',
            unsafe_allow_html=True
        )
        st.markdown("""
<div class="info-box">
All papers retrieved from academic databases for your query are listed
below. Use the source filter to view papers from specific databases.
The Cluster column shows which subtopic group each paper was assigned to.
Download the full dataset as a CSV file.
</div>
""", unsafe_allow_html=True)

        source_filter = st.multiselect(
            "Filter by database source",
            options=df["source"].unique().tolist(),
            default=df["source"].unique().tolist()
        )

        filtered_df = df[df["source"].isin(source_filter)].copy()

        if "fulltext_available" in filtered_df.columns:
            ft_count = int(filtered_df["fulltext_available"].sum())
            total_count = len(filtered_df)
            st.markdown(
                f"Full text retrieved for **{ft_count} of {total_count}** "
                f"papers ({round(ft_count / total_count * 100) if total_count > 0 else 0}%). "
                f"Remaining papers use abstract only."
            )
            if ft_count > 0:
                source_breakdown = filtered_df[
                    filtered_df["fulltext_available"] == True
                ]["fulltext_source"].value_counts().to_dict()
                breakdown_str = "  |  ".join(
                    f"{k}: {v}" for k, v in source_breakdown.items()
                )
                st.caption(f"Full text sources: {breakdown_str}")

        display_cols = [
            c for c in [
                "title", "year", "source", "cluster",
                "fulltext_available", "url"
            ]
            if c in filtered_df.columns
        ]
        display_df = filtered_df[display_cols].copy()
        display_df["cluster"] = display_df["cluster"].apply(
            lambda x: "Unclustered" if x == -1 else f"Cluster {x}"
        )

        st.markdown(f"Showing **{len(display_df)}** papers")
        st.dataframe(display_df, use_container_width=True, height=480)

        st.download_button(
            "Download All Papers as CSV",
            filtered_df.to_csv(index=False),
            file_name=f"papers_{topic[:30].replace(' ', '_')}.csv",
            mime="text/csv",
            use_container_width=True
        )
