from datetime import datetime
import pandas as pd


def generate_report(
    topic: str,
    df,
    summary,
    keywords: dict,
    gaps: list,
    contradictions: list,
    demo_gaps: list
) -> str:
    """
    Generates a structured research gap report in plain English.
    No API needed-built from analysis outputs.
    """

    date_str = datetime.now().strftime("%B %d, %Y")
    total = len(df)
    sources = df["source"].value_counts().to_dict()
    source_str = ", ".join(f"{v} from {k}" for k, v in sources.items())

    real_clusters = summary[summary["cluster_id"] != -1]
    dense = real_clusters[real_clusters["density"] == "Dense"]
    sparse = real_clusters[real_clusters["density"] == "Sparse"]
    moderate = real_clusters[real_clusters["density"] == "Moderate"]

    high_gaps = [g for g in gaps if g["severity"] == "High"]
    medium_gaps = [g for g in gaps if g["severity"] == "Medium"]
    strong_contradictions = [
        c for c in contradictions if c["contradiction_strength"] in ["Strong", "Moderate"]]
    critical_demo = [d for d in demo_gaps if d["severity"]
                     in ["Critical", "High"]]

    lines = []

    # ── Header ────────────────────────────────────────────────────────────
    lines += [
        "=" * 70,
        "RESEARCH GAP ANALYSIS REPORT",
        "=" * 70,
        f"Topic:       {topic.title()}",
        f"Generated:   {date_str}",
        f"Papers:      {total} ({source_str})",
        f"Clusters:    {len(real_clusters)} subtopic clusters identified",
        "=" * 70,
        "",
    ]

    # ── Section 1: Executive Summary ─────────────────────────────────────
    lines += ["SECTION 1-EXECUTIVE SUMMARY", "-" * 40]

    if len(real_clusters) == 0:
        lines.append(
            f"Analysis of {total} papers on '{topic}' revealed no clear subtopic "
            f"clustering, suggesting the literature is either highly fragmented or "
            f"the topic requires a more specific query."
        )
    else:
        well_studied = ", ".join(
            keywords.get(row["cluster_id"], ["unknown"])[0]
            for _, row in dense.head(2).iterrows()
        ) if len(dense) > 0 else "none identified"

        underexplored = ", ".join(
            keywords.get(row["cluster_id"], ["unknown"])[0]
            for _, row in sparse.head(2).iterrows()
        ) if len(sparse) > 0 else "none identified"

        lines.append(
            f"A systematic analysis of {total} papers retrieved from {len(sources)} "
            f"academic databases identified {len(real_clusters)} distinct subtopic clusters "
            f"within the literature on '{topic}'. "
        )
        lines.append(
            f"Research activity is concentrated around {well_studied}, "
            f"while areas related to {underexplored} remain significantly underexplored. "
            f"A total of {len(gaps)} research gaps and {len(contradictions)} potential "
            f"contradictions were identified across the literature."
        )
    lines.append("")

    # ── Section 2: Knowledge Landscape ───────────────────────────────────
    lines += ["SECTION 2-KNOWLEDGE LANDSCAPE", "-" * 40]
    lines.append(
        f"The {total} retrieved papers cluster into {len(real_clusters)} distinct "
        f"subtopic areas. The distribution of research effort is as follows:"
    )
    lines.append("")

    for _, row in real_clusters.sort_values("paper_count", ascending=False).iterrows():
        cid = row["cluster_id"]
        kws = ", ".join(keywords.get(cid, ["unknown"])[:4])
        years = row.get("years", [])
        year_range = f"{min(years)}–{max(years)}" if years and len(
            years) > 1 else (years[0] if years else "unknown")

        density_comment = {
            "Dense": "well-studied area with substantial literature",
            "Moderate": "moderately studied with room for further research",
            "Sparse": "critically underexplored with very limited coverage"
        }.get(row["density"], "unknown coverage")

        lines.append(
            f"  [{row['density'].upper()}] Cluster {cid}-{row['paper_count']} papers "
            f"({year_range})"
        )
        lines.append(f"  Keywords: {kws}")
        lines.append(f"  Assessment: This is a {density_comment}.")
        if row.get("sample_titles"):
            lines.append(f"  Example: \"{row['sample_titles'][0][:80]}\"")
        lines.append("")

    # ── Section 3: Research Gaps ──────────────────────────────────────────
    lines += ["SECTION 3-PRIMARY RESEARCH GAPS", "-" * 40]

    if not gaps:
        lines.append("No significant research gaps were identified for this topic.")
    else:
        lines.append(f"The analysis identified {len(gaps)} research gaps:")
        lines.append("")

        for i, gap in enumerate(gaps, 1):
            lines.append(f"Gap {i} — {gap['gap_type']} [{gap['severity']} Priority]")
            lines.append("")
            lines.append(gap["description"])
            lines.append("")
            lines.append("What this means:")
            lines.append(gap["what_it_means"])
            lines.append("")
            lines.append("What you can do:")
            lines.append(gap["what_to_do"])
            lines.append("")
        
        if gap.get("titles"):
            lines.append("Papers in this gap cluster:")
            for title in gap["titles"][:5]:
                lines.append(f"  • {title}")
        lines.append("")
        lines.append("-" * 40)
        lines.append("")

    # ── Section 4: Contradictions ─────────────────────────────────────────
    lines += ["SECTION 4-CONTRADICTIONS AND UNRESOLVED DEBATES", "-" * 40]

    if not contradictions:
        lines.append(
            "No significant contradictions were detected. The retrieved literature "
            "appears broadly consistent in its findings and conclusions."
        )
    else:
        lines.append(
            f"The analysis identified {len(contradictions)} pairs of papers with "
            f"potentially contradictory findings ({len(strong_contradictions)} "
            f"moderate-to-strong contradictions). These represent unresolved debates "
            f"in the literature where replication studies or meta-analyses are needed:"
        )
        lines.append("")

        for i, c in enumerate(contradictions[:5], 1):
            lines.append(
                f"Contradiction {i} [{c['contradiction_strength']}] "
                f"(semantic similarity: {c['similarity']})"
            )
            lines.append(
                f"  Paper A ({c['paper_a_year']}, {c['sentiment_a']} conclusion):"
            )
            lines.append(f"    \"{c['paper_a_title'][:80]}\"")
            lines.append(
                f"  Paper B ({c['paper_b_year']}, {c['sentiment_b']} conclusion):"
            )
            lines.append(f"    \"{c['paper_b_title'][:80]}\"")
            lines.append(
                f"  These two papers address the same research question but reach "
                f"divergent conclusions, suggesting methodological differences, "
                f"population differences, or genuine scientific uncertainty."
            )
            lines.append("")

    # ── Section 5: Demographics ───────────────────────────────────────────
    lines += ["SECTION 5-DEMOGRAPHIC AND POPULATION GAPS", "-" * 40]

    if not demo_gaps:
        lines.append(
            "No major demographic representation gaps were identified. "
            "The retrieved literature appears to address a range of populations."
        )
    else:
        lines.append(
            f"The analysis identified {len(demo_gaps)} demographic representation "
            f"gaps. The following populations are systematically underrepresented "
            f"in the literature on '{topic}':"
        )
        lines.append("")

        by_category = {}
        for gap in demo_gaps:
            by_category.setdefault(gap["category"], []).append(gap)

        for category, cat_gaps in by_category.items():
            lines.append(f"  [{category.upper()}]")
            for gap in cat_gaps:
                lines.append(f"  • {gap['description']}")
                if gap["severity"] == "Critical":
                    lines.append(
                        f"    This is a critical gap-research on '{topic}' that "
                        f"excludes {gap['group']} populations cannot be considered "
                        f"globally generalisable."
                    )
            lines.append("")

    # ── Section 6: Recommended Research Directions ────────────────────────
    lines += ["SECTION 6-RECOMMENDED RESEARCH DIRECTIONS", "-" * 40]
    lines.append(
        "Based on the above analysis, the following research directions are "
        "recommended as high-value contributions to the field:"
    )
    lines.append("")

    rec_num = 1

    for gap in gaps[:3]:
        kws = gap.get("keywords", [])
        kw = kws[0] if kws else "this area"
        lines.append(
            f"  {rec_num}. Conduct a systematic review of {kw} within the context "
            f"of {topic}, specifically addressing the identified gap in "
            f"{gap['gap_type'].lower()} coverage."
        )
        rec_num += 1

    for demo in critical_demo[:2]:
        lines.append(
            f"  {rec_num}. Design a study specifically targeting {demo['group']} "
            f"populations to address the representational gap identified in the "
            f"{demo['category']} dimension."
        )
        rec_num += 1

    if contradictions:
        c = contradictions[0]
        lines.append(
            f"  {rec_num}. Conduct a pre-registered replication study to resolve "
            f"the contradiction between papers reaching opposing conclusions on "
            f"the same question, using a standardised methodology."
        )
        rec_num += 1

    lines.append("")

    # ── Section 7: Limitations ────────────────────────────────────────────
    lines += ["SECTION 7-LIMITATIONS OF THIS ANALYSIS", "-" * 40]
    lines += [
        "This report was generated by automated analysis and has the following limitations:",
        "",
        "  1. Retrieval bias-only open-access papers and those indexed by PubMed,",
        "     Semantic Scholar, and arXiv are included. Paywalled journals may contain",
        "     relevant work not captured here.",
        "",
        "  2. Abstract-only analysis-embeddings and sentiment analysis are based on",
        "     abstracts only, not full paper text. Nuanced findings in full papers",
        "     may not be captured.",
        "",
        "  3. Contradiction detection is approximate-sentiment analysis using keyword",
        "     matching is a proxy for true contradictions. Manual review of flagged",
        "     pairs is recommended before drawing conclusions.",
        "",
        "  4. Cluster labels are algorithmic-cluster keywords are extracted by TF-IDF",
        "     and may not perfectly represent the true subtopic.",
        "",
        "  5. This analysis should complement, not replace, expert literature review.",
        "",
    ]

    # ── Footer ────────────────────────────────────────────────────────────
    lines += [
        "=" * 70,
        "END OF REPORT",
        f"Generated by Research Gap Finder | {date_str}",
        "=" * 70,
    ]

    return "\n".join(lines)
