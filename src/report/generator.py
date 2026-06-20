from datetime import datetime
import pandas as pd
import re


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
    Generates a clean, concise research gap report.
    Only includes the most useful sections: Executive Summary, Knowledge Landscape,
    Research Gaps with specific recommendations, and a 'Top Opportunities' section.
    Contradictions and Demographics are omitted to reduce noise.
    """
    date_str = datetime.now().strftime("%B %d, %Y")
    total = len(df)
    sources = df["source"].value_counts().to_dict()
    source_str = ", ".join(f"{v} from {k}" for k, v in sources.items())

    real_clusters = summary[summary["cluster_id"] != -1]
    dense = real_clusters[real_clusters["density"] == "Dense"]
    sparse = real_clusters[real_clusters["density"] == "Sparse"]

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
    lines += ["SECTION 1 - EXECUTIVE SUMMARY", "-" * 40]

    if len(real_clusters) == 0:
        lines.append(
            f"Analysis of {total} papers on '{topic}' revealed no clear subtopic "
            f"clustering. Try broadening your query or increasing the number of papers."
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
            f"A systematic analysis of {total} papers from {len(sources)} academic "
            f"databases identified {len(real_clusters)} distinct subtopic clusters "
            f"within the literature on '{topic}'. "
        )
        lines.append(
            f"Research activity is concentrated around {well_studied}, "
            f"while areas related to {underexplored} remain significantly underexplored. "
            f"A total of {len(gaps)} research gaps were identified."
        )
    lines.append("")

    # ── Section 2: Knowledge Landscape ───────────────────────────────────
    lines += ["SECTION 2 - KNOWLEDGE LANDSCAPE", "-" * 40]
    lines.append(
        f"The {total} retrieved papers cluster into {len(real_clusters)} distinct "
        f"subtopic areas. The distribution of research effort is as follows:"
    )
    lines.append("")

    for _, row in real_clusters.sort_values("paper_count", ascending=False).iterrows():
        cid = row["cluster_id"]
        kws = ", ".join(keywords.get(cid, ["unknown"])[:4])
        years = row.get("years", [])
        year_range = f"{min(years)}–{max(years)}" if years and len(years) > 1 else (years[0] if years else "unknown")

        density_label = {
            "Dense": "well-studied area with substantial literature",
            "Moderate": "moderately studied with room for further research",
            "Sparse": "critically underexplored with very limited coverage"
        }.get(row["density"], "unknown coverage")

        lines.append(
            f"  [{row['density'].upper()}] Cluster {cid} – {row['paper_count']} papers "
            f"({year_range})"
        )
        lines.append(f"  Keywords: {kws}")
        lines.append(f"  Assessment: This is a {density_label}.")
        if row.get("sample_titles"):
            lines.append(f"  Example: \"{row['sample_titles'][0][:80]}\"")
        lines.append("")

    # ── Section 3: Research Gaps ──────────────────────────────────────────
    lines += ["SECTION 3 - RESEARCH GAPS", "-" * 40]

    if not gaps:
        lines.append(
            "No significant research gaps were identified for this topic."
        )
    else:
        high = [g for g in gaps if g["severity"] == "High"]
        medium = [g for g in gaps if g["severity"] == "Medium"]
        lines.append(
            f"The analysis identified {len(gaps)} research gaps "
            f"({len(high)} high priority, {len(medium)} medium priority)."
        )
        lines.append("")

        for i, gap in enumerate(gaps, 1):
            lines.append(f"Gap {i} – {gap['gap_type']} [{gap['severity']} Priority]")
            lines.append("")
            lines.append(gap["description"])
            lines.append("")
            lines.append("What this means:")
            lines.append(gap["what_it_means"])
            lines.append("")
            lines.append("What you can do:")
            lines.append(gap["what_to_do"])
            lines.append("")

            if gap.get("paper_list"):
                lines.append("Existing papers in this cluster:")
                for paper in gap["paper_list"]:
                    # remove markdown links for plain text
                    clean = re.sub(r"\[Link\]\([^)]*\)", "", paper).strip()
                    lines.append(f"  {clean}")
            lines.append("")
            lines.append("-" * 40)
            lines.append("")

    # ── Section 4: Top 3 Opportunities ────────────────────────────────────
    lines += ["SECTION 4 - TOP 3 RESEARCH OPPORTUNITIES", "-" * 40]
    lines.append(
        "Based on the analysis, the following research directions are "
        "recommended as high‑value contributions:"
    )
    lines.append("")

    top_gaps = sorted(gaps, key=lambda g: 0 if g["severity"] == "High" else 1)[:3]
    if top_gaps:
        for i, gap in enumerate(top_gaps, 1):
            lines.append(f"{i}. **{gap['topic']}** ({gap['gap_type']})")
            lines.append(f"   • {gap['description']}")
            # extract first sentence of what_to_do
            action = gap["what_to_do"].split(". ")[0] + "."
            lines.append(f"   • Action: {action}")
            lines.append("")
    else:
        lines.append("No significant opportunities identified at this time.")

    # ── Section 5: Limitations ────────────────────────────────────────────
    lines += ["SECTION 5 - LIMITATIONS OF THIS ANALYSIS", "-" * 40]
    lines += [
        "This report was generated by automated analysis and has the following limitations:",
        "",
        "  1. Retrieval bias – only open‑access papers and those indexed by PubMed,",
        "     Semantic Scholar, and arXiv are included. Paywalled journals may contain",
        "     relevant work not captured here.",
        "",
        "  2. Abstract‑only analysis – embeddings and sentiment analysis are based on",
        "     abstracts only, not full paper text. Nuanced findings in full papers",
        "     may not be captured.",
        "",
        "  3. Cluster labels are algorithmic – cluster keywords are extracted by TF‑IDF",
        "     and may not perfectly represent the true subtopic.",
        "",
        "  4. This analysis should complement, not replace, expert literature review.",
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
