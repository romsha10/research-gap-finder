import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import re
from collections import Counter


def _extract_topic_from_titles(titles: list[str]) -> str:
    """
    Extract a human-readable topic from a list of paper titles.
    Uses the most common significant words to form a concise topic label.
    """
    if not titles:
        return "an underexplored area"
    
    # Combine all titles, remove common stopwords, pick most frequent meaningful words
    stopwords = {
        "the", "and", "for", "with", "from", "this", "that", "are",
        "was", "were", "have", "has", "been", "being", "study", "paper",
        "using", "based", "among", "their", "more", "than", "into",
        "about", "its", "not", "but", "can", "all", "our", "which",
        "analysis", "research", "investigation", "examination", "approach",
        "method", "model", "framework", "system", "approach", "technique"
    }
    
    all_words = []
    for title in titles:
        words = re.findall(r'\b[a-zA-Z]{4,}\b', title.lower())
        all_words.extend([w for w in words if w not in stopwords])
    
    if not all_words:
        # fallback: use the first title truncated
        return titles[0][:60] + ("..." if len(titles[0]) > 60 else "")
    
    freq = Counter(all_words)
    # Take top 3 most common words
    top_words = [w for w, _ in freq.most_common(3)]
    return " ".join(top_words)


def _format_paper_list(cluster_papers: pd.DataFrame) -> list[str]:
    """Format paper titles with years and links for display."""
    paper_lines = []
    for _, paper in cluster_papers.iterrows():
        title = paper.get("title", "Unknown title")
        year = paper.get("year", "?")
        url = paper.get("url", "")
        line = f"• {title} ({year})"
        if url:
            line += f" - [Link]({url})"
        paper_lines.append(line)
    return paper_lines


def detect_research_gaps(
    df: pd.DataFrame,
    embeddings: np.ndarray,
    cluster_summary: pd.DataFrame,
    cluster_keywords: dict
) -> list[dict]:
    """
    Detect research gaps from clusters and return a list of gap dicts
    with clear descriptions and specific research suggestions.
    """
    gaps = []

    # Build cluster centroids
    cluster_centroids = {}
    for cid in df["cluster"].unique():
        if cid == -1:
            continue
        mask = df["cluster"] == cid
        cluster_embeddings = embeddings[mask.values]
        cluster_centroids[cid] = cluster_embeddings.mean(axis=0)

    real_clusters = cluster_summary[cluster_summary["cluster_id"] != -1]

    # Define dense and sparse clusters
    dense_clusters = real_clusters[real_clusters["paper_count"] >= 5]["cluster_id"].tolist()
    sparse_clusters = real_clusters[real_clusters["paper_count"] <= 2]["cluster_id"].tolist()

    # ── Strategy 1: Underexplored Subtopic ───────────────────────────────
    for _, row in real_clusters.iterrows():
        cid = row["cluster_id"]
        if cid not in sparse_clusters:
            continue

        cluster_papers = df[df["cluster"] == cid]
        titles = cluster_papers["title"].dropna().tolist()
        if not titles:
            continue

        topic = _extract_topic_from_titles(titles)
        paper_count = len(cluster_papers)

        years = []
        for y in cluster_papers["year"]:
            try:
                years.append(int(y))
            except (ValueError, TypeError):
                continue

        year_context = ""
        if years:
            if len(years) == 1:
                year_context = f"The only paper was published in {years[0]}."
            else:
                year_context = f"The {len(years)} papers were published between {min(years)} and {max(years)}."

        paper_list = _format_paper_list(cluster_papers)

        description = (
            f"There are only **{paper_count} papers** on **'{topic}'** "
            f"in the entire retrieved literature. {year_context} "
            f"This area has been almost completely ignored by researchers."
        )

        what_means = (
            f"'{topic}' is a **significant research gap**. With only {paper_count} paper(s), "
            "there is virtually no existing work to compete with. "
            "The existing papers are listed below."
        )

        # Generate specific research question
        research_q = (
            f"**Suggested research question**: How can modern techniques (e.g., deep learning, "
            f"large language models) be applied to '{topic}'? or What are the key factors "
            f"affecting '{topic}' in the context of your broader topic?"
        )

        what_to_do = (
            f"**Recommended action**: Conduct a systematic review or empirical study "
            f"specifically on '{topic}'. {research_q}"
        )

        gaps.append({
            "gap_type": "Underexplored Subtopic",
            "cluster_id": cid,
            "severity": "High" if paper_count == 1 else "Medium",
            "paper_count": paper_count,
            "topic": topic,
            "description": description,
            "what_it_means": what_means,
            "what_to_do": what_to_do,
            "paper_list": paper_list,
            "titles": titles,
            "years": years
        })

    # ── Strategy 2: Adjacent gaps ─────────────────────────────────────────
    for sparse_cid in sparse_clusters:
        if sparse_cid not in cluster_centroids:
            continue

        sparse_centroid = cluster_centroids[sparse_cid].reshape(1, -1)

        best_similarity = 0
        best_dense_cid = None
        for dense_cid in dense_clusters:
            if dense_cid not in cluster_centroids:
                continue
            dense_centroid = cluster_centroids[dense_cid].reshape(1, -1)
            sim = float(cosine_similarity(sparse_centroid, dense_centroid)[0][0])
            if sim > best_similarity:
                best_similarity = sim
                best_dense_cid = dense_cid

        if best_dense_cid is None or best_similarity < 0.50:
            continue

        sparse_papers = df[df["cluster"] == sparse_cid]
        dense_papers = df[df["cluster"] == best_dense_cid]
        sparse_titles = sparse_papers["title"].dropna().tolist()
        dense_titles = dense_papers["title"].dropna().tolist()
        if not sparse_titles or not dense_titles:
            continue

        sparse_topic = _extract_topic_from_titles(sparse_titles)
        dense_topic = _extract_topic_from_titles(dense_titles)
        sparse_count = len(sparse_papers)
        dense_count = len(dense_papers)
        paper_list = _format_paper_list(sparse_papers)

        description = (
            f"**'{sparse_topic}'** has only **{sparse_count} papers**, "
            f"but it is semantically close to a well-studied area "
            f"**'{dense_topic}'** which has **{dense_count} papers** "
            f"(similarity: {best_similarity:.2f}). Researchers have extensively studied "
            f"'{dense_topic}' but have ignored '{sparse_topic}' – even though methods "
            f"from '{dense_topic}' could be directly adapted."
        )

        what_means = (
            f"'{sparse_topic}' is a **high‑value research gap** because it is adjacent "
            f"to an active research area. You can borrow existing methodology from "
            f"'{dense_topic}' research and apply it to '{sparse_topic}'."
        )

        research_q = (
            f"**Suggested research question**: Can methods from '{dense_topic}' be effectively "
            f"applied to '{sparse_topic}'? or What are the unique challenges of studying "
            f"'{sparse_topic}' compared to '{dense_topic}'?"
        )

        what_to_do = (
            f"**Recommended action**: Design a study that adapts the methodology "
            f"from '{dense_topic}' research and applies it to '{sparse_topic}'. {research_q}"
        )

        gaps.append({
            "gap_type": "Adjacent Gap",
            "cluster_id": sparse_cid,
            "adjacent_to_cluster": best_dense_cid,
            "severity": "High",
            "similarity_to_dense": round(best_similarity, 3),
            "topic": sparse_topic,
            "dense_topic": dense_topic,
            "paper_count": sparse_count,
            "description": description,
            "what_it_means": what_means,
            "what_to_do": what_to_do,
            "paper_list": paper_list,
            "titles": sparse_titles
        })

    # ── Strategy 3: Temporal gaps ─────────────────────────────────────────
    for _, row in real_clusters.iterrows():
        cid = row["cluster_id"]
        cluster_papers = df[df["cluster"] == cid]
        titles = cluster_papers["title"].dropna().tolist()
        if not titles:
            continue

        years = []
        for y in cluster_papers["year"]:
            try:
                years.append(int(y))
            except (ValueError, TypeError):
                continue

        if not years:
            continue

        max_year = max(years)
        current_year = 2025
        if max_year < 2020 and row["paper_count"] >= 2:
            years_since = current_year - max_year
            paper_count = row["paper_count"]
            topic = _extract_topic_from_titles(titles)
            paper_list = _format_paper_list(cluster_papers)

            description = (
                f"**'{topic}'** has **{paper_count} papers**, but the most recent is from "
                f"**{max_year}** – {years_since} years ago. This topic was studied in the past "
                f"but has since been abandoned. Modern techniques (AI, ML, larger datasets) "
                f"have never been applied to it."
            )

            what_means = (
                f"'{topic}' is a **temporal gap**. The foundational work exists (from {max_year}) "
                f"but nobody has revisited it with current tools. Replication studies with "
                "modern methods are among the most reliably publishable research."
            )

            research_q = (
                f"**Suggested research question**: How do modern AI/ML techniques compare to "
                f"the {max_year} approaches for '{topic}'? or What new insights can be gained "
                f"by applying current methods to '{topic}'?"
            )

            what_to_do = (
                f"**Recommended action**: Replicate and extend the {max_year} work on '{topic}' "
                f"using modern methods. {research_q}"
            )

            gaps.append({
                "gap_type": "Temporal Gap",
                "cluster_id": cid,
                "severity": "Medium",
                "last_paper_year": max_year,
                "years_since": years_since,
                "paper_count": paper_count,
                "topic": topic,
                "description": description,
                "what_it_means": what_means,
                "what_to_do": what_to_do,
                "paper_list": paper_list,
                "titles": titles,
                "years": years
            })

    # Sort by severity
    severity_order = {"High": 0, "Medium": 1, "Low": 2}
    gaps.sort(key=lambda x: severity_order.get(x["severity"], 3))

    # Remove duplicate cluster entries (keep only the first two types per cluster)
    seen = set()
    unique_gaps = []
    for gap in gaps:
        cid = gap["cluster_id"]
        if cid not in seen:
            seen.add(cid)
            unique_gaps.append(gap)
        else:
            # If we already have a gap for this cluster, skip additional ones
            continue

    print(f"Identified {len(unique_gaps)} research gaps")
    return unique_gaps


def format_gaps_as_text(gaps: list[dict], topic: str) -> str:
    """Generate a plain‑text version of the gaps for download."""
    if not gaps:
        return f"No significant research gaps identified for '{topic}'."

    lines = [
        f"RESEARCH GAP SUMMARY FOR: {topic.upper()}",
        "=" * 60,
        ""
    ]

    for i, gap in enumerate(gaps, 1):
        lines.append(f"GAP {i}: {gap['gap_type']} [{gap['severity']} Priority]")
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
                # Remove markdown link formatting for plain text
                clean = re.sub(r"\[Link\]\([^)]*\)", "", paper).strip()
                lines.append(f"  {clean}")
        lines.append("")
        lines.append("-" * 40)
        lines.append("")

    return "\n".join(lines)
