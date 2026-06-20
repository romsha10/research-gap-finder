import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import re
from collections import Counter


def _extract_topic_from_titles(titles: list[str]) -> str:
    """Extracts a human-readable topic from paper titles."""
    if not titles:
        return "an underexplored area"
    
    # Remove common words and join to form a phrase
    stopwords = {
        "the", "and", "for", "with", "from", "this", "that", "are",
        "was", "were", "have", "has", "been", "being", "study", "paper",
        "using", "based", "among", "their", "more", "than", "into",
        "about", "its", "not", "but", "can", "all", "our", "which",
        "analysis", "research", "investigation", "examination"
    }
    
    # Get first 3-4 significant words from the first title
    first_title = titles[0]
    words = re.findall(r'\b[a-zA-Z]{4,}\b', first_title)
    meaningful = [w for w in words if w.lower() not in stopwords]
    
    if len(meaningful) >= 2:
        return " ".join(meaningful[:3])
    else:
        # Fallback: use entire title
        return first_title[:60] + ("..." if len(first_title) > 60 else "")


def detect_research_gaps(
    df: pd.DataFrame,
    embeddings: np.ndarray,
    cluster_summary: pd.DataFrame,
    cluster_keywords: dict
) -> list[dict]:

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
    
    dense_clusters = real_clusters[
        real_clusters["paper_count"] >= 5
    ]["cluster_id"].tolist()
    
    sparse_clusters = real_clusters[
        real_clusters["paper_count"] <= 2
    ]["cluster_id"].tolist()

    # ── Strategy 1: Underexplored Subtopic ───────────────────────────────
    for _, row in real_clusters.iterrows():
        cid = row["cluster_id"]
        if cid not in sparse_clusters:
            continue

        cluster_papers = df[df["cluster"] == cid]
        titles = cluster_papers["title"].dropna().tolist()
        
        topic_name = _extract_topic_from_titles(titles)
        
        years = []
        for y in cluster_papers["year"]:
            try:
                years.append(int(y))
            except (ValueError, TypeError):
                continue

        paper_count = row["paper_count"]
        
        # Build paper list with links
        paper_list = []
        for _, paper in cluster_papers.iterrows():
            title = paper.get("title", "Unknown title")
            url = paper.get("url", "")
            year = paper.get("year", "?")
            paper_list.append(f"• {title} ({year})" + (f" - [Link]({url})" if url else ""))

        year_context = ""
        if years:
            if len(years) == 1:
                year_context = f"The only paper was published in {years[0]}."
            else:
                year_context = f"The {len(years)} papers were published between {min(years)} and {max(years)}."

        description = (
            f"There are only **{paper_count} papers** on **'{topic_name}'** "
            f"in the entire retrieved literature. "
            f"{year_context} "
            f"This area has been almost completely ignored by researchers."
        )

        what_it_means = (
            f"'{topic_name}' is a significant research gap. "
            f"With only {paper_count} paper(s), there is virtually no existing work to compete with. "
            f"The existing papers are:"
        )

        what_to_do = (
            f"**Recommended research**: Conduct a systematic review or empirical study "
            f"specifically on '{topic_name}'. "
            f"Possible research question: 'How does {topic_name} relate to your broader topic?' "
            f"or 'What are the key factors affecting {topic_name}?'"
        )

        gaps.append({
            "gap_type": "Underexplored Subtopic",
            "cluster_id": cid,
            "severity": "High" if paper_count == 1 else "Medium",
            "paper_count": paper_count,
            "topic_name": topic_name,
            "description": description,
            "what_it_means": what_it_means,
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
        
        # Find best dense neighbour
        best_similarity = 0
        best_dense_cid = None

        for dense_cid in dense_clusters:
            if dense_cid not in cluster_centroids:
                continue
            dense_centroid = cluster_centroids[dense_cid].reshape(1, -1)
            similarity = float(
                cosine_similarity(sparse_centroid, dense_centroid)[0][0]
            )
            if similarity > best_similarity:
                best_similarity = similarity
                best_dense_cid = dense_cid

        if best_dense_cid is None or best_similarity < 0.50:
            continue

        sparse_papers = df[df["cluster"] == sparse_cid]
        dense_papers = df[df["cluster"] == best_dense_cid]
        
        sparse_titles = sparse_papers["title"].dropna().tolist()
        dense_titles = dense_papers["title"].dropna().tolist()
        
        sparse_topic = _extract_topic_from_titles(sparse_titles)
        dense_topic = _extract_topic_from_titles(dense_titles)
        
        sparse_count = len(sparse_papers)
        dense_count = len(dense_papers)
        
        # Build paper list with links
        paper_list = []
        for _, paper in sparse_papers.iterrows():
            title = paper.get("title", "Unknown title")
            url = paper.get("url", "")
            year = paper.get("year", "?")
            paper_list.append(f"• {title} ({year})" + (f" - [Link]({url})" if url else ""))

        description = (
            f"**'{sparse_topic}'** has only **{sparse_count} papers**, "
            f"but it is semantically close to a well-studied area "
            f"**'{dense_topic}'** which has **{dense_count} papers** "
            f"(similarity score: {best_similarity:.2f}). "
            f"Researchers have extensively studied '{dense_topic}' but have ignored "
            f"'{sparse_topic}' – even though the methods and frameworks could be directly adapted."
        )

        what_it_means = (
            f"'{sparse_topic}' is a **high-value research gap** because it's adjacent "
            f"to an active research area. You can borrow existing methodology from "
            f"'{dense_topic}' research and apply it to '{sparse_topic}'."
        )

        what_to_do = (
            f"**Recommended research**: Design a study that adapts the methodology "
            f"from '{dense_topic}' research and applies it to '{sparse_topic}'. "
            f"Possible research question: 'Can methods from {dense_topic} be effectively "
            f"applied to {sparse_topic}?' or 'What are the unique challenges of "
            f"studying {sparse_topic} compared to {dense_topic}?'"
        )

        gaps.append({
            "gap_type": "Adjacent Gap",
            "cluster_id": sparse_cid,
            "adjacent_to_cluster": best_dense_cid,
            "severity": "High",
            "similarity_to_dense": round(best_similarity, 3),
            "topic_name": sparse_topic,
            "dense_topic": dense_topic,
            "paper_count": sparse_count,
            "description": description,
            "what_it_means": what_it_means,
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
            
        topic_name = _extract_topic_from_titles(titles)
        
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
            
            # Build paper list with links
            paper_list = []
            for _, paper in cluster_papers.iterrows():
                title = paper.get("title", "Unknown title")
                url = paper.get("url", "")
                year = paper.get("year", "?")
                paper_list.append(f"• {title} ({year})" + (f" - [Link]({url})" if url else ""))

            description = (
                f"**'{topic_name}'** has **{paper_count} papers**, "
                f"but the most recent is from **{max_year}** – {years_since} years ago. "
                f"This topic was studied in the past but has since been abandoned. "
                f"Modern techniques (AI, ML, larger datasets) have never been applied to it."
            )

            what_it_means = (
                f"'{topic_name}' is a **temporal gap**. The foundational work exists "
                f"(from {max_year}) but nobody has revisited it with current tools. "
                f"Replication studies with modern methods are among the most reliably "
                f"publishable research."
            )

            what_to_do = (
                f"**Recommended research**: Replicate and extend the {max_year} work "
                f"on '{topic_name}' using modern methods. "
                f"Possible research question: 'How do modern AI/ML techniques compare "
                f"to the {max_year} approaches for {topic_name}?' or 'What new insights "
                f"can be gained by applying current methods to {topic_name}?'"
            )

            gaps.append({
                "gap_type": "Temporal Gap",
                "cluster_id": cid,
                "severity": "Medium",
                "last_paper_year": max_year,
                "years_since": years_since,
                "paper_count": paper_count,
                "topic_name": topic_name,
                "description": description,
                "what_it_means": what_it_means,
                "what_to_do": what_to_do,
                "paper_list": paper_list,
                "titles": titles,
                "years": years
            })

    # Sort by severity
    severity_order = {"High": 0, "Medium": 1, "Low": 2}
    gaps.sort(key=lambda x: severity_order.get(x["severity"], 3))

    # Deduplicate
    seen_clusters = Counter()
    deduped = []
    for gap in gaps:
        cid = gap["cluster_id"]
        if seen_clusters[cid] < 2:
            deduped.append(gap)
            seen_clusters[cid] += 1

    print(f"Identified {len(deduped)} research gaps")
    return deduped


def format_gaps_as_text(gaps: list[dict], topic: str) -> str:
    if not gaps:
        return f"No significant research gaps identified for '{topic}'."

    lines = [
        f"RESEARCH GAP SUMMARY FOR: {topic.upper()}",
        "=" * 60, ""
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
            lines.append("Existing papers in this gap cluster:")
            for paper in gap["paper_list"][:5]:  # limit to 5 in text report
                lines.append(f"  {paper}")
        lines.append("")
        lines.append("-" * 40)
        lines.append("")

    return "\n".join(lines)
