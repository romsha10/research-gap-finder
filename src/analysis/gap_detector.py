import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import re
from collections import Counter


def _infer_topic_from_titles(titles: list[str], keywords: list[str]) -> str:
    if not titles:
        return ", ".join(keywords[:2]) if keywords else "unknown subtopic"

    cleaned = [t for t in titles if 15 < len(t) < 120]
    if not cleaned:
        return ", ".join(keywords[:2]) if keywords else "unknown subtopic"

    stopwords = {
        "the", "and", "for", "with", "from", "this", "that", "are",
        "was", "were", "have", "has", "been", "being", "study", "paper",
        "using", "based", "among", "their", "more", "than", "into",
        "about", "its", "not", "but", "can", "all", "our", "which"
    }

    all_words = []
    for title in cleaned:
        words = re.findall(r'\b[a-zA-Z]{4,}\b', title.lower())
        all_words.extend(w for w in words if w not in stopwords)

    if not all_words:
        return ", ".join(keywords[:2]) if keywords else "unknown subtopic"

    freq = Counter(all_words)
    top_words = [w for w, _ in freq.most_common(3)]
    return ", ".join(top_words)


def _summarise_cluster(cluster_papers: pd.DataFrame) -> str:
    titles = cluster_papers["title"].dropna().tolist()
    if not titles:
        return "papers on an unidentified subtopic"
    sorted_titles = sorted(titles, key=len)
    representative = sorted_titles[0] if sorted_titles else titles[0]
    return f'papers such as "{representative[:80]}"'


def detect_research_gaps(
    df: pd.DataFrame,
    embeddings: np.ndarray,
    cluster_summary: pd.DataFrame,
    cluster_keywords: dict
) -> list[dict]:

    gaps = []

    # ── Build centroids ───────────────────────────────────────────────────
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

    # ── Strategy 1: Underexplored subtopic ───────────────────────────────
    for _, row in real_clusters.iterrows():
        cid = row["cluster_id"]
        if cid not in sparse_clusters:
            continue

        keywords = cluster_keywords.get(cid, [])
        cluster_papers = df[df["cluster"] == cid]
        cluster_summary_text = _summarise_cluster(cluster_papers)

        years = []
        for y in cluster_papers["year"]:
            try:
                years.append(int(y))
            except (ValueError, TypeError):
                continue

        year_context = ""
        if years:
            year_context = (
                f" The {len(years)} existing paper(s) were published "
                f"between {min(years)} and {max(years)}."
            )

        kw_str = ", ".join(keywords[:3]) if keywords else "this subtopic"

        description = (
            f"The subtopic of '{kw_str}' has only {row['paper_count']} "
            f"paper(s) in the retrieved literature- meaning almost no "
            f"researchers have worked on this angle of the topic."
            f"{year_context} "
            f"This cluster contains {cluster_summary_text}. "
            f"A researcher who publishes work specifically on '{kw_str}' "
            f"within the context of this topic would be entering largely "
            f"uncharted territory with very little competition."
        )

        gaps.append({
            "gap_type": "Underexplored Subtopic",
            "cluster_id": cid,
            "severity": "High" if row["paper_count"] == 1 else "Medium",
            "paper_count": row["paper_count"],
            "keywords": keywords,
            "description": description,
            "what_it_means": (
                f"Only {row['paper_count']} paper(s) exist on '{kw_str}'. "
                f"The academic community has barely touched this angle."
            ),
            "what_to_do": (
                f"A systematic review or empirical study focused specifically "
                f"on '{kw_str}' in the context of this topic would directly "
                f"fill this gap and is likely publishable with low competition."
            ),
            "sample_titles": row["sample_titles"]
        })

    # ── Strategy 2: Adjacent gaps- one per sparse cluster ───────────────
    for sparse_cid in sparse_clusters:
        if sparse_cid not in cluster_centroids:
            continue

        sparse_centroid = cluster_centroids[sparse_cid].reshape(1, -1)

        # Find the single best dense neighbour only
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

        sparse_kws = cluster_keywords.get(sparse_cid, ["this subtopic"])
        dense_kws = cluster_keywords.get(
            best_dense_cid, ["the well-studied area"])

        sparse_papers = df[df["cluster"] == sparse_cid]
        dense_papers = df[df["cluster"] == best_dense_cid]

        sparse_titles = sparse_papers["title"].dropna().tolist()
        dense_titles = dense_papers["title"].dropna().tolist()

        sparse_topic = _infer_topic_from_titles(sparse_titles, sparse_kws)
        dense_topic = _infer_topic_from_titles(dense_titles, dense_kws)

        sparse_count = real_clusters[
            real_clusters["cluster_id"] == sparse_cid
        ]["paper_count"].values[0]

        dense_count = real_clusters[
            real_clusters["cluster_id"] == best_dense_cid
        ]["paper_count"].values[0]

        description = (
            f"The angle of '{sparse_topic}' has only {sparse_count} paper(s) "
            f"in the retrieved literature, yet it sits semantically close "
            f"(similarity score: {best_similarity:.2f}) to a well-studied area "
            f"on '{dense_topic}' which has {dense_count} papers. "
            f"Researchers are actively publishing on '{dense_topic}' but have "
            f"not crossed into '{sparse_topic}'. "
            f"This makes it a high-value gap- the methods, datasets, and "
            f"frameworks from '{dense_topic}' research can be directly adapted "
            f"to study '{sparse_topic}' without starting from scratch."
        )

        gaps.append({
            "gap_type": "Adjacent Gap",
            "cluster_id": sparse_cid,
            "adjacent_to_cluster": best_dense_cid,
            "severity": "High",
            "similarity_to_dense": round(best_similarity, 3),
            "keywords": sparse_kws,
            "sparse_topic": sparse_topic,
            "dense_topic": dense_topic,
            "description": description,
            "what_it_means": (
                f"'{dense_topic}' is well-studied ({dense_count} papers) but "
                f"'{sparse_topic}'- a closely related angle- has barely been "
                f"touched. This is a bridge waiting to be built."
            ),
            "what_to_do": (
                f"Design a study that takes the methodology from '{dense_topic}' "
                f"research and applies it specifically to '{sparse_topic}'. "
                f"This cross-pollination approach is one of the most reliable "
                f"ways to produce novel, publishable work."
            ),
            "sample_titles": real_clusters[
                real_clusters["cluster_id"] == sparse_cid
            ]["sample_titles"].values[0]
        })

    # ── Strategy 3: Temporal gaps ─────────────────────────────────────────
    for _, row in real_clusters.iterrows():
        cid = row["cluster_id"]
        cluster_papers = df[df["cluster"] == cid]

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
            keywords = cluster_keywords.get(cid, ["this subtopic"])
            years_since = current_year - max_year
            kw_str = ", ".join(keywords[:2])

            description = (
                f"The subtopic of '{kw_str}' has {row['paper_count']} "
                f"papers but the most recent is from {max_year}- "
                f"{years_since} years ago. "
                f"This means the existing work predates modern techniques "
                f"such as large language models, transformer architectures, "
                f"and the large datasets that became available after 2020. "
                f"The foundational questions in this cluster remain relevant "
                f"but nobody has revisited them with current tools."
            )

            gaps.append({
                "gap_type": "Temporal Gap",
                "cluster_id": cid,
                "severity": "Medium",
                "last_paper_year": max_year,
                "years_since": years_since,
                "paper_count": row["paper_count"],
                "keywords": keywords,
                "description": description,
                "what_it_means": (
                    f"This area was studied in {max_year} but has since been "
                    f"abandoned. Modern methods have not been applied to it."
                ),
                "what_to_do": (
                    f"Replicate and extend the {max_year} work using modern "
                    f"methods. A replication and extension paper is highly "
                    f"publishable and lower risk than entirely new research."
                ),
                "sample_titles": row["sample_titles"]
            })

    # Sort by severity
    severity_order = {"High": 0, "Medium": 1, "Low": 2}
    gaps.sort(key=lambda x: severity_order.get(x["severity"], 3))

    # Deduplicate- same cluster_id should not appear more than twice
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
        lines.append(
            f"GAP {i}- {gap['gap_type']} [{gap['severity']} Priority]"
        )
        lines.append(gap["description"])
        if gap.get("what_to_do"):
            lines.append(f"Recommended action: {gap['what_to_do']}")
        if gap.get("sample_titles"):
            lines.append("Papers in this cluster:")
            for title in gap["sample_titles"][:2]:
                lines.append(f"  - {title}")
        lines.append("")

    return "\n".join(lines)
