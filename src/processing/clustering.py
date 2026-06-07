import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import Counter
import re


def cluster_papers(
    embeddings: np.ndarray,
    df: pd.DataFrame,
    min_cluster_size: int = 2
) -> tuple[pd.DataFrame, np.ndarray]:

    df = df.copy()
    n_samples = len(embeddings)
    n_components = min(10, n_samples - 1)

    print(f"Reducing dimensions: 384 -> {n_components} via PCA...")
    pca = PCA(n_components=n_components, random_state=42)
    reduced = pca.fit_transform(embeddings)

    explained = pca.explained_variance_ratio_.sum()
    print(f"PCA retains {explained:.1%} of variance")

    print("Clustering papers by subtopic...")

    labels = None
    for eps in [2.0, 3.0, 5.0, 8.0]:
        clusterer = DBSCAN(
            eps=eps,
            min_samples=min_cluster_size,
            metric="euclidean",
            n_jobs=-1
        )
        candidate_labels = clusterer.fit_predict(reduced)
        n_clusters = len(set(candidate_labels)) - \
            (1 if -1 in candidate_labels else 0)

        if n_clusters >= 2:
            labels = candidate_labels
            print(f"Found {n_clusters} clusters at eps={eps}")
            break
        else:
            print(
                f"Only {n_clusters} cluster(s) at eps={eps}, trying larger...")

    if labels is None:
        print("DBSCAN found no clusters- using KMeans fallback")
        from sklearn.cluster import KMeans
        k = max(2, n_samples // 4)
        clusterer = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = clusterer.fit_predict(reduced)
        print(f"KMeans assigned {k} clusters")

    df["cluster"] = labels

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = int((labels == -1).sum())
    print(f"Final: {n_clusters} clusters, {n_noise} unclustered papers")

    return df, reduced


def describe_clusters(df: pd.DataFrame) -> pd.DataFrame:

    cluster_ids = sorted(df["cluster"].unique())
    summaries = []

    for cid in cluster_ids:
        cluster_papers = df[df["cluster"] == cid]
        label = "Unclustered" if cid == -1 else f"Cluster {cid}"

        years = []
        for y in cluster_papers["year"]:
            try:
                years.append(str(int(y)))
            except (ValueError, TypeError):
                continue

        summaries.append({
            "cluster_id": cid,
            "label": label,
            "paper_count": len(cluster_papers),
            "sources": cluster_papers["source"].value_counts().to_dict(),
            "years": sorted(set(years)),
            "sample_titles": cluster_papers["title"].dropna().head(3).tolist(),
            "density": (
                "Dense" if len(cluster_papers) >= 5 else
                "Moderate" if len(cluster_papers) >= 3 else
                "Sparse"
            )
        })

    return pd.DataFrame(summaries).sort_values(
        "paper_count", ascending=False
    ).reset_index(drop=True)


def get_cluster_keywords(df: pd.DataFrame, top_n: int = 5) -> dict:
    """
    Extracts keywords that are distinctive to each cluster versus all others.
    Uses TF-IDF across cluster documents so common words like 'health' and
    'students' that appear in every cluster get low scores.
    """

    cluster_keywords = {}
    cluster_ids = [c for c in df["cluster"].unique() if c != -1]

    # Words too generic to be meaningful as cluster labels
    extra_stopwords = {
        "paper", "study", "research", "results", "method", "approach",
        "proposed", "using", "based", "show", "shown", "used", "use",
        "also", "however", "model", "data", "analysis", "new", "task",
        "work", "propose", "present", "these", "this", "with", "that",
        "from", "their", "been", "were", "have", "abstract", "conclusion",
        "introduction", "background", "objective", "findings", "between",
        "among", "patients", "participants", "sample", "significant",
        "associated", "effect", "effects", "related", "including",
        "intervention", "group", "groups", "level", "levels", "high",
        "higher", "lower", "increased", "decreased", "reported", "found"
    }

    if len(cluster_ids) < 2:
        for cid in cluster_ids:
            cluster_papers = df[df["cluster"] == cid]
            titles = " ".join(cluster_papers["title"].fillna("").tolist())
            words = re.findall(r'\b[a-zA-Z]{5,}\b', titles.lower())
            freq = Counter(w for w in words if w not in extra_stopwords)
            cluster_keywords[cid] = [w for w, _ in freq.most_common(top_n)]
        return cluster_keywords

    # One document per cluster- all titles and abstracts combined
    cluster_docs = {}
    for cid in cluster_ids:
        cluster_papers = df[df["cluster"] == cid]
        combined = " ".join(
            str(row.get("title", "")) + " " + str(row.get("abstract", ""))
            for _, row in cluster_papers.iterrows()
        )
        cluster_docs[cid] = combined

    docs_list = [cluster_docs[cid] for cid in cluster_ids]

    try:
        vectorizer = TfidfVectorizer(
            max_features=3000,
            stop_words="english",
            ngram_range=(1, 2),
            min_df=1,
            sublinear_tf=True
        )
        tfidf_matrix = vectorizer.fit_transform(docs_list)
        feature_names = vectorizer.get_feature_names_out()

        for idx, cid in enumerate(cluster_ids):
            scores = tfidf_matrix[idx].toarray().flatten()
            top_indices = scores.argsort()[-top_n * 4:][::-1]

            keywords = []
            for i in top_indices:
                term = feature_names[i]
                term_words = term.split()
                if all(w not in extra_stopwords for w in term_words):
                    if len(term) > 3:
                        keywords.append(term)
                if len(keywords) >= top_n:
                    break

            if not keywords:
                # Fallback to title word frequency
                cluster_papers = df[df["cluster"] == cid]
                titles = " ".join(cluster_papers["title"].fillna("").tolist())
                words = re.findall(r'\b[a-zA-Z]{5,}\b', titles.lower())
                freq = Counter(
                    w for w in words if w not in extra_stopwords
                )
                keywords = [w for w, _ in freq.most_common(top_n)]

            cluster_keywords[cid] = keywords if keywords else ["general topic"]

    except Exception as e:
        print(f"Keyword extraction error: {e} - using fallback")
        for cid in cluster_ids:
            cluster_papers = df[df["cluster"] == cid]
            titles = " ".join(cluster_papers["title"].fillna("").tolist())
            words = re.findall(r'\b[a-zA-Z]{5,}\b', titles.lower())
            freq = Counter(w for w in words if w not in extra_stopwords)
            cluster_keywords[cid] = [w for w, _ in freq.most_common(top_n)]

    return cluster_keywords
