from src.retrieval.retriever import retrieve_all_papers
from src.processing.embeddings import generate_embeddings, find_similar_pairs
from src.processing.clustering import cluster_papers, describe_clusters, get_cluster_keywords
from src.analysis.gap_detector import detect_research_gaps, format_gaps_as_text
from src.processing.contradiction import detect_contradictions

if __name__ == "__main__":

    topic = "depression treatment adolescents"

    # 1-Retrieve
    df = retrieve_all_papers(topic, max_per_source=10)
    print(f"Retrieved {len(df)} papers\n")

    # 2-Embed
    df, embeddings = generate_embeddings(df)

    # 3-Cluster
    df, reduced = cluster_papers(embeddings, df, min_cluster_size=2)
    summary = describe_clusters(df)
    keywords = get_cluster_keywords(df)

    print("\n--- KNOWLEDGE LANDSCAPE ---")
    print(summary[["label", "paper_count", "density"]].to_string(index=False))

    print("\n--- CLUSTER TOPICS ---")
    for cid, kws in keywords.items():
        print(f"  Cluster {cid}: {', '.join(kws)}")

    # 4-Gap detection
    gaps = detect_research_gaps(df, embeddings, summary, keywords)
    report = format_gaps_as_text(gaps, topic)
    print("\n" + report)

    # 5-Contradiction detection
    contradictions = detect_contradictions(
        df, embeddings, similarity_threshold=0.55)
    print(f"\n--- CONTRADICTIONS ({len(contradictions)} found) ---")
    for c in contradictions[:3]:
        print(
            f"\n  [{c['contradiction_strength']}] Similarity: {c['similarity']}")
        print(f"  A ({c['sentiment_a']}): {c['paper_a_title'][:65]}")
        print(f"  B ({c['sentiment_b']}): {c['paper_b_title'][:65]}")
