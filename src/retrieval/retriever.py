from src.retrieval.pubmed import search_pubmed
from src.retrieval.semantic_scholar import search_semantic_scholar
from src.retrieval.arxiv import search_arxiv
from src.retrieval.openalex import search_openalex, detect_field
from src.retrieval.cache import load_from_cache, save_to_cache
import pandas as pd

MEDICAL_FIELDS = {"medicine", "psychology", "biology", "chemistry"}
CS_FIELDS = {"computer science", "machine learning",
             "artificial intelligence", "engineering"}


def retrieve_all_papers(query: str, max_per_source: int = 50) -> pd.DataFrame:

    print(f"\nStarting retrieval for: '{query}'")

    # Check cache first
    cached = load_from_cache(query, max_per_source)
    if cached is not None:
        df = pd.DataFrame(cached)
        print(f"Served from cache: {len(df)} papers")
        return df

    detected = detect_field(query)
    print(f"Field detected: {detected or 'General'}")

    all_papers = []

    print("\nQuerying OpenAlex...")
    all_papers += search_openalex(query, max_per_source)

    print("\nQuerying Semantic Scholar...")
    all_papers += search_semantic_scholar(query, max_per_source)

    if detected in MEDICAL_FIELDS or detected is None:
        print("\nQuerying PubMed...")
        all_papers += search_pubmed(query, max_per_source)

    if detected not in MEDICAL_FIELDS or detected is None:
        print("\nQuerying arXiv...")
        all_papers += search_arxiv(query, max_per_source)

    if detected in MEDICAL_FIELDS:
        print("\nQuerying arXiv (medical AI subset)...")
        all_papers += search_arxiv(query, max_results=30)

    print(f"\nTotal raw papers: {len(all_papers)}")

    if not all_papers:
        return pd.DataFrame()

    df = pd.DataFrame(all_papers)
    df["title_normalised"] = (
        df["title"]
        .str.lower()
        .str.strip()
        .str.replace(r"[^\w\s]", "", regex=True)
    )
    df = df.drop_duplicates(subset="title_normalised")
    df = df.drop(columns=["title_normalised"])
    df = df.reset_index(drop=True)

    print(f"After deduplication: {len(df)} papers")

    # Save to cache for next time
    save_to_cache(query, max_per_source, df.to_dict(orient="records"))

    return df
