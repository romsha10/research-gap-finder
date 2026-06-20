from src.retrieval.pubmed import search_pubmed
from src.retrieval.semantic_scholar import search_semantic_scholar
from src.retrieval.arxiv import search_arxiv
from src.retrieval.openalex import search_openalex, detect_field
from src.retrieval.cache import load_from_cache, save_to_cache
import pandas as pd

MEDICAL_FIELDS = {"medicine", "psychology", "biology", "chemistry"}
CS_FIELDS = {
    "computer science", "machine learning",
    "artificial intelligence", "engineering"
}


def retrieve_all_papers(
    query: str,
    max_per_source: int = 50,
    fetch_fulltext: bool = False,
    sources: list = None
) -> pd.DataFrame:

    print(f"\nStarting retrieval for: '{query}'")

    # If no sources specified, use all
    if not sources:
        sources = ["PubMed", "OpenAlex", "Semantic Scholar", "arXiv"]

    # Cache key includes the sources list (sorted to ensure consistency)
    cache_key = f"{query}_{max_per_source}_{'ft' if fetch_fulltext else 'abs'}_{'_'.join(sorted(sources))}"
    cached = load_from_cache(cache_key, max_per_source)
    if cached is not None:
        df = pd.DataFrame(cached)
        print(f"Served from cache: {len(df)} papers")
        return df

    detected = detect_field(query)
    print(f"Field detected: {detected or 'General'}")

    all_papers = []

    # Query only selected sources
    if "OpenAlex" in sources:
        print("\nQuerying OpenAlex...")
        all_papers += search_openalex(query, max_per_source)

    if "Semantic Scholar" in sources:
        print("\nQuerying Semantic Scholar...")
        all_papers += search_semantic_scholar(query, max_per_source)

    if "PubMed" in sources and (detected in MEDICAL_FIELDS or detected is None):
        print("\nQuerying PubMed...")
        all_papers += search_pubmed(query, max_per_source)

    if "arXiv" in sources and (detected not in MEDICAL_FIELDS or detected is None):
        print("\nQuerying arXiv...")
        all_papers += search_arxiv(query, max_per_source)

    # Optional extra for medical + arXiv
    if "arXiv" in sources and detected in MEDICAL_FIELDS:
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
    print(f"Sources: {df['source'].value_counts().to_dict()}")

    # Full text enrichment — optional, takes extra time
    if fetch_fulltext:
        from src.retrieval.fulltext import enrich_with_fulltext
        print("\nFetching full text where available...")
        papers_list = df.to_dict(orient="records")
        enriched = enrich_with_fulltext(papers_list)
        df = pd.DataFrame(enriched)

    # Save to cache (include sources in cache key)
    save_to_cache(cache_key, max_per_source, df.to_dict(orient="records"))

    return df
