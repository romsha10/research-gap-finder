import arxiv
import time


def search_arxiv(query: str, max_results: int = 100) -> list[dict]:
    """
    Searches arXiv for papers matching the query.
    """

    # Longer delay helps avoid 429s
    client = arxiv.Client(
        page_size=50,
        delay_seconds=3,
        num_retries=5
    )

    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.Relevance
    )

    papers = []

    try:
        for result in client.results(search):
            abstract = result.summary.strip()
            if not abstract:
                continue

            papers.append({
                "title": result.title,
                "abstract": abstract,
                "authors": [str(a) for a in result.authors],
                "year": str(result.published.year),
                "source": "arXiv",
                "url": result.entry_id,
                "arxiv_id": result.entry_id.split("/")[-1],
                "categories": result.categories
            })

    except Exception as e:
        print(f"arXiv error (returning what we have): {e}")

    print(f"Retrieved {len(papers)} arXiv papers for: {query}")
    return papers
