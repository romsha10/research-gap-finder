import requests
import time

BASE_URL = "https://api.semanticscholar.org/graph/v1"


def search_semantic_scholar(query: str, max_results: int = 100) -> list[dict]:
    """
    Searches Semantic Scholar. Handles 429s with exponential backoff.
    """

    papers = []
    offset = 0
    batch_size = min(100, max_results)

    while len(papers) < max_results:
        params = {
            "query": query,
            "limit": min(batch_size, max_results - len(papers)),
            "offset": offset,
            "fields": "title,abstract,authors,year,externalIds,citationCount,url"
        }

        # Retry up to 3 times with backoff on 429
        for attempt in range(3):
            response = requests.get(BASE_URL + "/paper/search", params=params)

            if response.status_code == 200:
                break
            elif response.status_code == 429:
                wait = 5 * (attempt + 1)
                print(f"Semantic Scholar rate limited. Waiting {wait}s...")
                time.sleep(wait)
            else:
                print(f"Semantic Scholar error: {response.status_code}")
                return papers
        else:
            print("Semantic Scholar: max retries hit, returning what we have")
            return papers

        data = response.json()
        batch = data.get("data", [])

        if not batch:
            break

        for paper in batch:
            abstract = paper.get("abstract") or ""
            if not abstract.strip():
                continue

            authors = [a.get("name", "") for a in paper.get("authors", [])]

            papers.append({
                "title": paper.get("title", "No title"),
                "abstract": abstract,
                "authors": authors,
                "year": str(paper.get("year") or "Unknown"),
                "citation_count": paper.get("citationCount", 0),
                "source": "Semantic Scholar",
                "url": paper.get("url", ""),
                "doi": paper.get("externalIds", {}).get("DOI", "")
            })

        offset += len(batch)
        time.sleep(2)  # be polite between pages

        if len(batch) < batch_size:
            break

    print(f"Retrieved {len(papers)} Semantic Scholar papers for: {query}")
    return papers
