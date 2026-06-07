import requests
import time

BASE_URL = "https://api.openalex.org"

# Field-specific concept IDs in OpenAlex
# These are used to bias results toward specific domains
FIELD_CONCEPTS = {
    "computer science": "C41008148",
    "machine learning": "C119857082",
    "artificial intelligence": "C154945302",
    "medicine": "C71924100",
    "biology": "C86803240",
    "physics": "C121332964",
    "mathematics": "C33923547",
    "economics": "C162324750",
    "psychology": "C15744967",
    "chemistry": "C185592680",
    "engineering": "C127413603",
    "social science": "C144024400",
    "law": "C203764120",
    "education": "C142362112",
}


def detect_field(query: str) -> str:
    """
    Detects the likely academic field from the query keywords.
    Returns a field name from FIELD_CONCEPTS or None.
    """
    query_lower = query.lower()

    field_keywords = {
        "computer science": [
            "algorithm", "software", "programming", "network", "database",
            "computing", "computer", "code", "system", "cyber", "cloud",
            "distributed", "operating system", "compiler"
        ],
        "machine learning": [
            "machine learning", "deep learning", "neural network", "nlp",
            "natural language", "computer vision", "transformer", "bert",
            "gpt", "llm", "reinforcement learning", "classification",
            "regression", "clustering", "embedding"
        ],
        "artificial intelligence": [
            "artificial intelligence", "ai ", " ai", "robot", "autonomous",
            "intelligent system", "knowledge graph", "reasoning"
        ],
        "physics": [
            "quantum", "particle", "relativity", "thermodynamic", "optics",
            "electromagnetic", "nuclear", "astrophysic", "cosmology"
        ],
        "mathematics": [
            "theorem", "proof", "topology", "algebra", "calculus",
            "differential equation", "number theory", "combinatorics"
        ],
        "economics": [
            "economic", "market", "gdp", "inflation", "trade", "fiscal",
            "monetary", "finance", "investment", "poverty", "inequality"
        ],
        "psychology": [
            "cognitive", "behavior", "mental", "emotion", "personality",
            "therapy", "psychiatric", "psychological", "trauma", "anxiety"
        ],
        "medicine": [
            "disease", "treatment", "clinical", "patient", "drug",
            "symptom", "diagnosis", "hospital", "surgery", "cancer",
            "diabetes", "depression", "vaccine", "medical"
        ],
        "engineering": [
            "engineering", "mechanical", "electrical", "civil", "structural",
            "manufacturing", "robotics", "sensor", "control system"
        ],
        "social science": [
            "society", "social", "culture", "political", "democracy",
            "policy", "governance", "community", "ethnicity", "gender"
        ],
        "law": [
            "legal", "law", "court", "justice", "regulation", "legislation",
            "constitutional", "criminal", "civil rights"
        ],
        "education": [
            "education", "learning", "teaching", "student", "school",
            "university", "curriculum", "pedagogy", "literacy"
        ]
    }

    scores = {}
    for field, kws in field_keywords.items():
        scores[field] = sum(1 for kw in kws if kw in query_lower)

    best_field = max(scores, key=scores.get)
    return best_field if scores[best_field] > 0 else None


def search_openalex(query: str, max_results: int = 100) -> list[dict]:
    """
    Searches OpenAlex-covers ALL academic fields.
    250 million papers, completely free, no API key needed.
    Automatically detects field and filters accordingly.
    """

    papers = []
    page = 1
    per_page = min(50, max_results)

    detected_field = detect_field(query)
    if detected_field:
        print(f"Detected field: {detected_field}-applying field filter")

    while len(papers) < max_results:
        params = {
            "search": query,
            "per-page": per_page,
            "page": page,
            "filter": "has_abstract:true",
            "select": "id,title,abstract_inverted_index,authorships,publication_year,doi,primary_location,concepts,cited_by_count",
            "mailto": "research-gap-finder@example.com"  # polite API usage
        }

        # Add field filter if detected
        if detected_field and detected_field in FIELD_CONCEPTS:
            concept_id = FIELD_CONCEPTS[detected_field]
            params["filter"] = f"has_abstract:true,concepts.id:{concept_id}"

        try:
            response = requests.get(
                BASE_URL + "/works",
                params=params,
                timeout=15
            )

            if response.status_code != 200:
                print(f"OpenAlex error: {response.status_code}")
                break

            data = response.json()
            results = data.get("results", [])

            if not results:
                break

            for work in results:
                # OpenAlex stores abstracts as inverted index-reconstruct it
                abstract = reconstruct_abstract(
                    work.get("abstract_inverted_index", {})
                )

                if not abstract or len(abstract) < 50:
                    continue

                # Authors
                authors = []
                for auth in work.get("authorships", []):
                    name = auth.get("author", {}).get("display_name", "")
                    if name:
                        authors.append(name)

                # URL
                doi = work.get("doi", "")
                url = f"https://doi.org/{doi}" if doi else work.get("id", "")

                # Source/journal
                location = work.get("primary_location") or {}
                source = location.get("source") or {}
                journal = source.get("display_name", "Unknown journal")

                papers.append({
                    "title": work.get("title", "No title"),
                    "abstract": abstract,
                    "authors": authors,
                    "year": str(work.get("publication_year") or "Unknown"),
                    "citation_count": work.get("cited_by_count", 0),
                    "source": "OpenAlex",
                    "journal": journal,
                    "url": url,
                    "doi": doi,
                    "field": detected_field or "General"
                })

            page += 1
            time.sleep(0.5)

            if len(results) < per_page:
                break

        except Exception as e:
            print(f"OpenAlex error: {e}")
            break

    print(f"Retrieved {len(papers)} OpenAlex papers for: {query}")
    return papers[:max_results]


def reconstruct_abstract(inverted_index: dict) -> str:
    """
    OpenAlex stores abstracts as an inverted index:
    {"word": [position1, position2], ...}

    We reconstruct the original sentence by placing words at their positions.
    """
    if not inverted_index:
        return ""

    try:
        position_word = {}
        for word, positions in inverted_index.items():
            for pos in positions:
                position_word[pos] = word

        if not position_word:
            return ""

        max_pos = max(position_word.keys())
        words = [position_word.get(i, "") for i in range(max_pos + 1)]
        return " ".join(w for w in words if w).strip()

    except Exception:
        return ""
