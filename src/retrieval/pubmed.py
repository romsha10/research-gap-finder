import requests
import os
from dotenv import load_dotenv
import time

load_dotenv()

NCBI_API_KEY = os.getenv("NCBI_API_KEY")
BASE_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"


def search_pubmed(query: str, max_results: int = 100) -> list[dict]:
    """
    Given a research topic query, fetches papers from PubMed.
    Returns a list of dicts with title, abstract, authors, year, pmid.
    """

    # Step 1-search for paper IDs matching the query
    search_params = {
        "db": "pubmed",
        "term": query,
        "retmax": max_results,
        "retmode": "json",
        "api_key": NCBI_API_KEY
    }

    search_response = requests.get(
        BASE_URL + "esearch.fcgi", params=search_params)
    search_data = search_response.json()
    id_list = search_data.get("esearchresult", {}).get("idlist", [])

    if not id_list:
        print(f"No PubMed results found for: {query}")
        return []

    print(f"Found {len(id_list)} PubMed papers for: {query}")

    # Step 2-fetch full details for those IDs
    fetch_params = {
        "db": "pubmed",
        "id": ",".join(id_list),
        "retmode": "xml",
        "rettype": "abstract",
        "api_key": NCBI_API_KEY
    }

    fetch_response = requests.get(
        BASE_URL + "efetch.fcgi", params=fetch_params)

    # Step 3-parse XML response
    papers = parse_pubmed_xml(fetch_response.text, id_list)

    time.sleep(0.5)  # be polite to the API
    return papers


def parse_pubmed_xml(xml_text: str, id_list: list) -> list[dict]:
    """
    Parses the raw PubMed XML and extracts structured fields.
    """
    import xml.etree.ElementTree as ET

    papers = []

    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError:
        print("Failed to parse PubMed XML response")
        return []

    for article in root.findall(".//PubmedArticle"):
        try:
            # Title
            title_el = article.find(".//ArticleTitle")
            title = title_el.text if title_el is not None else "No title"

            # Abstract
            abstract_parts = article.findall(".//AbstractText")
            abstract = " ".join(
                (el.text or "") for el in abstract_parts
            ).strip()

            if not abstract:
                continue  # skip papers with no abstract-useless for embeddings

            # Authors
            authors = []
            for author in article.findall(".//Author"):
                last = author.find("LastName")
                fore = author.find("ForeName")
                if last is not None:
                    name = last.text
                    if fore is not None:
                        name += f", {fore.text}"
                    authors.append(name)

            # Year
            year_el = article.find(".//PubDate/Year")
            year = year_el.text if year_el is not None else "Unknown"

            # PMID
            pmid_el = article.find(".//PMID")
            pmid = pmid_el.text if pmid_el is not None else "Unknown"

            papers.append({
                "title": title,
                "abstract": abstract,
                "authors": authors,
                "year": year,
                "pmid": pmid,
                "source": "PubMed",
                "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"
            })

        except Exception as e:
            print(f"Skipped one article due to error: {e}")
            continue

    print(f"Successfully parsed {len(papers)} PubMed papers with abstracts")
    return papers
