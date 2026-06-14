import requests
import time
import re
import io
from typing import Optional


# ── Unpaywall ─────────────────────────────────────────────────────────────────

def get_open_access_url(doi: str) -> Optional[str]:
    """
    Queries Unpaywall to find a free legal PDF URL for a given DOI.
    Unpaywall is completely free, no API key needed.
    Returns the PDF URL if found, None otherwise.
    """
    if not doi:
        return None

    doi = doi.strip().lstrip("https://doi.org/").lstrip("http://doi.org/")
    if not doi:
        return None

    try:
        url = f"https://api.unpaywall.org/v2/{doi}"
        params = {"email": "research-gap-finder@example.com"}
        response = requests.get(url, params=params, timeout=10)

        if response.status_code != 200:
            return None

        data = response.json()

        if not data.get("is_oa"):
            return None

        best_location = data.get("best_oa_location")
        if not best_location:
            return None

        pdf_url = best_location.get("url_for_pdf")
        if pdf_url:
            return pdf_url

        # Sometimes only a landing page is available - skip those
        return None

    except Exception:
        return None


# ── PDF Text Extraction ───────────────────────────────────────────────────────

def extract_text_from_pdf_url(pdf_url: str) -> Optional[str]:
    """
    Downloads a PDF from a URL and extracts clean body text.
    Tries pdfplumber first, falls back to pymupdf.
    Returns None if extraction fails.
    """
    try:
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (compatible; ResearchGapFinder/1.0; "
                "mailto:research-gap-finder@example.com)"
            )
        }
        response = requests.get(pdf_url, headers=headers, timeout=20)

        if response.status_code != 200:
            return None

        if "pdf" not in response.headers.get("content-type", "").lower():
            return None

        pdf_bytes = io.BytesIO(response.content)

        # Try pdfplumber first - better at structured academic PDFs
        try:
            import pdfplumber
            text_parts = []
            with pdfplumber.open(pdf_bytes) as pdf:
                for page in pdf.pages[:20]:  # cap at 20 pages
                    text = page.extract_text()
                    if text:
                        text_parts.append(text)
            full_text = "\n".join(text_parts)
            if len(full_text.strip()) > 500:
                return clean_extracted_text(full_text)
        except Exception:
            pass

        # Fallback to pymupdf
        try:
            import fitz
            pdf_bytes.seek(0)
            doc = fitz.open(stream=pdf_bytes.read(), filetype="pdf")
            text_parts = []
            for page_num in range(min(20, len(doc))):
                text_parts.append(doc[page_num].get_text())
            full_text = "\n".join(text_parts)
            if len(full_text.strip()) > 500:
                return clean_extracted_text(full_text)
        except Exception:
            pass

        return None

    except Exception:
        return None


def clean_extracted_text(text: str) -> str:
    """
    Cleans raw PDF text by removing common academic paper noise.
    Keeps the body text - removes headers, footers, references section.
    """
    lines = text.split("\n")
    cleaned_lines = []

    # Patterns that indicate non-body content to skip
    skip_patterns = [
        r"^\s*\d+\s*$",                          # page numbers
        r"^doi\s*:",                              # DOI lines
        r"^\s*©",                                 # copyright notices
        r"^(received|accepted|published)\s*:",   # submission dates
        r"^\s*figure\s+\d+",                     # figure captions
        r"^\s*table\s+\d+",                      # table captions
        r"^\s*references\s*$",                   # references header
        r"^\s*acknowledgement",                  # acknowledgements
        r"^\s*funding\s*$",                      # funding section
    ]

    in_references = False

    for line in lines:
        line_lower = line.lower().strip()

        # Stop at references section - not useful for gap analysis
        if re.match(r"^\s*references\s*$", line_lower):
            in_references = True

        if in_references:
            continue

        # Skip lines matching noise patterns
        if any(re.match(p, line_lower) for p in skip_patterns):
            continue

        # Skip very short lines (likely headers/footers)
        if len(line.strip()) < 20:
            continue

        cleaned_lines.append(line.strip())

    cleaned = " ".join(cleaned_lines)

    # Collapse multiple spaces
    cleaned = re.sub(r"\s+", " ", cleaned).strip()

    return cleaned


# ── PubMed Central Full Text ──────────────────────────────────────────────────

def get_pmc_fulltext(pmid: str) -> Optional[str]:
    """
    Fetches full text from PubMed Central for papers that are in PMC.
    Uses the NCBI efetch API - same one we already use, no extra key.
    About 40% of PubMed papers have a free PMC full text version.
    """
    if not pmid:
        return None

    try:
        # First check if this PMID has a PMC version
        search_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/elink.fcgi"
        params = {
            "dbfrom": "pubmed",
            "db": "pmc",
            "id": pmid,
            "retmode": "json"
        }
        response = requests.get(search_url, params=params, timeout=10)
        if response.status_code != 200:
            return None

        data = response.json()
        linksets = data.get("linksets", [])
        if not linksets:
            return None

        pmc_ids = []
        for linkset in linksets:
            for linksetdb in linkset.get("linksetdbs", []):
                if linksetdb.get("dbto") == "pmc":
                    pmc_ids = linksetdb.get("links", [])
                    break

        if not pmc_ids:
            return None

        pmc_id = pmc_ids[0]

        # Fetch full text XML from PMC
        fetch_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
        fetch_params = {
            "db": "pmc",
            "id": pmc_id,
            "rettype": "full",
            "retmode": "xml"
        }
        fetch_response = requests.get(fetch_url, params=fetch_params, timeout=15)
        if fetch_response.status_code != 200:
            return None

        return extract_text_from_pmc_xml(fetch_response.text)

    except Exception:
        return None


def extract_text_from_pmc_xml(xml_text: str) -> Optional[str]:
    """
    Extracts body text from PMC XML format.
    Focuses on abstract, introduction, methods, results, discussion.
    Skips references and acknowledgements.
    """
    import xml.etree.ElementTree as ET

    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError:
        return None

    text_parts = []

    # Sections we want
    target_sections = {
        "abstract", "introduction", "background",
        "methods", "methodology", "results",
        "discussion", "conclusion", "conclusions"
    }

    # Extract abstract
    for abstract in root.findall(".//abstract"):
        for elem in abstract.iter():
            if elem.text:
                text_parts.append(elem.text.strip())

    # Extract body sections
    for sec in root.findall(".//sec"):
        title_elem = sec.find("title")
        section_title = ""
        if title_elem is not None and title_elem.text:
            section_title = title_elem.text.lower().strip()

        # Skip references and acknowledgements
        if any(skip in section_title for skip in [
            "reference", "acknowledgement", "funding",
            "conflict", "supplement", "appendix"
        ]):
            continue

        # Include if it is a target section or has no title (body text)
        if not section_title or any(
            target in section_title for target in target_sections
        ):
            for elem in sec.iter():
                if elem.text and len(elem.text.strip()) > 30:
                    text_parts.append(elem.text.strip())

    if not text_parts:
        return None

    full_text = " ".join(text_parts)
    full_text = re.sub(r"\s+", " ", full_text).strip()

    return full_text if len(full_text) > 500 else None


# ── arXiv Full Text ───────────────────────────────────────────────────────────

def get_arxiv_fulltext(arxiv_id: str) -> Optional[str]:
    """
    Downloads and extracts full text from an arXiv paper PDF.
    Every arXiv paper has a freely available PDF - no authentication needed.
    """
    if not arxiv_id:
        return None

    # Clean the ID - remove version suffix if present
    arxiv_id = arxiv_id.strip()
    arxiv_id = re.sub(r"v\d+$", "", arxiv_id)

    pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
    time.sleep(1)  # be polite to arXiv

    return extract_text_from_pdf_url(pdf_url)


# ── Main Enrichment Function ──────────────────────────────────────────────────

def enrich_with_fulltext(papers: list[dict]) -> list[dict]:
    """
    Attempts to fetch full text for each paper.
    Priority order:
        1. PubMed Central (for PubMed papers with PMID)
        2. arXiv PDF (for arXiv papers)
        3. Unpaywall open access PDF (for any paper with a DOI)

    Falls back to abstract if full text is unavailable.
    Adds a 'fulltext_available' flag to each paper.
    """

    enriched = []
    total = len(papers)
    success_count = 0

    print(f"Attempting full text retrieval for {total} papers...")

    for i, paper in enumerate(papers):
        paper = paper.copy()
        full_text = None
        source = paper.get("source", "")

        # Strategy 1 - PMC for PubMed papers
        if source == "PubMed" and paper.get("pmid"):
            full_text = get_pmc_fulltext(paper["pmid"])
            if full_text:
                paper["fulltext_source"] = "PubMed Central"

        # Strategy 2 - arXiv PDF
        if full_text is None and source == "arXiv" and paper.get("arxiv_id"):
            full_text = get_arxiv_fulltext(paper["arxiv_id"])
            if full_text:
                paper["fulltext_source"] = "arXiv PDF"

        # Strategy 3 - Unpaywall for any paper with DOI
        if full_text is None and paper.get("doi"):
            oa_url = get_open_access_url(paper["doi"])
            if oa_url:
                full_text = extract_text_from_pdf_url(oa_url)
                if full_text:
                    paper["fulltext_source"] = "Unpaywall OA"

        if full_text and len(full_text) > 500:
            # Use full text but keep abstract as well
            paper["full_text"] = full_text[:8000]  # cap at 8000 chars
            paper["fulltext_available"] = True
            paper["text_for_analysis"] = full_text[:8000]
            success_count += 1
        else:
            paper["full_text"] = None
            paper["fulltext_available"] = False
            paper["text_for_analysis"] = paper.get("abstract", "")
            paper["fulltext_source"] = "Abstract only"

        enriched.append(paper)

        # Progress update every 10 papers
        if (i + 1) % 10 == 0:
            print(
                f"  Progress: {i+1}/{total} papers processed, "
                f"{success_count} full texts retrieved"
            )

        time.sleep(0.3)  # rate limiting across all sources

    print(
        f"Full text retrieval complete: "
        f"{success_count}/{total} papers have full text "
        f"({round(success_count/total*100)}%)"
    )

    return enriched