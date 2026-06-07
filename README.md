# Research Gap Finder

Automated research gap detection for any academic topic. Given a research 
question, the system retrieves papers from multiple academic databases, maps 
the knowledge landscape using NLP, identifies contradictions between papers, 
and generates a structured research gap report.

## Live Demo

[Coming soon - Streamlit Cloud deployment]

## What It Does

- Retrieves papers from PubMed, Semantic Scholar, OpenAlex, and arXiv
- Detects the academic field automatically and queries relevant databases
- Generates semantic embeddings using Sentence Transformers
- Clusters papers into subtopic groups without requiring a predefined number of clusters
- Identifies three types of research gaps: underexplored subtopics, adjacent gaps, and temporal gaps
- Detects contradictions between papers using semantic similarity and sentiment analysis
- Analyses demographic representation across the retrieved literature
- Builds an interactive citation network visualisation
- Generates a structured plain-English research gap report

## Tech Stack

Python, Streamlit, Sentence Transformers, FAISS, Scikit-learn, 
PubMed API, Semantic Scholar API, OpenAlex API, arXiv API, 
Plotly, NetworkX, Pandas

## Setup

### 1. Clone the repository

git clone https://github.com/YOUR_USERNAME/research-gap-finder.git
cd research-gap-finder

### 2. Create virtual environment

python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Mac/Linux

### 3. Install dependencies

pip install -r requirements.txt

### 4. Configure API keys

Create a .env file in the root directory:

NCBI_API_KEY=your_pubmed_api_key_here

Get a free PubMed API key at https://www.ncbi.nlm.nih.gov/account/settings/
No other API keys are required - all other sources are freely accessible.

### 5. Run the app

streamlit run app.py

Open http://localhost:8501 in your browser.

## How It Works

### Retrieval
Papers are fetched from four databases simultaneously. The system detects 
the academic field from the query and weights sources accordingly - medical 
topics get more PubMed results, CS topics get more arXiv and OpenAlex results.

### Embeddings
Each paper's title and abstract are combined and encoded into a 384-dimensional 
vector using the all-MiniLM-L6-v2 Sentence Transformer model. Papers on similar 
topics receive similar vectors.

### Clustering
Embeddings are reduced to 10 dimensions using PCA and clustered using DBSCAN. 
The algorithm determines the number of clusters automatically. A KMeans fallback 
handles cases where DBSCAN finds no structure.

### Gap Detection
Three gap types are identified:
- Underexplored subtopics: clusters with 1-2 papers
- Adjacent gaps: thin clusters semantically close to dense clusters
- Temporal gaps: clusters where all papers are more than 5 years old

### Contradiction Detection
Pairs of papers with high semantic similarity but opposing conclusion 
sentiment are flagged as potential contradictions.

### Demographic Analysis
Abstracts are scanned for mentions of specific population groups across 
four dimensions: age, gender, geography, and socioeconomic status.

## Project Structure

research-gap-finder/
├── app.py                        # Streamlit application
├── requirements.txt
├── src/
│   ├── retrieval/
│   │   ├── pubmed.py             # PubMed API
│   │   ├── semantic_scholar.py   # Semantic Scholar API
│   │   ├── arxiv.py              # arXiv API
│   │   ├── openalex.py           # OpenAlex API
│   │   ├── cache.py              # Local result caching
│   │   └── retriever.py          # Unified retrieval with field detection
│   ├── processing/
│   │   ├── embeddings.py         # Sentence Transformer embeddings
│   │   ├── clustering.py         # DBSCAN clustering and keyword extraction
│   │   └── contradiction.py      # Contradiction detection
│   ├── analysis/
│   │   ├── gap_detector.py       # Research gap identification
│   │   └── demographics.py       # Demographic representation analysis
│   ├── visualisation/
│   │   └── citation_graph.py     # NetworkX and Plotly citation network
│   └── report/
│       └── generator.py          # Structured report generation

## Limitations

- Analysis is based on abstracts only, not full paper text
- Contradiction detection uses lexical sentiment analysis which is approximate
- Results depend on what the free APIs return - paywalled papers are not included
- Clustering quality improves significantly with more papers (50+ per source recommended)

## Acknowledgements

Built as part of MSc research at University of Glasgow. 
APIs used: PubMed/NCBI, Semantic Scholar, OpenAlex, arXiv.
