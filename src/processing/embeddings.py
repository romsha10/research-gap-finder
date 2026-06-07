from sentence_transformers import SentenceTransformer
import numpy as np
import pandas as pd
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"

MODEL_NAME = "all-MiniLM-L6-v2"
_model = None


def get_model() -> SentenceTransformer:
    """
    Lazy loads the model on first call and reuses it for the rest of the session.
    The model is 80MB, runs on CPU, and is optimised for semantic similarity
    on scientific and academic text.
    """
    global _model
    if _model is None:
        print(f"Loading embedding model: {MODEL_NAME}")
        _model = SentenceTransformer(MODEL_NAME)
        print("Model loaded.")
    return _model


def generate_embeddings(df: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    """
    Converts each paper's title and abstract into a 384-dimensional vector.

    Why title + abstract together:
    The title tells us the topic. The abstract tells us the method, findings,
    and population. Combining them gives the model more signal than either alone.

    Why normalise embeddings:
    Normalising to unit length means cosine similarity between two vectors
    equals their dot product-which is faster to compute and easier to
    interpret (1.0 = identical meaning, 0.0 = completely unrelated).

    Returns:
        df        -original DataFrame with added text_for_embedding column
        embeddings-numpy array of shape (n_papers, 384)
    """
    model = get_model()

    df = df.copy()
    df["text_for_embedding"] = (
        df["title"].fillna("") + ". " + df["abstract"].fillna("")
    )

    texts = df["text_for_embedding"].tolist()
    print(f"Generating embeddings for {len(texts)} papers...")

    embeddings = model.encode(
        texts,
        batch_size=32,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    print(f"Embeddings shape: {embeddings.shape}")
    return df, embeddings


def compute_similarity_matrix(embeddings: np.ndarray) -> np.ndarray:
    """
    Computes cosine similarity between every pair of papers.

    Because embeddings are normalised to unit length, cosine similarity
    is equivalent to the dot product-so we use matrix multiplication
    which is fast even for 200+ papers.

    Result is an (n x n) symmetric matrix where:
        entry [i][j] = semantic similarity between paper i and paper j
        1.0 = papers discuss the same thing
        0.0 = papers are completely unrelated
        values below 0 are rare with normalised embeddings
    """
    similarity_matrix = np.dot(embeddings, embeddings.T)
    return similarity_matrix


def find_similar_pairs(
    df: pd.DataFrame,
    embeddings: np.ndarray,
    threshold: float = 0.75
) -> list[dict]:
    """
    Finds all pairs of papers with similarity above the threshold.
    These are the candidates passed to the contradiction detector.

    Threshold guidance:
        0.85+ = near-identical papers (possible duplicates or replications)
        0.75  = papers clearly on the same specific question (default)
        0.60  = papers on the same broad topic but different angles
        below 0.50 = likely unrelated

    Only the upper triangle of the similarity matrix is checked
    to avoid returning each pair twice (paper A vs B and paper B vs A).
    """
    sim_matrix = compute_similarity_matrix(embeddings)
    n = len(df)
    pairs = []

    for i in range(n):
        for j in range(i + 1, n):
            score = float(sim_matrix[i][j])
            if score >= threshold:
                pairs.append({
                    "paper_a_idx": i,
                    "paper_b_idx": j,
                    "paper_a_title": df.iloc[i]["title"],
                    "paper_b_title": df.iloc[j]["title"],
                    "similarity_score": round(score, 4),
                    "paper_a_source": df.iloc[i].get("source", "Unknown"),
                    "paper_b_source": df.iloc[j].get("source", "Unknown"),
                    "paper_a_year": df.iloc[i].get("year", "?"),
                    "paper_b_year": df.iloc[j].get("year", "?"),
                    "paper_a_url": df.iloc[i].get("url", ""),
                    "paper_b_url": df.iloc[j].get("url", "")
                })

    pairs.sort(key=lambda x: x["similarity_score"], reverse=True)
    print(f"Found {len(pairs)} similar paper pairs above threshold {threshold}")
    return pairs


def get_embedding_for_text(text: str) -> np.ndarray:
    """
    Generates an embedding for a single piece of text.
    Used when we need to embed a custom query or a cluster centroid label
    rather than a full paper.
    """
    model = get_model()
    embedding = model.encode(
        [text],
        convert_to_numpy=True,
        normalize_embeddings=True
    )
    return embedding[0]


def compute_centroid(embeddings: np.ndarray) -> np.ndarray:
    """
    Computes the centroid (average vector) of a set of embeddings.
    Used in gap detection to find the geometric centre of a cluster
    and measure how far other clusters are from it.
    """
    return embeddings.mean(axis=0)
