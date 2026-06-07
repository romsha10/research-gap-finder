import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


# Words that signal positive/supportive findings
POSITIVE_SIGNALS = [
    "effective", "significant", "improved", "benefit", "successful",
    "positive", "reduced", "increased", "better", "superior",
    "efficacious", "promising", "robust", "strong", "supported"
]

# Words that signal negative/null findings
NEGATIVE_SIGNALS = [
    "ineffective", "no significant", "failed", "no difference",
    "no benefit", "negative", "inferior", "worse", "null",
    "not supported", "no improvement", "no effect", "limited",
    "inconclusive", "weak"
]


def get_conclusion_sentiment(abstract: str) -> dict:
    """
    Scores an abstract's conclusion as positive, negative, or neutral.
    Simple lexicon-based approach-fast, interpretable, no model needed.

    Returns a dict with:
        positive_score: 0-1
        negative_score: 0-1
        dominant: 'positive' | 'negative' | 'neutral'
    """

    text = abstract.lower()

    # Focus on the last 40% of the abstract-that's where conclusions live
    conclusion_start = int(len(text) * 0.6)
    conclusion_text = text[conclusion_start:]

    pos_count = sum(1 for word in POSITIVE_SIGNALS if word in conclusion_text)
    neg_count = sum(1 for word in NEGATIVE_SIGNALS if word in conclusion_text)

    total = pos_count + neg_count
    if total == 0:
        return {"positive_score": 0.5, "negative_score": 0.5, "dominant": "neutral"}

    pos_score = pos_count / total
    neg_score = neg_count / total

    if pos_score > 0.6:
        dominant = "positive"
    elif neg_score > 0.6:
        dominant = "negative"
    else:
        dominant = "neutral"

    return {
        "positive_score": round(pos_score, 3),
        "negative_score": round(neg_score, 3),
        "dominant": dominant
    }


def detect_contradictions(
    df: pd.DataFrame,
    embeddings: np.ndarray,
    similarity_threshold: float = 0.60,
    min_sentiment_divergence: float = 0.4
) -> list[dict]:
    """
    Finds pairs of papers that:
    1. Are semantically similar (about the same question)-high cosine similarity
    2. Reach opposite conclusions-one positive dominant, one negative dominant

    These are genuine contradictions in the literature.
    """

    print("Running contradiction detection...")

    # Compute pairwise similarity
    sim_matrix = cosine_similarity(embeddings)
    n = len(df)
    contradictions = []

    for i in range(n):
        for j in range(i + 1, n):
            sim = float(sim_matrix[i][j])
            if sim < similarity_threshold:
                continue  # not about the same topic

            abstract_a = str(df.iloc[i].get("abstract", ""))
            abstract_b = str(df.iloc[j].get("abstract", ""))

            sentiment_a = get_conclusion_sentiment(abstract_a)
            sentiment_b = get_conclusion_sentiment(abstract_b)

            # Contradiction = one positive dominant, one negative dominant
            is_contradiction = (
                sentiment_a["dominant"] == "positive" and
                sentiment_b["dominant"] == "negative"
            ) or (
                sentiment_a["dominant"] == "negative" and
                sentiment_b["dominant"] == "positive"
            )

            # Also catch cases where sentiment scores diverge strongly
            sentiment_divergence = abs(
                sentiment_a["positive_score"] - sentiment_b["positive_score"]
            )

            if is_contradiction or sentiment_divergence >= min_sentiment_divergence:
                contradictions.append({
                    "paper_a_idx": i,
                    "paper_b_idx": j,
                    "paper_a_title": df.iloc[i]["title"],
                    "paper_b_title": df.iloc[j]["title"],
                    "paper_a_year": df.iloc[i].get("year", "?"),
                    "paper_b_year": df.iloc[j].get("year", "?"),
                    "paper_a_source": df.iloc[i].get("source", "?"),
                    "paper_b_source": df.iloc[j].get("source", "?"),
                    "similarity": round(sim, 3),
                    "sentiment_a": sentiment_a["dominant"],
                    "sentiment_b": sentiment_b["dominant"],
                    "sentiment_divergence": round(sentiment_divergence, 3),
                    "paper_a_url": df.iloc[i].get("url", ""),
                    "paper_b_url": df.iloc[j].get("url", ""),
                    "contradiction_strength": (
                        "Strong" if is_contradiction and sentiment_divergence > 0.5
                        else "Moderate" if is_contradiction
                        else "Weak"
                    )
                })

    contradictions.sort(
        key=lambda x: (x["similarity"] + x["sentiment_divergence"]),
        reverse=True
    )

    print(f"Found {len(contradictions)} potential contradictions")
    return contradictions
