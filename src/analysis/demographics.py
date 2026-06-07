import pandas as pd
import re

# Population groups we track for representation gaps
DEMOGRAPHIC_GROUPS = {
    "age": {
        "children": ["children", "child", "pediatric", "paediatric", "infant", "toddler"],
        "adolescents": ["adolescent", "teenager", "teen", "youth", "young people", "juvenile"],
        "young adults": ["young adult", "college student", "university student"],
        "older adults": ["older adult", "elderly", "geriatric", "aged", "senior"],
        "adults": ["adult", "middle-aged"]
    },
    "gender": {
        "female": ["women", "female", "girls", "woman", "maternal", "postpartum"],
        "male": ["men", "male", "boys", "man", "paternal"],
        "non-binary / LGBTQ+": ["lgbtq", "transgender", "non-binary", "gender minority", "queer"]
    },
    "geography": {
        "Western / North American": ["united states", "canada", "western", "north american", "american", "usa", "uk", "british", "european"],
        "South Asian": ["india", "pakistan", "bangladesh", "south asia", "indian", "sri lanka"],
        "East Asian": ["china", "japan", "korea", "chinese", "japanese", "korean", "asian"],
        "African": ["africa", "african", "sub-saharan", "nigeria", "kenya", "ethiopia"],
        "Latin American": ["latin america", "brazil", "mexico", "hispanic", "latino", "latina"],
        "Middle Eastern": ["middle east", "arab", "iran", "turkey", "egypt"]
    },
    "socioeconomic": {
        "low income": ["low income", "low-income", "poverty", "deprived", "disadvantaged", "rural", "underserved"],
        "high income": ["high income", "affluent", "private", "insured"]
    },
    "clinical": {
        "comorbid conditions": ["comorbid", "comorbidity", "dual diagnosis", "multiple conditions"],
        "treatment-resistant": ["treatment-resistant", "refractory", "treatment resistant"],
        "first episode": ["first episode", "first-episode", "early onset"]
    }
}


def analyse_demographics(df: pd.DataFrame) -> dict:
    """
    Scans all paper abstracts for demographic mentions.
    Returns counts per group and flags underrepresented populations.
    """

    results = {}

    for category, groups in DEMOGRAPHIC_GROUPS.items():
        category_counts = {}

        for group_name, keywords in groups.items():
            count = 0
            papers_mentioning = []

            for idx, row in df.iterrows():
                text = (
                    str(row.get("title", "")) + " " +
                    str(row.get("abstract", ""))
                ).lower()

                if any(kw in text for kw in keywords):
                    count += 1
                    papers_mentioning.append(idx)

            category_counts[group_name] = {
                "count": count,
                "percentage": round(count / len(df) * 100, 1),
                "paper_indices": papers_mentioning
            }

        results[category] = category_counts

    return results


def identify_demographic_gaps(
    demo_results: dict,
    df: pd.DataFrame,
    low_threshold: float = 10.0
) -> list[dict]:
    """
    Flags demographic groups that appear in fewer than low_threshold% of papers.
    These are representation gaps-the literature ignores these populations.
    """

    gaps = []
    total_papers = len(df)

    for category, groups in demo_results.items():
        for group_name, data in groups.items():
            pct = data["percentage"]

            if pct < low_threshold:
                gaps.append({
                    "category": category,
                    "group": group_name,
                    "papers_found": data["count"],
                    "percentage": pct,
                    "severity": (
                        "Critical" if pct == 0 else
                        "High" if pct < 5 else
                        "Medium"
                    ),
                    "description": (
                        f"No papers mention {group_name} populations."
                        if pct == 0 else
                        f"Only {data['count']} of {total_papers} papers "
                        f"({pct}%) address {group_name} populations."
                    )
                })

    gaps.sort(key=lambda x: x["percentage"])
    return gaps


def format_demographic_report(gaps: list[dict], topic: str) -> str:
    """Formats demographic gaps as readable English."""

    if not gaps:
        return "No significant demographic representation gaps identified."

    lines = ["DEMOGRAPHIC REPRESENTATION GAPS", "=" * 40, ""]

    # Group by category
    by_category = {}
    for gap in gaps:
        cat = gap["category"]
        by_category.setdefault(cat, []).append(gap)

    for category, cat_gaps in by_category.items():
        lines.append(f"[{category.upper()}]")
        for gap in cat_gaps:
            lines.append(f"  • {gap['description']}")
        lines.append("")

    return "\n".join(lines)
