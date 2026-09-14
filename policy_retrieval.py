"""
CREDIT RISK PIPELINE - PHASE 7a: POLICY DOCUMENT RETRIEVAL
========================================================
Goal: given a question like "why was this applicant rejected?", find the
specific passages of the underwriting policy document that are actually
relevant -- so the LLM explanation step (Phase 7b) can be grounded in
real policy text instead of the model's own (unverified) beliefs about
"how underwriting usually works."

RETRIEVAL METHOD, IN PLAIN ENGLISH:
We're using TF-IDF + cosine similarity here, not a semantic embedding
model (like the ones OpenAI/Voyage/sentence-transformers provide).
TF-IDF scores a chunk of text against a query by how much they share
distinctive words (weighted down for common words like "the", weighted
up for words that are rare across the whole document but appear in both
the chunk and the query). It's simpler and needs no extra heavyweight
dependency or paid API call -- appropriate here because our "corpus" is
one short policy document with a handful of sections, not a large
document collection where TF-IDF's lack of semantic understanding
(e.g. matching "cost" to "price") would actually cost us relevant hits.
"""

import re
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

POLICY_PATH = Path(__file__).parent / "policy_docs" / "underwriting_policy.md"


def load_chunks(path=POLICY_PATH):
    """
    Splits the policy markdown file into chunks along its "## " section
    headers, so each chunk is one coherent policy topic (e.g. "Decision
    Tiers" or "Fair Lending and Prohibited Factors") rather than an
    arbitrary fixed-length slice that might cut a rule in half.
    """
    text = Path(path).read_text()
    # Split right before each "## " heading, keeping the heading attached
    # to the section that follows it.
    raw_sections = re.split(r"\n(?=## )", text)

    chunks = []
    for section in raw_sections:
        section = section.strip()
        if not section or not section.startswith("## "):
            continue  # skip the title/intro block before the first "## "
        title = section.splitlines()[0].lstrip("# ").strip()
        chunks.append({"title": title, "text": section})
    return chunks


class PolicyRetriever:
    """Builds a TF-IDF index over the policy chunks once, then answers
    retrieve(query, k) calls cheaply against that index."""

    def __init__(self, path=POLICY_PATH):
        self.chunks = load_chunks(path)
        self.vectorizer = TfidfVectorizer(stop_words="english")
        self.matrix = self.vectorizer.fit_transform(
            [c["text"] for c in self.chunks]
        )

    def retrieve(self, query, k=3):
        """Returns the top-k chunks most relevant to `query`, each with
        its similarity score, highest first."""
        query_vec = self.vectorizer.transform([query])
        scores = cosine_similarity(query_vec, self.matrix)[0]
        ranked = sorted(
            zip(self.chunks, scores), key=lambda pair: pair[1], reverse=True
        )
        return [
            {"title": c["title"], "text": c["text"], "score": round(float(s), 4)}
            for c, s in ranked[:k]
        ]


if __name__ == "__main__":
    retriever = PolicyRetriever()
    print(f"Indexed {len(retriever.chunks)} policy sections:")
    for c in retriever.chunks:
        print(f"  - {c['title']}")

    demo_queries = [
        "why was this applicant rejected, what reasons can we give them?",
        "what does the REVIEW tier mean for the applicant?",
        "can we use SHAP factors to explain a decision?",
        "are we allowed to consider the applicant's age or marital status?",
    ]
    for q in demo_queries:
        print(f"\nQuery: {q!r}")
        for hit in retriever.retrieve(q, k=2):
            print(f"  [{hit['score']}] {hit['title']}")
