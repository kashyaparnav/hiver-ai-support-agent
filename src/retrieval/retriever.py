from pathlib import Path
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# --------------------------------------------------
# Project paths
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "amazonhelp_conversations.csv"
)


# --------------------------------------------------
# Retriever
# --------------------------------------------------

class SupportRetriever:

    def __init__(self, max_documents=10000):

        self.max_documents = max_documents

        self.vectorizer = TfidfVectorizer(
            lowercase=True,
            stop_words="english",
            ngram_range=(1, 2),
            min_df=2,
            max_features=30000
        )

        self.data = None
        self.matrix = None

    # --------------------------------------------------
    # Load conversations
    # --------------------------------------------------

    def load_data(self):

        print("Loading support conversations...")

        df = pd.read_csv(DATA_PATH)

        df = df.dropna(
            subset=["customer_text", "brand_text"]
        ).copy()

        df["customer_text"] = (
            df["customer_text"]
            .astype(str)
            .str.strip()
        )

        df["brand_text"] = (
            df["brand_text"]
            .astype(str)
            .str.strip()
        )

        # Remove empty conversations
        df = df[
            (df["customer_text"].str.len() > 10)
            & (df["brand_text"].str.len() > 5)
        ]

        # Remove duplicates
        df = df.drop_duplicates(
            subset=["customer_text"]
        )

        # Limit data for faster local retrieval
        if len(df) > self.max_documents:

            df = df.sample(
                self.max_documents,
                random_state=42
            )

        self.data = df.reset_index(drop=True)

        print(
            f"Conversations loaded: {len(self.data):,}"
        )

    # --------------------------------------------------
    # Build TF-IDF index
    # --------------------------------------------------

    def build_index(self):

        if self.data is None:

            self.load_data()

        print("Building TF-IDF retrieval index...")

        self.matrix = self.vectorizer.fit_transform(
            self.data["customer_text"]
        )

        print(
            f"Index built successfully!"
        )

        print(
            f"Matrix shape: {self.matrix.shape}"
        )

    # --------------------------------------------------
    # Retrieve similar conversations
    # --------------------------------------------------

    def retrieve(
        self,
        query,
        top_k=5
    ):

        if self.matrix is None:

            self.build_index()

        query_vector = (
            self.vectorizer.transform([query])
        )

        scores = cosine_similarity(
            query_vector,
            self.matrix
        ).flatten()

        # Get highest scoring documents
        top_indices = scores.argsort()[
            ::-1
        ][:top_k]

        results = []

        for index in top_indices:

            row = self.data.iloc[index]

            results.append(
                {
                    "customer_text":
                        row["customer_text"],

                    "brand_text":
                        row["brand_text"],

                    "score":
                        float(scores[index])
                }
            )

        return results


# --------------------------------------------------
# Quick test
# --------------------------------------------------

if __name__ == "__main__":

    retriever = SupportRetriever(
        max_documents=10000
    )

    retriever.build_index()

    test_queries = [
        "My package is late",
        "Where is my refund?",
        "I want to cancel my order"
    ]

    for query in test_queries:

        print("\n" + "=" * 70)

        print(
            f"QUERY: {query}"
        )

        print("=" * 70)

        results = retriever.retrieve(
            query,
            top_k=3
        )

        for i, result in enumerate(
            results,
            start=1
        ):

            print(
                f"\nResult {i}"
            )

            print(
                f"Score: {result['score']:.3f}"
            )

            print(
                f"Customer: "
                f"{result['customer_text'][:300]}"
            )

            print(
                f"Support: "
                f"{result['brand_text'][:300]}"
            )
