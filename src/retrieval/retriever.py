import sys
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ---------------------------------------------------------
# Project paths
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "amazonhelp_conversations.csv"
)


# ---------------------------------------------------------
# Support Retriever
# ---------------------------------------------------------

class SupportRetriever:

    def __init__(self, max_documents=10000):

        self.max_documents = max_documents

        self.documents = None
        self.vectorizer = None
        self.matrix = None

        print("Loading support conversations...")


    # -----------------------------------------------------
    # Load conversations
    # -----------------------------------------------------

    def load_data(self):

        if not DATA_FILE.exists():
            raise FileNotFoundError(
                f"Conversation file not found:\n{DATA_FILE}"
            )

        df = pd.read_csv(DATA_FILE)

        df = df[
            ["customer_text", "brand_text"]
        ].dropna()

        # Limit documents for local development
        if len(df) > self.max_documents:
            df = df.sample(
                n=self.max_documents,
                random_state=42
            ).reset_index(drop=True)

        else:
            df = df.reset_index(drop=True)

        self.documents = df

        print(
            f"Conversations loaded: {len(self.documents):,}"
        )


    # -----------------------------------------------------
    # Build TF-IDF index
    # -----------------------------------------------------

    def build_index(self):

        self.load_data()

        print("Building TF-IDF retrieval index...")

        self.vectorizer = TfidfVectorizer(
            lowercase=True,
            stop_words="english",
            ngram_range=(1, 2),
            min_df=2,
            max_features=30000
        )

        self.matrix = self.vectorizer.fit_transform(
            self.documents["customer_text"].astype(str)
        )

        print("Index built successfully!")

        print(
            f"Matrix shape: {self.matrix.shape}"
        )


    # -----------------------------------------------------
    # Intent keyword hints
    # -----------------------------------------------------

    INTENT_HINTS = {

        "account_issue": [
            "account",
            "login",
            "log in",
            "password",
            "sign in",
            "access"
        ],

        "delivery_date": [
            "delivery date",
            "arrive",
            "arrival",
            "when will",
            "when should",
            "expected delivery"
        ],

        "delivery_delay": [
            "late",
            "delayed",
            "delay",
            "days late",
            "not arrived",
            "overdue"
        ],

        "delivery_status": [
            "where is",
            "track",
            "tracking",
            "shipment",
            "package status",
            "order status"
        ],

        "missing_package": [
            "missing",
            "lost",
            "not received",
            "didn't receive",
            "did not receive",
            "package missing"
        ],

        "order_cancellation": [
            "cancel",
            "cancellation",
            "cancel my order"
        ],

        "payment_issue": [
            "payment",
            "charged",
            "charge",
            "card",
            "credit card",
            "debit card",
            "charged twice"
        ],

        "prime_issue": [
            "prime",
            "prime membership",
            "prime subscription"
        ],

        "product_issue": [
            "product",
            "item",
            "damaged",
            "broken",
            "defective"
        ],

        "refund_status": [
            "refund",
            "refunded",
            "money back",
            "refund status"
        ],

        "return_issue": [
            "return",
            "return item",
            "send back",
            "returning"
        ],
    }


    # -----------------------------------------------------
    # Intent-aware keyword score
    # -----------------------------------------------------

    def intent_keyword_score(
        self,
        text,
        intent
    ):

        if not intent:
            return 0.0

        text = str(text).lower()

        keywords = self.INTENT_HINTS.get(
            intent,
            []
        )

        if not keywords:
            return 0.0

        matches = sum(
            1
            for keyword in keywords
            if keyword.lower() in text
        )

        return min(
            matches / max(len(keywords), 1),
            1.0
        )


    # -----------------------------------------------------
    # Retrieve relevant conversations
    # -----------------------------------------------------

    def retrieve(
        self,
        query,
        top_k=3,
        intent=None
    ):

        if self.documents is None:
            self.build_index()

        if not query or not str(query).strip():
            return []

        # Vectorize customer query
        query_vector = self.vectorizer.transform(
            [str(query)]
        )

        # Calculate cosine similarity
        similarities = cosine_similarity(
            query_vector,
            self.matrix
        ).flatten()

        results = []

        for index, similarity in enumerate(
            similarities
        ):

            customer_text = self.documents.iloc[
                index
            ]["customer_text"]

            brand_text = self.documents.iloc[
                index
            ]["brand_text"]

            # Intent keyword relevance
            intent_score = self.intent_keyword_score(
                customer_text,
                intent
            )

            # Combined ranking score
            #
            # Similarity remains the main signal.
            # Intent relevance provides an additional
            # boost to cases matching the detected intent.

            final_score = (
                0.75 * float(similarity)
                +
                0.25 * float(intent_score)
            )

            results.append(
                {
                    "customer_text": customer_text,
                    "brand_text": brand_text,
                    "similarity": float(similarity),
                    "intent_score": float(intent_score),
                    "score": float(final_score)
                }
            )

        # Sort by combined score
        results.sort(
            key=lambda x: x["score"],
            reverse=True
        )

        return results[:top_k]


# ---------------------------------------------------------
# Test retriever
# ---------------------------------------------------------

if __name__ == "__main__":

    retriever = SupportRetriever(
        max_documents=10000
    )

    retriever.build_index()

    test_queries = [
        (
            "My package is three days late",
            "delivery_delay"
        ),
        (
            "Where is my refund?",
            "refund_status"
        ),
        (
            "I cannot login to my account",
            "account_issue"
        ),
    ]

    for query, intent in test_queries:

        print("\n")
        print("=" * 80)

        print(f"Query: {query}")
        print(f"Intent: {intent}")

        results = retriever.retrieve(
            query,
            top_k=3,
            intent=intent
        )

        for i, result in enumerate(
            results,
            start=1
        ):

            print(
                f"\nResult {i}"
            )

            print(
                f"Combined Score: "
                f"{result['score']:.3f}"
            )

            print(
                f"Similarity: "
                f"{result['similarity']:.3f}"
            )

            print(
                f"Intent Score: "
                f"{result['intent_score']:.3f}"
            )

            print(
                f"Customer: "
                f"{result['customer_text']}"
            )

            print(
                f"Support: "
                f"{result['brand_text']}"
            )

        print("=" * 80)