import sys
from pathlib import Path


# --------------------------------------------------
# Add src to Python path
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

SRC_DIR = PROJECT_ROOT / "src"

sys.path.insert(
    0,
    str(SRC_DIR)
)


# --------------------------------------------------
# Imports
# --------------------------------------------------

from intents.classifier import IntentClassifier
from retrieval.retriever import SupportRetriever


# --------------------------------------------------
# Support AI Pipeline
# --------------------------------------------------

class SupportPipeline:

    def __init__(self):

        print("Initializing Support AI Pipeline...")

        # -----------------------------
        # Intent classifier
        # -----------------------------

        self.classifier = IntentClassifier()

        self.classifier.load()

        print(
            "Intent classifier loaded."
        )

        # -----------------------------
        # Retriever
        # -----------------------------

        self.retriever = SupportRetriever(
            max_documents=10000
        )

        self.retriever.build_index()

        print(
            "Support retrieval index loaded."
        )

        print(
            "\nPipeline ready!"
        )

    # --------------------------------------------------
    # Process customer message
    # --------------------------------------------------

    def process(
        self,
        message,
        top_k=3
    ):

        # -----------------------------
        # Step 1: Intent classification
        # -----------------------------

        classification = (
            self.classifier.predict(
                message
            )
        )

        intent = classification[
            "intent"
        ]

        confidence = classification[
            "confidence"
        ]

        # -----------------------------
        # Step 2: Retrieve examples
        # -----------------------------

        retrieved = (
            self.retriever.retrieve(
                message,
                top_k=top_k
            )
        )

        # -----------------------------
        # Step 3: Build result
        # -----------------------------

        return {
            "message": message,
            "intent": intent,
            "confidence": confidence,
            "retrieved": retrieved
        }


# --------------------------------------------------
# Test pipeline
# --------------------------------------------------

if __name__ == "__main__":

    pipeline = SupportPipeline()

    test_messages = [

        "My package is three days late",

        "Where is my refund?",

        "I want to cancel my order",

        "My payment was charged twice",

        "I cannot login to my account"

    ]

    for message in test_messages:

        print("\n")
        print("=" * 80)

        result = pipeline.process(
            message,
            top_k=3
        )

        print(
            f"Customer: "
            f"{result['message']}"
        )

        print(
            f"Intent: "
            f"{result['intent']}"
        )

        print(
            f"Confidence: "
            f"{result['confidence']:.3f}"
        )

        print(
            "\nRetrieved conversations:"
        )

        for i, item in enumerate(
            result["retrieved"],
            start=1
        ):

            print(
                f"\nResult {i}"
            )

            print(
                f"Similarity: "
                f"{item['score']:.3f}"
            )

            print(
                f"Customer: "
                f"{item['customer_text'][:250]}"
            )

            print(
                f"Support: "
                f"{item['brand_text'][:250]}"
            )

        print("=" * 80)