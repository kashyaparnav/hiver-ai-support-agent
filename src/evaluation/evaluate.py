import sys
import json
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, classification_report

# ---------------------------------------------------------
# Project paths
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"

sys.path.insert(0, str(SRC_DIR))

# ---------------------------------------------------------
# Internal imports
# ---------------------------------------------------------

from intents.classifier import IntentClassifier
from retrieval.retriever import SupportRetriever


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

TEST_FILE = PROJECT_ROOT / "data" / "golden" / "test.csv"
OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "evaluation_results.json"
)

RETRIEVAL_SAMPLE_SIZE = 100


# ---------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------

def main():

    print("=" * 80)
    print("END-TO-END SUPPORT AGENT EVALUATION")
    print("=" * 80)

    # -----------------------------------------------------
    # Load test dataset
    # -----------------------------------------------------

    print("\nLoading test dataset...")

    test_df = pd.read_csv(TEST_FILE)

    print(
        f"Test samples available: {len(test_df)}"
    )

    # -----------------------------------------------------
    # Load classifier
    # -----------------------------------------------------

    print("\nLoading intent classifier...")

    classifier = IntentClassifier()
    classifier.load()

    print("✓ Classifier loaded")

    # -----------------------------------------------------
    # Evaluate intent classification
    # -----------------------------------------------------

    y_true = test_df["intent"].tolist()

    y_pred = []

    print("\nRunning intent predictions...")

    for text in test_df["customer_text"]:

        prediction = classifier.predict(text)

        y_pred.append(
            prediction["intent"]
        )

    accuracy = accuracy_score(
        y_true,
        y_pred
    )

    report = classification_report(
        y_true,
        y_pred,
        output_dict=True,
        zero_division=0
    )

    print(
        f"\nIntent Accuracy: "
        f"{accuracy:.4f}"
    )

    print(
        f"Macro F1: "
        f"{report['macro avg']['f1-score']:.4f}"
    )

    print(
        f"Weighted F1: "
        f"{report['weighted avg']['f1-score']:.4f}"
    )

    # -----------------------------------------------------
    # Retrieval evaluation
    # -----------------------------------------------------

    print("\nLoading retrieval system...")

    retriever = SupportRetriever(
        max_documents=10000
    )

    retriever.build_index()

    print("✓ Retriever loaded")

    # Use a small sample to avoid unnecessary computation
    retrieval_df = test_df.sample(
        n=min(
            RETRIEVAL_SAMPLE_SIZE,
            len(test_df)
        ),
        random_state=42
    )

    print(
        f"\nEvaluating retrieval on "
        f"{len(retrieval_df)} samples..."
    )

    retrieval_results = []

    for _, row in retrieval_df.iterrows():

            results = retriever.retrieve(
                row["customer_text"],
                top_k=3
            )

            top_score = 0.0

            if results:

                first_result = results[0]

                if isinstance(first_result, dict):

                    top_score = float(
                        first_result.get(
                            "similarity",
                            first_result.get(
                                "score",
                                first_result.get(
                                    "similarity_score",
                                    0.0
                                )
                            )
                        )
                    )

            retrieval_results.append(
                top_score
            )

    avg_retrieval_score = 0.0

    if retrieval_results:

        avg_retrieval_score = sum(
            retrieval_results
        ) / len(retrieval_results)

    print(
        f"Average top-1 retrieval similarity: "
        f"{avg_retrieval_score:.4f}"
    )

    # -----------------------------------------------------
    # Final evaluation results
    # -----------------------------------------------------

    results = {
        "dataset": {
            "test_samples": len(test_df),
            "retrieval_evaluation_samples": len(
                retrieval_df
            )
        },

        "intent_classification": {
            "accuracy": round(
                float(accuracy),
                4
            ),
            "macro_f1": round(
                float(
                    report["macro avg"]["f1-score"]
                ),
                4
            ),
            "weighted_f1": round(
                float(
                    report["weighted avg"]["f1-score"]
                ),
                4
            )
        },

        "retrieval": {
            "average_top1_similarity": round(
                float(avg_retrieval_score),
                4
            )
        }
    }

    # -----------------------------------------------------
    # Save results
    # -----------------------------------------------------

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            results,
            f,
            indent=4
        )

    print(
        f"\n✓ Evaluation results saved to:"
    )

    print(
        OUTPUT_FILE
    )

    print("\n" + "=" * 80)
    print("EVALUATION COMPLETE")
    print("=" * 80)


# ---------------------------------------------------------
# Entry point
# ---------------------------------------------------------

if __name__ == "__main__":
    main()
