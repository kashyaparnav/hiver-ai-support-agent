from pathlib import Path
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from intents.classifier import IntentClassifier


PROJECT_ROOT = Path(__file__).resolve().parents[2]

TEST_PATH = (
    PROJECT_ROOT
    / "data"
    / "golden"
    / "test.csv"
)

MODEL_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "models"
)


def evaluate_classifier():

    print("Loading test dataset...")

    test_df = pd.read_csv(TEST_PATH)

    print(
        f"Test examples: {len(test_df):,}"
    )

    # Load trained classifier
    classifier = IntentClassifier()

    classifier.load()

    # Predictions
    predictions = []

    for text in test_df["customer_text"]:

        result = classifier.predict(text)

        predictions.append(
            result["intent"]
        )

    y_true = test_df["intent"]
    y_pred = predictions

    # Accuracy
    accuracy = accuracy_score(
        y_true,
        y_pred
    )

    print("\n" + "=" * 70)
    print("CLASSIFIER EVALUATION")
    print("=" * 70)

    print(
        f"\nAccuracy: {accuracy:.4f}"
    )

    # Detailed report
    print("\nClassification Report:\n")

    print(
        classification_report(
            y_true,
            y_pred,
            zero_division=0
        )
    )

    # Confusion matrix
    labels = sorted(
        test_df["intent"].unique()
    )

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=labels
    )

    print("\nConfusion Matrix:\n")

    print(
        pd.DataFrame(
            cm,
            index=labels,
            columns=labels
        )
    )


if __name__ == "__main__":

    evaluate_classifier()
