from pathlib import Path

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from intents.classifier import IntentClassifier
import pandas as pd
from classifier import IntentClassifier
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

TRAIN_PATH = (
    PROJECT_ROOT
    / "data"
    / "golden"
    / "train.csv"
)


def main():

    print("Loading training dataset...")

    train_df = pd.read_csv(TRAIN_PATH)

    print(
        f"Training examples: {len(train_df):,}"
    )

    print("\nIntent distribution:")
    print(
        train_df["intent"]
        .value_counts()
        .sort_index()
    )

    # Create classifier
    classifier = IntentClassifier()

    # Train
    classifier.train(
        train_df["customer_text"],
        train_df["intent"]
    )

    # Save
    classifier.save()

    # Quick test
    print("\nQuick prediction tests:\n")

    test_messages = [
        "My package is three days late",
        "Where is my refund?",
        "I want to cancel my order",
        "My payment was charged twice",
        "I cannot login to my account"
    ]

    for message in test_messages:

        result = classifier.predict(message)

        print(f"Message: {message}")
        print(f"Intent: {result['intent']}")
        print(
            f"Confidence: "
            f"{result['confidence']:.3f}"
        )
        print("-" * 60)


if __name__ == "__main__":
    main()