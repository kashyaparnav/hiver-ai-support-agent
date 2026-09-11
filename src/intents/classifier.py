"""
Intent classifier for the AmazonHelp support agent.

Uses TF-IDF + Logistic Regression to classify customer
messages into the intent taxonomy.
"""

from pathlib import Path
import joblib
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from intents.taxonomy import INTENTS


PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_DIR = PROJECT_ROOT / "data" / "processed" / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

VECTORIZER_PATH = MODEL_DIR / "intent_vectorizer.joblib"
MODEL_PATH = MODEL_DIR / "intent_classifier.joblib"


class IntentClassifier:

    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            stop_words="english",
            lowercase=True,
            ngram_range=(1, 2),
            min_df=2,
            max_features=10000
        )

        self.model = LogisticRegression(
            max_iter=1000,
            class_weight="balanced"
        )

        self.is_trained = False

    def train(self, texts, labels):
        """
        Train the classifier.
        """

        print("Training intent classifier...")

        X = self.vectorizer.fit_transform(texts)

        print(
            f"Training samples: {X.shape[0]:,}"
        )

        print(
            f"Vocabulary size: {X.shape[1]:,}"
        )

        self.model.fit(X, labels)

        self.is_trained = True

        print("Training complete.")

    def predict(self, text):
        """
        Predict intent and confidence for one message.
        """

        if not self.is_trained:
            raise RuntimeError(
                "Classifier has not been trained."
            )

        X = self.vectorizer.transform([text])

        probabilities = self.model.predict_proba(X)[0]

        best_index = probabilities.argmax()

        intent = self.model.classes_[best_index]

        confidence = float(
            probabilities[best_index]
        )

        return {
            "intent": intent,
            "confidence": confidence
        }

    def save(self):
        """
        Save model and vectorizer.
        """

        joblib.dump(
            self.vectorizer,
            VECTORIZER_PATH
        )

        joblib.dump(
            self.model,
            MODEL_PATH
        )

        print(
            f"Vectorizer saved to:\n{VECTORIZER_PATH}"
        )

        print(
            f"Classifier saved to:\n{MODEL_PATH}"
        )

    def load(self):
        """
        Load trained model and vectorizer.
        """

        self.vectorizer = joblib.load(
            VECTORIZER_PATH
        )

        self.model = joblib.load(
            MODEL_PATH
        )

        self.is_trained = True

        print("Classifier loaded successfully.")


def train_from_labeled_data(
    csv_path,
    text_column="customer_text",
    label_column="intent"
):
    """
    Train classifier from a labeled CSV.
    """

    df = pd.read_csv(csv_path)

    required_columns = {
        text_column,
        label_column
    }

    missing = required_columns - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing columns: {missing}"
        )

    df = df.dropna(
        subset=[
            text_column,
            label_column
        ]
    )

    # Only use intents defined in taxonomy
    df = df[
        df[label_column].isin(
            INTENTS.keys()
        )
    ]

    if df.empty:
        raise ValueError(
            "No valid labeled data found."
        )

    train_df, test_df = train_test_split(
        df,
        test_size=0.20,
        random_state=42,
        stratify=df[label_column]
    )

    classifier = IntentClassifier()

    classifier.train(
        train_df[text_column],
        train_df[label_column]
    )

    predictions = [
        classifier.predict(text)["intent"]
        for text in test_df[text_column]
    ]

    print("\nClassification Report:\n")

    print(
        classification_report(
            test_df[label_column],
            predictions,
            zero_division=0
        )
    )

    classifier.save()

    return classifier


if __name__ == "__main__":

    print("Intent classifier module loaded.")
    print(
        f"Available intents: {len(INTENTS)}"
    )

    for intent in INTENTS:
        print(f"- {intent}")
