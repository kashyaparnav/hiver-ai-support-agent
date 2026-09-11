from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split


PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "golden"
    / "amazonhelp_golden_candidates.csv"
)

GOLDEN_DIR = (
    PROJECT_ROOT
    / "data"
    / "golden"
)


def prepare_dataset():

    print("Loading candidate dataset...")

    df = pd.read_csv(INPUT_PATH)

    print(f"Original rows: {len(df):,}")

    # --------------------------------------------------
    # Basic quality checks
    # --------------------------------------------------

    df = df.dropna(
        subset=[
            "customer_text",
            "intent"
        ]
    )

    df["customer_text"] = (
        df["customer_text"]
        .astype(str)
        .str.strip()
    )

    df["intent"] = (
        df["intent"]
        .astype(str)
        .str.strip()
    )

    # Remove empty messages
    df = df[
        df["customer_text"].str.len() >= 15
    ]

    # Remove duplicate messages
    df = df.drop_duplicates(
        subset=["customer_text"]
    )

    print(
        f"After quality checks: {len(df):,}"
    )

    # --------------------------------------------------
    # Show intent distribution
    # --------------------------------------------------

    print("\nIntent distribution:")

    print(
        df["intent"]
        .value_counts()
        .sort_index()
    )

    # --------------------------------------------------
    # Stratified train/test split
    # --------------------------------------------------

    train_df, test_df = train_test_split(
        df,
        test_size=0.20,
        random_state=42,
        stratify=df["intent"]
    )

    # --------------------------------------------------
    # Train/validation split
    # --------------------------------------------------

    train_df, validation_df = train_test_split(
        train_df,
        test_size=0.20,
        random_state=42,
        stratify=train_df["intent"]
    )

    # --------------------------------------------------
    # Save datasets
    # --------------------------------------------------

    train_path = (
        GOLDEN_DIR
        / "train.csv"
    )

    validation_path = (
        GOLDEN_DIR
        / "validation.csv"
    )

    test_path = (
        GOLDEN_DIR
        / "test.csv"
    )

    train_df.to_csv(
        train_path,
        index=False
    )

    validation_df.to_csv(
        validation_path,
        index=False
    )

    test_df.to_csv(
        test_path,
        index=False
    )

    # --------------------------------------------------
    # Final summary
    # --------------------------------------------------

    print("\nDataset preparation complete!")

    print(
        f"Training set:   {len(train_df):,}"
    )

    print(
        f"Validation set: {len(validation_df):,}"
    )

    print(
        f"Test set:       {len(test_df):,}"
    )

    print("\nFiles created:")

    print(train_path)
    print(validation_path)
    print(test_path)


if __name__ == "__main__":
    prepare_dataset()