from pathlib import Path
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "golden"
    / "amazonhelp_golden_candidates.csv"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "golden"
    / "amazonhelp_golden.csv"
)


def create_review_file():

    df = pd.read_csv(INPUT_PATH)

    # Keep only the columns needed for review
    review_df = df[
        [
            "customer_text",
            "brand_text",
            "intent"
        ]
    ].copy()

    # Human verification fields
    review_df["correct_intent"] = ""
    review_df["reviewed"] = False

    review_df.to_csv(
        OUTPUT_PATH,
        index=False
    )

    print("Review dataset created successfully!")
    print(f"Rows: {len(review_df):,}")
    print(f"Saved to:\n{OUTPUT_PATH}")

    print("\nColumns:")
    print(review_df.columns.tolist())


if __name__ == "__main__":
    create_review_file()