from pathlib import Path
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"


def load_dataset(filename="twcs.csv"):
    file_path = RAW_DATA_DIR / filename

    if not file_path.exists():
        raise FileNotFoundError(
            f"Dataset not found at: {file_path}"
        )

    print(f"Loading dataset from: {file_path}")

    df = pd.read_csv(file_path)

    print("\nDataset loaded successfully!")
    print(f"Rows: {len(df):,}")
    print(f"Columns: {len(df.columns)}")

    print("\nColumns:")
    for column in df.columns:
        print(f" - {column}")

    return df


if __name__ == "__main__":
    df = load_dataset()

    print("\nFirst 5 rows:")
    print(df.head())