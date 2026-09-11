import pandas as pd


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Remove duplicate rows."""
    return df.drop_duplicates().copy()


def remove_empty_text(df: pd.DataFrame, text_column: str = "text") -> pd.DataFrame:
    """Remove rows where tweet text is empty."""
    df = df.copy()

    if text_column not in df.columns:
        raise ValueError(f"Column '{text_column}' not found.")

    df[text_column] = df[text_column].fillna("").astype(str).str.strip()

    return df[df[text_column] != ""].copy()
