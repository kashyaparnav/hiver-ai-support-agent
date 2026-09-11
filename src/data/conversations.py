from pathlib import Path
import pandas as pd

# Project Paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DATA = PROJECT_ROOT / "data" / "raw" / "twcs.csv"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
PROCESSED_DIR.mkdir(exist_ok=True)


def build_brand_conversations(brand_name="AmazonHelp", chunksize=200000):
    """
    Build Customer → Brand conversation pairs from TWCS dataset.
    """

    print(f"\nBuilding conversations for: {brand_name}")

    # ---------- Step 1 : Load Brand Tweets ----------
    brand_chunks = []

    for chunk in pd.read_csv(RAW_DATA, chunksize=chunksize):
        brand = chunk[
            chunk["author_id"].astype(str) == brand_name
        ]
        if not brand.empty:
            brand_chunks.append(brand)

    brand_df = pd.concat(brand_chunks, ignore_index=True)

    print(f"Brand replies found: {len(brand_df):,}")

    # ---------- Step 2 : Map Brand Tweet IDs ----------
    brand_lookup = (
        brand_df.set_index(
            brand_df["tweet_id"].astype(str)
        )
        .to_dict("index")
    )

    brand_ids = set(brand_lookup.keys())

    # ---------- Step 3 : Find Customer Tweets ----------
    conversation_rows = []

    for chunk in pd.read_csv(RAW_DATA, chunksize=chunksize):

        customer_rows = chunk[
            chunk["inbound"] == True
        ]

        for _, row in customer_rows.iterrows():

            if pd.isna(row["response_tweet_id"]):
                continue

            response_ids = [
                x.strip()
                for x in str(row["response_tweet_id"]).split(",")
                if x.strip()
            ]

            for rid in response_ids:

                if rid in brand_ids:

                    brand = brand_lookup[rid]

                    conversation_rows.append(
                        {
                            "customer_tweet_id": row["tweet_id"],
                            "customer_text": row["text"],
                            "brand_tweet_id": brand["tweet_id"],
                            "brand_text": brand["text"],
                            "brand_name": brand_name,
                        }
                    )

    conversations = pd.DataFrame(conversation_rows)

    print(f"Conversation pairs created: {len(conversations):,}")

    # ---------- Step 4 : Save ----------
    output_path = (
        PROCESSED_DIR
        / f"{brand_name.lower()}_conversations.csv"
    )

    conversations.to_csv(output_path, index=False)

    print(f"Saved to:\n{output_path}")

    return conversations


if __name__ == "__main__":

    conversations = build_brand_conversations("AmazonHelp")

    print("\nFirst 5 Conversations:\n")
    print(
        conversations[
            ["customer_text", "brand_text"]
        ].head()
    )