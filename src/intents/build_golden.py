from pathlib import Path
import pandas as pd
import re


# --------------------------------------------------
# Paths
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "amazonhelp_conversations.csv"
)

GOLDEN_DIR = (
    PROJECT_ROOT
    / "data"
    / "golden"
)

GOLDEN_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# --------------------------------------------------
# Intent keyword rules
# --------------------------------------------------

INTENT_RULES = {

    "delivery_delay": [
        r"\blate\b",
        r"\bdelayed\b",
        r"\bdelay\b",
        r"\bhasn't arrived\b",
        r"\bhas not arrived\b",
        r"\bnot arrived\b",
        r"\btaking too long\b",
        r"\bstill waiting\b"
    ],

    "delivery_status": [
        r"\bwhere is my\b",
        r"\bwhere's my\b",
        r"\btracking\b",
        r"\btrack my\b",
        r"\border status\b",
        r"\bdelivery status\b"
    ],

    "delivery_date": [
        r"\bdelivery date\b",
        r"\barrive on\b",
        r"\barriving\b",
        r"\bwhen will.*arrive\b",
        r"\bwhen.*delivery\b"
    ],

    "missing_package": [
        r"\bmissing package\b",
        r"\bpackage.*missing\b",
        r"\bmarked delivered\b",
        r"\bsays delivered\b",
        r"\bdidn't receive\b",
        r"\bdid not receive\b",
        r"\bnever received\b"
    ],

    "prime_issue": [
        r"\bprime\b",
        r"\bprime membership\b",
        r"\bprime subscription\b"
    ],

    "product_issue": [
        r"\bbroken\b",
        r"\bdamaged\b",
        r"\bdefective\b",
        r"\bwrong item\b",
        r"\bwrong product\b",
        r"\bproduct.*issue\b",
        r"\bitem.*issue\b"
    ],

    "payment_issue": [
        r"\bpayment\b",
        r"\bcharged\b",
        r"\bcharge\b",
        r"\bbilling\b",
        r"\bpaid\b",
        r"\bcard\b"
    ],

    "account_issue": [
        r"\baccount\b",
        r"\blogin\b",
        r"\blog in\b",
        r"\bpassword\b",
        r"\blocked\b",
        r"\bsign in\b",
        r"\baccess.*account\b"
    ],

    "refund_status": [
        r"\brefund\b",
        r"\brefunded\b",
        r"\bmoney back\b",
        r"\breimburse"
    ],

    "return_issue": [
        r"\breturn\b",
        r"\breplacement\b",
        r"\breplace\b"
    ],

    "order_cancellation": [
        r"\bcancel\b",
        r"\bcancelled\b",
        r"\bcanceled\b",
        r"\bcancellation\b"
    ]
}


# --------------------------------------------------
# Label one message
# --------------------------------------------------

def find_intent(text):

    text = str(text).lower()

    matches = []

    for intent, patterns in INTENT_RULES.items():

        for pattern in patterns:

            if re.search(pattern, text):

                matches.append(intent)
                break

    # Only accept messages with exactly one
    # matching intent.
    if len(matches) == 1:
        return matches[0]

    return None


# --------------------------------------------------
# Build candidate golden dataset
# --------------------------------------------------

def main():

    print("Loading conversations...")

    df = pd.read_csv(INPUT_PATH)

    print(
        f"Loaded {len(df):,} conversations."
    )

    # Keep customer messages
    data = df[
        ["customer_text", "brand_text"]
    ].dropna(
        subset=["customer_text"]
    ).copy()

    # Remove duplicates
    data = data.drop_duplicates(
        subset=["customer_text"]
    )

    # Apply intent rules
    data["intent"] = (
        data["customer_text"]
        .apply(find_intent)
    )

    # Keep only confidently matched examples
    candidates = data[
        data["intent"].notna()
    ].copy()

    print(
        f"Candidate labeled messages: "
        f"{len(candidates):,}"
    )

    # Balance the dataset
    samples = []

    for intent in INTENT_RULES:

        intent_rows = candidates[
            candidates["intent"] == intent
        ]

        if len(intent_rows) == 0:
            continue

        # Maximum 300 examples per intent
        n = min(300, len(intent_rows))

        sampled = intent_rows.sample(
            n=n,
            random_state=42
        )

        samples.append(sampled)

        print(
            f"{intent}: {len(intent_rows):,} "
            f"available → {n} selected"
        )

    golden = pd.concat(
        samples,
        ignore_index=True
    )

    # Shuffle
    golden = golden.sample(
        frac=1,
        random_state=42
    ).reset_index(drop=True)

    # Add review column
    golden["reviewed"] = False

    output_path = (
        GOLDEN_DIR
        / "amazonhelp_golden_candidates.csv"
    )

    golden.to_csv(
        output_path,
        index=False
    )

    print(
        "\nCandidate golden dataset created!"
    )

    print(
        f"Saved to:\n{output_path}"
    )

    print(
        "\nIntent distribution:"
    )

    print(
        golden["intent"]
        .value_counts()
        .sort_index()
    )


if __name__ == "__main__":
    main()