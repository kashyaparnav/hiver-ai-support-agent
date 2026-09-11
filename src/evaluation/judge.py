import json
from pathlib import Path


# ---------------------------------------------------------
# Project paths
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "response_quality_results.json"
)


# ---------------------------------------------------------
# Response Quality Judge
# ---------------------------------------------------------

class ResponseQualityJudge:

    def __init__(self):
        self.max_score = 5

    def evaluate(self, customer_message, response):

        score = 0
        reasons = []

        response_lower = response.lower()

        # -------------------------------------------------
        # 1. Response exists
        # -------------------------------------------------

        if response.strip():
            score += 1
            reasons.append("Response is present.")
        else:
            reasons.append("Response is empty.")

        # -------------------------------------------------
        # 2. Reasonable response length
        # -------------------------------------------------

        word_count = len(response.split())

        if 5 <= word_count <= 150:
            score += 1
            reasons.append("Response has reasonable length.")
        else:
            reasons.append(
                "Response length may be too short or too long."
            )

        # -------------------------------------------------
        # 3. Helpful language
        # -------------------------------------------------

        helpful_terms = [
            "help",
            "sorry",
            "please",
            "thank",
            "assist",
            "understand",
            "check"
        ]

        if any(
            term in response_lower
            for term in helpful_terms
        ):
            score += 1
            reasons.append(
                "Response uses helpful/supportive language."
            )
        else:
            reasons.append(
                "Response may not be sufficiently helpful."
            )

        # -------------------------------------------------
        # 4. Avoid internal system details
        # -------------------------------------------------

        internal_terms = [
            "classifier",
            "retrieval system",
            "prompt",
            "internal process",
            "tf-idf",
            "model confidence"
        ]

        if not any(
            term in response_lower
            for term in internal_terms
        ):
            score += 1
            reasons.append(
                "No internal system details exposed."
            )
        else:
            reasons.append(
                "Response exposes internal system details."
            )

        # -------------------------------------------------
        # 5. Avoid fabricated specific information
        # -------------------------------------------------

        risky_phrases = [
            "your order number is",
            "your refund amount is",
            "your package will arrive on",
            "i can see your order",
            "i checked your account"
        ]

        if not any(
            phrase in response_lower
            for phrase in risky_phrases
        ):
            score += 1
            reasons.append(
                "No obvious fabricated customer-specific details."
            )
        else:
            reasons.append(
                "Response may contain fabricated customer-specific details."
            )

        return {
            "score": score,
            "max_score": self.max_score,
            "percentage": round(
                (score / self.max_score) * 100,
                2
            ),
            "reasons": reasons
        }


# ---------------------------------------------------------
# Example evaluation
# ---------------------------------------------------------

def main():

    print("=" * 80)
    print("RESPONSE QUALITY EVALUATION")
    print("=" * 80)

    judge = ResponseQualityJudge()

    examples = [
        {
            "customer": "My package is three days late",
            "response": (
                "I'm sorry your package is delayed. "
                "Please check the latest delivery information "
                "in your order details."
            )
        },
        {
            "customer": "Where is my refund?",
            "response": (
                "I understand you're waiting for your refund. "
                "Please check your refund status in your order details."
            )
        },
        {
            "customer": "I want to cancel my order",
            "response": (
                "I can help with that. "
                "Please check whether your order is still eligible "
                "for cancellation."
            )
        }
    ]

    results = []

    for example in examples:

        evaluation = judge.evaluate(
            example["customer"],
            example["response"]
        )

        result = {
            "customer": example["customer"],
            "response": example["response"],
            "evaluation": evaluation
        }

        results.append(result)

        print("\nCustomer:")
        print(example["customer"])

        print("\nResponse:")
        print(example["response"])

        print(
            f"\nScore: "
            f"{evaluation['score']}/"
            f"{evaluation['max_score']}"
        )

        print(
            f"Quality: "
            f"{evaluation['percentage']}%"
        )

    # -----------------------------------------------------
    # Average score
    # -----------------------------------------------------

    average_score = (
        sum(
            item["evaluation"]["score"]
            for item in results
        )
        / len(results)
    )

    average_percentage = round(
        (average_score / judge.max_score) * 100,
        2
    )

    final_results = {
        "examples_evaluated": len(results),
        "average_score": round(
            average_score,
            2
        ),
        "average_percentage": average_percentage,
        "results": results
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
            final_results,
            f,
            indent=4
        )

    print("\n" + "=" * 80)

    print(
        f"Average Quality: "
        f"{average_percentage}%"
    )

    print(
        "\n✓ Results saved to:"
    )

    print(OUTPUT_FILE)

    print("\n" + "=" * 80)
    print("RESPONSE QUALITY EVALUATION COMPLETE")
    print("=" * 80)


# ---------------------------------------------------------
# Entry point
# ---------------------------------------------------------

if __name__ == "__main__":
    main()
