class EscalationManager:

    def __init__(self, confidence_threshold=0.65):
        self.confidence_threshold = confidence_threshold

    def should_escalate(self, analysis):

        message = analysis["message"].lower()
        confidence = analysis["confidence"]
        intent = analysis["intent"]

        # 1. Low classifier confidence
        if confidence < self.confidence_threshold:
            return {
                "escalate": True,
                "reason": "Low intent classification confidence"
            }

        # 2. Sensitive / high-risk situations
        sensitive_keywords = [
            "fraud",
            "scam",
            "hacked",
            "stolen",
            "unauthorized",
            "chargeback",
            "legal",
            "lawyer",
            "police"
        ]

        for keyword in sensitive_keywords:
            if keyword in message:
                return {
                    "escalate": True,
                    "reason": f"Sensitive issue detected: {keyword}"
                }

        # 3. Account access problems
        if intent == "account_issue":
            account_keywords = [
                "hacked",
                "stolen",
                "unauthorized",
                "someone accessed"
            ]

            for keyword in account_keywords:
                if keyword in message:
                    return {
                        "escalate": True,
                        "reason": "Potential account security issue"
                    }

        # No escalation required
        return {
            "escalate": False,
            "reason": None
        }


if __name__ == "__main__":

    manager = EscalationManager()

    test_cases = [
        {
            "message": "My package is late",
            "intent": "delivery_delay",
            "confidence": 0.94
        },
        {
            "message": "I think someone hacked my account",
            "intent": "account_issue",
            "confidence": 0.91
        },
        {
            "message": "I don't know what happened",
            "intent": "account_issue",
            "confidence": 0.42
        }
    ]

    for case in test_cases:

        result = manager.should_escalate(case)

        print("\nCustomer:", case["message"])
        print("Escalate:", result["escalate"])
        print("Reason:", result["reason"])
