"""
Intent taxonomy for the AmazonHelp support agent.

These intents were derived from the initial exploration of
the AmazonHelp customer-support conversations.
"""

INTENTS = {
    "delivery_delay": {
        "description": "Customer reports that an order or package is arriving late or taking longer than expected.",
        "examples": [
            "My package is late",
            "My order has not arrived yet",
            "Why is my delivery delayed?"
        ]
    },

    "delivery_status": {
        "description": "Customer asks where an order is or wants an update on its delivery status.",
        "examples": [
            "Where is my package?",
            "Can you check my delivery?",
            "What is the status of my order?"
        ]
    },

    "delivery_date": {
        "description": "Customer asks about, disputes, or wants to change the expected delivery date.",
        "examples": [
            "When will my order arrive?",
            "Why did my delivery date change?",
            "Can I get it delivered earlier?"
        ]
    },

    "missing_package": {
        "description": "Customer says a package is missing or marked delivered but was not received.",
        "examples": [
            "My package says delivered but I don't have it",
            "My order is missing",
            "I never received my package"
        ]
    },

    "prime_issue": {
        "description": "Customer has a problem or question related to Amazon Prime membership or Prime service.",
        "examples": [
            "My Prime membership isn't working",
            "Why am I paying for Prime?",
            "I have a problem with Prime"
        ]
    },

    "product_issue": {
        "description": "Customer reports a problem with a purchased product, including damaged, defective, or incorrect items.",
        "examples": [
            "The product arrived broken",
            "I received the wrong item",
            "The item is defective"
        ]
    },

    "payment_issue": {
        "description": "Customer reports a payment, billing, charge, or payment-processing problem.",
        "examples": [
            "I was charged twice",
            "My payment failed",
            "Why was I charged?"
        ]
    },

    "account_issue": {
        "description": "Customer has trouble accessing, logging into, or managing their Amazon account.",
        "examples": [
            "I cannot login",
            "My account is locked",
            "I cannot access my account"
        ]
    },

    "refund_status": {
        "description": "Customer asks about a pending, missing, or delayed refund.",
        "examples": [
            "Where is my refund?",
            "I haven't received my refund",
            "When will my refund arrive?"
        ]
    },

    "return_issue": {
        "description": "Customer wants to return an item or has a problem with the return process.",
        "examples": [
            "I want to return this item",
            "How do I return my order?",
            "My return has a problem"
        ]
    },

    "order_cancellation": {
        "description": "Customer wants to cancel an order or is asking why an order was cancelled.",
        "examples": [
            "I want to cancel my order",
            "Please cancel my order",
            "Why was my order cancelled?"
        ]
    }
}


def get_intent_names():
    """Return all available intent names."""
    return list(INTENTS.keys())


def get_intent_description(intent_name):
    """Return the description of an intent."""
    return INTENTS[intent_name]["description"]
